import json
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Note: Dataset Code was partially inspired by: https://github.com/mkofinas/neural-graphs.git

# region: INRs

class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class INR_Network(nn.Module):
    def __init__(self, in_features=2, n_layers=3, hidden_features=32, out_features=1):
        super(INR_Network, self).__init__()
        self.seq = nn.ModuleList([nn.Linear(in_features, hidden_features)])
        for i in range(n_layers - 2):
            self.seq.append(nn.Linear(hidden_features, hidden_features))
        self.seq.append(nn.Linear(hidden_features, out_features))
        self.activation = Sine(w0=30.0)

    def make_coordinates(self, shape=(28, 28), bs=1, coord_range=(-1, 1)):
        x_coordinates = np.linspace(coord_range[0], coord_range[1], shape[0])
        y_coordinates = np.linspace(coord_range[0], coord_range[1], shape[1])
        x_coordinates, y_coordinates = np.meshgrid(x_coordinates, y_coordinates)
        x_coordinates = x_coordinates.flatten()
        y_coordinates = y_coordinates.flatten()
        coordinates = np.stack([x_coordinates, y_coordinates]).T
        coordinates = np.repeat(coordinates[np.newaxis, ...], bs, axis=0)
        return torch.from_numpy(coordinates).type(torch.float)

    def plot_INR_img(self):
        coords = self.make_coordinates(bs=1).to(self.seq[0].weight.device)
        with torch.no_grad():
            out = self.forward(coords)
        out = out.view(28, 28).cpu().numpy()
        return out

    def forward(self, x):
        for layer in self.seq[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.seq[-1](x)
        return x + 0.5

    def get_stats(self, acts, quantiles=[0., 0.25, 0.5, 0.75, 1.]):
        """
        activations: shape (bs, **)
        """
        feats = []
        flat_a = acts.flatten(start_dim=1)
        feats.append(flat_a.mean(dim=1))
        feats.append(flat_a.var(dim=1))
        for q in quantiles:
            feats.append(torch.quantile(flat_a, q, dim=1))
        feats = torch.stack(feats, dim=1)
        return feats

    def forward_and_extract_acts(self, x, max_size=3):
        if max_size >= len(self.seq):
            chosen_layers = list(range(len(self.seq)))
        else:
            chosen_layers = [int(l) for l in np.linspace(0, len(self.seq), max_size)]

        all_act_feats = []
        for i, layer in enumerate(self.seq):

            is_last_layer = i == len(self.seq) - 1
            x = layer(x)
            if is_last_layer:
                x = x + 0.5

            if i in chosen_layers:
                all_act_feats.append(self.get_stats(x))

            if not is_last_layer:
                x = self.activation(x)

        if max_size >= len(self.seq):
            zero_pad = torch.zeros(max_size - len(all_act_feats), all_act_feats[0].shape[1], device=all_act_feats[0].device)
            all_act_feats.append(zero_pad)

        all_act_feats = torch.cat(all_act_feats, dim=0)
        return x, all_act_feats

    def get_weights_stats(self, max_size=3):
        if max_size >= len(self.seq):
            chosen_layers = list(range(len(self.seq)))
        else:
            chosen_layers = [int(l) for l in np.linspace(0, len(self.seq), max_size)]
        all_w_stats = []
        all_layer_types = []
        all_act_types = []
        for i, layer in enumerate(self.seq):
            last_layer = i == len(self.seq) - 1
            if i in chosen_layers:
                layer_stats = torch.cat([self.get_stats(layer.weight.unsqueeze(0)),
                                         self.get_stats(layer.bias.unsqueeze(0))], dim=1)
                all_w_stats.append(layer_stats)
                all_layer_types.append(type(layer).__name__)
                all_act_types.append(type(self.activation).__name__ if not last_layer else 'none')
        if max_size >= len(self.seq) + 1:
            zero_pad = torch.zeros(max_size - len(all_w_stats), all_w_stats[0].shape[1], device=all_w_stats[0].device)
            all_w_stats.append(zero_pad)
            all_layer_types.extend(['none'] * (max_size - len(all_layer_types)))
            all_act_types.extend(['none'] * (max_size - len(all_act_types)))
        all_w_stats = torch.cat(all_w_stats, dim=0)
        return all_w_stats, all_layer_types, all_act_types


class INRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, splits_path, split="train"):
        self.split = split
        self.splits_path = (
            (Path(dataset_dir) / Path(splits_path)).expanduser().resolve()
        )
        self.root = self.splits_path.parent
        with self.splits_path.open("r") as f:
            self.dataset = json.load(f)[self.split]
        self.dataset["path"] = [
            Path(dataset_dir) / Path(p) for p in self.dataset["path"]
        ]

        self.all_data = [None for _ in range(len(self.dataset["label"]))]

    def __len__(self):
        return len(self.dataset["label"])

    def n_classes(self):
        return len(set(self.dataset["label"]))

    def __getitem__(self, item):
        if self.all_data[item] is None:
            path = str(self.dataset["path"][item])
            try:
                state_dict = torch.load(path, map_location='cpu')
            except Exception as e:
                print(f"Failed to load {path}")
                raise e
            if "label" in state_dict.keys():
                state_dict.pop("label")
            assert "label" not in state_dict.keys()
            label = int(self.dataset["label"][item])
            model = INR_Network()
            model.load_state_dict(state_dict)
            self.all_data[item] = (model, label)

        model, label = self.all_data[item]
        return model, label


# endregion: INRs


# region: CNN datasets

class Generic_CNN_Network(nn.Module):
    def __init__(self, n_layers, channels, kernel_sizes, strides, activations, paddings, out_size=10):
        super(Generic_CNN_Network, self).__init__()
        self.layers = nn.ModuleList([nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i]) for i in range(n_layers)])
        self.activations = nn.ModuleList([self.get_act(act) for act in activations])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(channels[-1], out_size)

    def load_weights(self, weights, biases):
        for layer, w, b in zip(self.layers, weights[:-1], biases[:-1]):
            assert layer.weight.data.shape == w.shape
            assert layer.bias.data.shape == b.shape
            layer.weight.data = w.clone()
            layer.bias.data = b.clone()
        self.fc.weight.data = weights[-1].clone()
        self.fc.bias.data = biases[-1].clone()

    def get_act(self, act_type):
        if act_type == 'relu':
            return nn.ReLU()
        elif act_type == 'gelu':
            return nn.GELU()
        elif act_type == 'sine':
            return Sine(w0=30.0)
        elif act_type == 'tanh':
            return nn.Tanh()
        elif act_type == 'sigmoid':
            return nn.Sigmoid()
        elif act_type == 'leaky_relu':
            return nn.LeakyReLU()
        elif act_type == 'none':
            return nn.Identity()
        else:
            raise ValueError(f"Activation type {act_type} not recognized.")

    def get_stats(self, acts, quantiles=[0., 0.25, 0.5, 0.75, 1.]):
        """
        activations: shape (bs, **)
        """
        feats = []
        flat_a = acts.flatten(start_dim=1)
        feats.append(flat_a.mean(dim=1))
        feats.append(flat_a.var(dim=1))
        for q in quantiles:
            feats.append(torch.quantile(flat_a, q, dim=1))
        feats = torch.stack(feats, dim=1)
        return feats

    def forward_and_extract_acts(self, x, max_size=3):
        chosen_layers = [int(l) for l in np.linspace(0, len(self.layers), max_size)]
        all_act_feats = [self.get_stats(x)]
        for i, (layer, act) in enumerate(zip(self.layers, self.activations)):
            x = layer(x)
            if i in chosen_layers:
                all_act_feats.append(x.mean(dim=(2, 3)))
            x = act(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        all_act_feats.append(x)
        # print([a.shape for a in all_act_feats])
        all_act_feats = torch.cat(all_act_feats, dim=1)
        return x, all_act_feats

    def get_weights_stats(self, max_size=6):
        if max_size >= len(self.layers) + 1:
            chosen_layers = list(range(len(self.layers) + 1))
        else:
            chosen_layers = [int(l) for l in np.linspace(0, len(self.layers) + 1, max_size)]
        all_w_stats = []
        all_layer_types = []
        all_act_types = []
        for i, (layer, act) in enumerate(zip(self.layers, self.activations)):
            if i in chosen_layers:
                layer_stats = torch.cat([self.get_stats(layer.weight.unsqueeze(0)),
                                         self.get_stats(layer.bias.unsqueeze(0))], dim=1)
                all_w_stats.append(layer_stats)
                all_layer_types.append(type(layer).__name__)
                all_act_types.append(type(act).__name__)
        assert len(self.layers) in chosen_layers
        layer_stats = torch.cat([self.get_stats(self.fc.weight.unsqueeze(0)),
                                 self.get_stats(self.fc.bias.unsqueeze(0))], dim=1)
        all_w_stats.append(layer_stats)
        all_layer_types.append(type(self.fc).__name__)
        all_act_types.append('none')
        if max_size >= len(self.layers) + 1:
            zero_pad = torch.zeros(max_size - len(all_w_stats), all_w_stats[0].shape[1], device=all_w_stats[0].device)
            all_w_stats.append(zero_pad)
            all_layer_types.extend(['none'] * (max_size - len(all_layer_types)))
            all_act_types.extend(['none'] * (max_size - len(all_act_types)))
        all_w_stats = torch.cat(all_w_stats, dim=0)
        return all_w_stats, all_layer_types, all_act_types

    def forward(self, x):
        for layer, act in zip(self.layers, self.activations):
            x = act(layer(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class NFNZooDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, split, idcs_file=None):
        data = np.load(os.path.join(data_path, "weights.npy"))
        # Hardcoded shuffle order for consistent test set.
        shuffled_idcs = pd.read_csv(idcs_file, header=None).values.flatten()
        data = data[shuffled_idcs]
        metrics = pd.read_csv(
            os.path.join(data_path, "metrics.csv.gz"), compression="gzip"
        )
        metrics['generalization'] = metrics['test_accuracy'] - metrics['train_accuracy']
        metrics = metrics.iloc[shuffled_idcs]
        self.layout = pd.read_csv(os.path.join(data_path, "layout.csv"))
        # filter to final-stage weights ("step" == 86 in metrics)
        isfinal = metrics["step"] == 86
        metrics = metrics[isfinal]
        data = data[isfinal]
        assert np.isfinite(data).all()

        metrics.index = np.arange(0, len(metrics))
        idcs = self._split_indices_iid(data)[split]
        data = data[idcs]
        self.metrics = metrics.iloc[idcs]
        self.metrics['sample_id'] = np.arange(len(self.metrics))
        self.metrics['chosen_label'] = self.metrics['test_accuracy']
        self.weights, self.biases = [], []
        for i, row in self.layout.iterrows():
            arr = data[:, row["start_idx"]: row["end_idx"]]
            bs = arr.shape[0]
            arr = arr.reshape((bs, *eval(row["shape"])))
            if row["varname"].endswith("kernel:0"):
                # tf to pytorch ordering
                if arr.ndim == 5:
                    arr = arr.transpose(0, 4, 3, 1, 2)
                elif arr.ndim == 3:
                    arr = arr.transpose(0, 2, 1)
                self.weights.append(arr)
            elif row["varname"].endswith("bias:0"):
                self.biases.append(arr)
            else:
                raise ValueError(f"varname {row['varname']} not recognized.")

        self.model_config = {}
        self.model_config['n_layers'] = 3
        self.model_config['strides'] = [2, 2, 2]
        self.model_config['kernel_sizes'] = [(3, 3), (3, 3), (3, 3)]
        self.model_config['channels'] = [1, 16, 16, 16]
        self.model_config['paddings'] = [1, 1, 1]

    def _split_indices_iid(self, data):
        splits = {}
        test_split_point = int(0.5 * len(data))
        splits["test"] = list(range(test_split_point, len(data)))

        trainval_idcs = list(range(test_split_point))
        val_point = int(0.8 * len(trainval_idcs))
        # use local seed to ensure consistent train/val split
        rng = random.Random(0)
        rng.shuffle(trainval_idcs)
        splits["train"] = trainval_idcs[:val_point]
        splits["val"] = trainval_idcs[val_point:]
        return splits

    def __len__(self):
        return self.weights[0].shape[0]

    def __getitem__(self, idx):
        weights = [torch.from_numpy(w[idx]) for w in self.weights]
        biases = [torch.from_numpy(b[idx]) for b in self.biases]

        activations = [self.metrics.iloc[idx]["config.activation"]] * 3
        model = Generic_CNN_Network(**self.model_config, activations=activations, out_size=10)
        model.load_weights(weights, biases)

        score = self.metrics.iloc[idx]['chosen_label']
        return model, score


class CNN_Park_ModelData(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, splits_path, split="train"):
        self.split = split
        self.splits_path = os.path.join(dataset_dir, splits_path)
        with open(self.splits_path, "r") as f:
            self.dataset = json.load(f)[self.split]
        self.dataset["path"] = [os.path.join(dataset_dir, 'cnn_wild', p) for p in self.dataset["path"]]
        self.all_data = [None for _ in range(len(self.dataset["score"]))]

    def __len__(self):
        return len(self.dataset["path"])

    def item_first_load(self, item):

        path = self.dataset["path"][item]
        with open(path, 'rb') as f:
            model_obj = torch.load(f, map_location='cpu')
        state_dict = model_obj["model"]

        label = self.dataset["score"][item]

        strides = model_obj['config']['stride']
        activations = model_obj['config']['activation']
        kernel_sizes = model_obj['config']['kernel_size']
        channels = model_obj['config']['channels']
        paddings = model_obj['config']['padding']
        n_layers = model_obj['config']['n_layers']
        config = {"n_layers": n_layers, "channels": channels, "kernel_sizes": kernel_sizes, "strides": strides,
                  "activations": activations, "paddings": paddings, "out_size": 10}

        self.all_data[item] = [state_dict, config, label]

    def __getitem__(self, item):
        if self.all_data[item] is None:
            self.item_first_load(item)
        sd, cfg, label = self.all_data[item]
        model = Generic_CNN_Network(**cfg)
        model.load_state_dict(sd)
        return model, label


# endregion: CNN datasets

