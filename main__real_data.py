import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import kendalltau
from data import INRDataset, NFNZooDataset, CNN_Park_ModelData
from ProbeGen import ProbeGen, Probing_Gen_Adapter


def set_seed(seed):
    """for reproducibility
    :param seed: seed value
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# region: setup

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dataset", type=str, default='mnist_inr',
                    choices=['mnist_inr', 'fmnist_inr', 'nfn_cnn_zoo', 'cnn_park'])

# architecture
parser.add_argument("--n_tokens", type=int, default=64)
parser.add_argument("--d_hid", type=int, default=64)
parser.add_argument("--n_layers", type=int, default=6)

# model
parser.add_argument("--gen_type", type=str, default="dead_leaves__no_opt")

# optimization
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--eval_every", type=int, default=500)
parser.add_argument("--n_workers", type=int, default=4)
parser.add_argument("--device", type=str, default="cuda")

args = parser.parse_args()

set_seed(args.seed)

torch.multiprocessing.set_sharing_strategy('file_system')

device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
exp_dir = f"experiments/{args.dataset}/runs/{args.exp_name}"
os.makedirs(exp_dir, exist_ok=True)

# Remove:
DATA_ROOT_PATH = '/cs/labs/yedid/jonkahana/projects/weight_classification/neural-graphs'

# save args in exp dir
with open(f"{exp_dir}/args.txt", "w") as f:
    for k, v in vars(args).items():
        f.write(f"{k}: {v}\n")

device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")


# endregion: setup


# region: data

def collate_fn(batch):
    return [batch[i][0] for i in range(len(batch))], torch.tensor([batch[i][1] for i in range(len(batch))])


if args.dataset in ['mnist_inr', 'fmnist_inr']:
    dataset_dir = os.path.join(DATA_ROOT_PATH, "experiments/inr_classification/dataset")
    splits_path = "mnist_splits.json" if args.dataset == 'mnist_inr' else "fmnist_splits.json"
    train_set = INRDataset(dataset_dir=dataset_dir, splits_path=splits_path, split="train")
    val_set = INRDataset(dataset_dir=dataset_dir, splits_path=splits_path, split="val")
    test_set = INRDataset(dataset_dir=dataset_dir, splits_path=splits_path, split="test")
    d_out = train_set.n_classes()
    models_c_out = 1
    models_c_in = 2
    is_regr = False
elif args.dataset == 'nfn_cnn_zoo':
    base_dataset_dir = os.path.join(DATA_ROOT_PATH, "experiments/cnn_generalization/dataset")
    dataset_dir = os.path.join(base_dataset_dir, "small-zoo-cifar10")
    splits_path = os.path.join(base_dataset_dir, "nfn_cifar10_split.csv")
    train_set = NFNZooDataset(data_path=dataset_dir, idcs_file=splits_path, split="train")
    val_set = NFNZooDataset(data_path=dataset_dir, idcs_file=splits_path, split="val")
    test_set = NFNZooDataset(data_path=dataset_dir, idcs_file=splits_path, split="test")
    d_out = 1
    models_c_out = 10
    models_c_in = 1
    is_regr = True
elif args.dataset == 'cnn_park':
    dataset_dir = os.path.join(DATA_ROOT_PATH, "experiments/cnn_generalization/dataset")
    splits_path = "cnn_park_splits.json"
    train_set = CNN_Park_ModelData(dataset_dir=dataset_dir, splits_path=splits_path, split="train")
    val_set = CNN_Park_ModelData(dataset_dir=dataset_dir, splits_path=splits_path, split="val")
    test_set = CNN_Park_ModelData(dataset_dir=dataset_dir, splits_path=splits_path, split="test")
    d_out = 1
    models_c_out = 10
    models_c_in = 3
    is_regr = True
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")

print(f"Train set: {len(train_set)}, Val set: {len(val_set)}, Test set: {len(test_set)}")
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, pin_memory=False, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=False, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=False, collate_fn=collate_fn)

# endregion: data


# region: model

model = ProbeGen(n_tokens=args.n_tokens, d_hidden=args.d_hid, models_c_in=models_c_in, models_c_out=models_c_out,
                 d_out=d_out, gen_type=args.gen_type,
                 gen_width=None, gen_latent_z=None, n_layers=args.n_layers)

model = model.float()
model = model.to(device)

adapter = Probing_Gen_Adapter(n_tokens=args.n_tokens, d_hidden=args.d_hid, models_c_out=models_c_out, d_out=d_out, n_layers=args.n_layers)
adapter = adapter.float()
adapter = adapter.to(device)

# endregion: model


# region: gather prev representations

@torch.no_grad()
def get_repr_data(model, loader, device):
    model.eval()
    reprs, labels = [], []
    for i, batch in tqdm(enumerate(loader), total=len(loader), ncols=150):

        label = batch[1].to(device)
        [x.to(device) for x in batch[0]]
        inputs = {'nets': batch[0]}
        out = model.forward_generator(**inputs)

        reprs.append(out.cpu())
        labels.append(label.cpu())

        [x.to('cpu') for x in batch[0]]
        [x.zero_grad() for x in batch[0]]

    reprs = torch.cat(reprs, dim=0)
    labels = torch.cat(labels, dim=0)
    dataset = torch.utils.data.TensorDataset(reprs, labels)
    return dataset

adapter_train_set = get_repr_data(model, train_loader, device)
adapter_val_set = get_repr_data(model, val_loader, device)
adapter_test_set = get_repr_data(model, test_loader, device)

adapter_train_loader = torch.utils.data.DataLoader(dataset=adapter_train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, pin_memory=True)
adapter_val_loader = torch.utils.data.DataLoader(dataset=adapter_val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=True)
adapter_test_loader = torch.utils.data.DataLoader(dataset=adapter_test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=True)


# region: training

@torch.no_grad()
def evaluate(adapter_model, adapter_loader, device):
    adapter_model.eval()
    loss = 0.0
    correct = 0.0
    total = 0.0
    predicted, gt = [], []
    for i, (x, y) in enumerate(adapter_loader):

        x = x.to(device)
        y = y.to(device)
        out = adapter_model(x)

        if is_regr:
            y = y.float()
            y = y.unsqueeze(1)
            loss += F.mse_loss(out, y, reduction="sum")
            predicted.extend(out.flatten().cpu().numpy().tolist())
        else:
            loss += F.cross_entropy(out, y, reduction="sum")
            pred = out.argmax(1)
            correct += pred.eq(y).sum()
            predicted.extend(pred.cpu().numpy().tolist())
        total += len(y)
        gt.extend(y.flatten().cpu().numpy().tolist())

    predicted = np.array(predicted)
    gt = np.array(gt)

    adapter_model.train()
    avg_loss = loss / total

    res_d = dict(avg_loss=avg_loss.item(), predicted=predicted, gt=gt)
    if is_regr:
        k_tau = kendalltau(predicted, gt).statistic
        res_d['kendalltau'] = k_tau
    else:
        avg_acc = correct / total
        avg_acc = avg_acc.item()
        res_d['avg_acc'] = avg_acc
    return res_d


optimizer = torch.optim.Adam(params=adapter.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(adapter_train_loader))

criterion = nn.MSELoss() if is_regr else nn.CrossEntropyLoss()

# endregion: optimizer, scheduler, criterion

global_step = 0
logging = pd.DataFrame(columns=['exp_name', 'epoch', 'global_step', 'train_loss', 'val_loss', 'val_acc', 'test_acc', 'val_kendalltau', 'test_kendalltau'])
epoch_iter = tqdm(range(args.epochs), ncols=150)

iters_train_loss = 0.0
for epoch in epoch_iter:

    for i, (x, y) in enumerate(adapter_train_loader):
        adapter.train()
        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)
        out = adapter(x)

        if is_regr:
            y = y.float()
            y = y.unsqueeze(1)

        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        iters_train_loss += loss.item()

        if global_step % args.eval_every == 0 and global_step > 0:
            adapter.eval()
            val_results_dict = evaluate(adapter, adapter_val_loader, device)
            test_results_dict = evaluate(adapter, adapter_test_loader, device)
            adapter.train()
            val_loss = val_results_dict["avg_loss"]
            test_loss = test_results_dict["avg_loss"]
            val_main_metric = val_results_dict["avg_acc"] if not is_regr else val_results_dict["kendalltau"]
            test_main_metric = test_results_dict["avg_acc"] if not is_regr else test_results_dict["kendalltau"]
            train_loss = iters_train_loss / args.eval_every

            log_row_df = {'exp_name': args.exp_name, 'epoch': epoch, 'global_step': global_step,
                          'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss}
            if is_regr:
                log_row_df.update({'val_kendalltau': val_main_metric, 'test_kendalltau': test_main_metric})
            else:
                log_row_df.update({'val_acc': val_main_metric, 'test_acc': test_main_metric})

            log_row_df = pd.DataFrame(log_row_df, index=[0])
            logging = pd.concat([logging, log_row_df], ignore_index=True)
            logging.to_csv(f"{exp_dir}/log.csv", index=False)

            iters_train_loss = 0.0
            torch.save(adapter.state_dict(), os.path.join(exp_dir, 'intermediate_checkpoint.pth'))

        global_step += 1

# endregion: training

torch.save(adapter.state_dict(), os.path.join(exp_dir, 'epoch_last.pth'))
