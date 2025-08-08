from trackastra.data import (
    # load_ctc_data_from_subfolders,
    BalancedBatchSampler,
    BalancedDataModule,
    CTCData,
    collate_sequence_padding,
)
from trackastra.utils import (
    blockwise_causal_norm,
    blockwise_sum,
    normalize,
    preallocate_memory,
    random_label_cmap,
    render_label,
    seed,
    str2bool,
)
from pathlib import Path
from trackastra.model import TrackingTransformer
import torch
import random
from torch.utils.data import DataLoader
from aim import Run
from train import WarmupCosineLRScheduler
import numpy as np

import sys

run = Run(experiment='doesitscale-multi-lr-savemodel-redo')

import argparse

parser = argparse.ArgumentParser(description="Set hyperparameters for training")

parser.add_argument('--ssl_fraction', type=float, required=True, help='Fraction of SSL data to use')
parser.add_argument('--ssl_size', type=int, required=True, help='Total size of SSL data')
parser.add_argument('--ssl_lr', type=float, required=True, help='Learning rate for self-supervised optimizer', default=3e-4)

args = parser.parse_args()

run['hparams'] = vars(args)

#run['hparams'] = {"train_scale": int(sys.argv[1]), "ssl_scale": int(sys.argv[2]), "ssl_fraction": float(sys.argv[3]), 'ssl_size': int(sys.argv[4])}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

default_ctcdata_kwargs = {
    "ndim": 2,
    "detection_folders": ["TRA"],
    "window_size": 10,
    "max_tokens": None,
    "augment": True,                # args.augment (train split)
    "features": "wrfeat",
    "slice_pct": (0.0, 1.0),     # args.slice_pct_train
    "downscale_temporal": 1,
    "downscale_spatial": 1,
    "sanity_dist": False,
    "crop_size": None,           # args.crop_size (train split)
    "compress": False,
}

import sys

#if sys.argv[3] is None:
#    ssl_fraction = 0.5

#else:
#    ssl_fraction = float(sys.argv[3])

#if sys.argv[4] is None:
#    ssl_size = 1
#else:
#    ssl_size = int(sys.argv[4])

#print("parameters", train_dataset_size, ssl_size, ssl_fraction, ssl_size)

train_data = CTCData(
    root=Path(f'./realhela/train/01/'),
    **default_ctcdata_kwargs
)

#ssl_train_data = CTCData(
#    root=Path(f'/mnt/lustre-grete/usr/u14148/MASTER_THESIS/fluo_splitter/splits/Fluo_13_{ssl_size}/02/'),
#    **default_ctcdata_kwargs
#)

from torch.utils.data import ConcatDataset, DataLoader

from pathlib import Path
from torch.utils.data import ConcatDataset

def load_all_bursts(root_dir: Path, ssl_size: int | None = None):
    """
    Discover all burst subdatasets under `root_dir`, instantiate CTCData for each,
    and return a single concatenated dataset. If ssl_size is given, only the first
    ssl_size bursts (sorted alphabetically) are used.
    """
    root_dir = Path(root_dir)
    burst_dirs = []

    for entry in sorted(root_dir.iterdir()):
        if not entry.is_dir():
            continue
        # Some bursts have a nested folder with the same name
        nested = entry / entry.name
        if nested.is_dir():
            burst_root = nested
        else:
            burst_root = entry
        burst_dirs.append(burst_root)

    if ssl_size is not None:
        burst_dirs = burst_dirs[:ssl_size]

    if not burst_dirs:
        raise RuntimeError(f"No burst directories found under {root_dir!r}")

    datasets = []
    for bd in burst_dirs:
        datasets.append(
            CTCData(
                root=bd,
                **default_ctcdata_kwargs
            )
        )

    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


# Usage: load everything
ssl_train_data = load_all_bursts(
    Path('/mnt/lustre-grete/usr/u14148/HeLa_dataset_full/CTCtrain/'),
    ssl_size=args.ssl_size  # or e.g. 2 to mimic previous behavior
)

"""
ssl_train_data1 = CTCData(
    root=Path(f'/mnt/lustre-grete/usr/u14148/HeLa_dataset_full/CTCtrain/Burst1_A1_1_VesselID-29_0-1/Burst1_A1_1_VesselID-29_0-1/'),
    **default_ctcdata_kwargs
)

_ = "/mnt/lustre-grete/usr/u14148/HeLa_dataset_full/CTCtrain/Burst1_A2_1_VesselID-29_1-2/Burst1_A2_1_VesselID-29_1-2"

ssl_train_data2 = CTCData(
    root=Path(f'/mnt/lustre-grete/usr/u14148/HeLa_dataset_full/CTCtrain/Burst1_A2_1_VesselID-29_1-2/Burst1_A2_1_VesselID-29_1-2'),
    **default_ctcdata_kwargs
)


if ssl_size == 2:
    print("doing 2 datasets")
    ssl_train_data = ConcatDataset([ssl_train_data1, ssl_train_data2])
else:
    print("single dataset")
    ssl_train_data = ssl_train_data1
"""

val_data = CTCData(
    root=Path('./realhela/train/02/'),
    **default_ctcdata_kwargs
)


default_transformer_kwargs = {
    "d_model": 256,
    "pos_embed_per_dim": 32,
    "feat_embed_per_dim": 8,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "dropout": 0.0,
    "window": 10,
    "spatial_pos_cutoff": 256,
    "attn_positional_bias": "rope",
    "attn_positional_bias_n_spatial": 16,
    "causal_norm": "quiet_softmax",
}

div_upweight = 2
delta_cutoff = 2
eps=torch.finfo(torch.float32).eps

# and you pull coord_dim/feat_dim straight off your dataset:
model = TrackingTransformer(
    coord_dim=train_data.ndim,
    feat_dim=train_data.feat_dim,
    **default_transformer_kwargs
)

model.to(device)



print("-"*100)

do_ssl = False

criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
criterion_softmax = torch.nn.BCELoss(reduction="none")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
optimizer_ssl = torch.optim.SGD(model.parameters(), lr=args.ssl_lr)

lr_scheduler=WarmupCosineLRScheduler(
                optimizer, 10, 100
            )

common_loader_args = dict(
        persistent_workers=False,
        pin_memory=True,
        collate_fn=collate_sequence_padding(),
    )

tasksampler = torch.utils.data.RandomSampler(
                train_data,
                num_samples=10000,
                replacement=True,
            )
sslsampler = torch.utils.data.RandomSampler(
                ssl_train_data,
                num_samples=10000,
                replacement=True,
            )

valsampler = torch.utils.data.RandomSampler(
                val_data,
                num_samples=10000,
                replacement=True,
            )

loader_train = DataLoader(
                train_data,
                sampler=tasksampler,
                batch_size=8,
                num_workers=1,
                **common_loader_args,
            )

ssl_loader_train = DataLoader(
                ssl_train_data,
                sampler=sslsampler,
                batch_size=8,
                num_workers=1,
                **common_loader_args,
            )


loader_val = DataLoader(
                val_data,
                sampler=valsampler,
                batch_size=8,
                num_workers=1,
                **common_loader_args,
            )



def do_training_step(batch, do_ssl=False):
    if True:
        feats = batch["features"].to(device)
        coords = batch["coords"].to(device)
        A = batch["assoc_matrix"].to(device)
        timepoints = batch["timepoints"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        padding_mask = padding_mask.bool()

        print("feats shape", feats.shape)

        if do_ssl:
            print("DOING SSL!")
            preserved_feats = feats.clone()
            _, N, _ = feats.shape
            masked = [random.uniform(0,1) > 0.3 for x in range(N)]
            feats[:, masked, :] = model.MASKED_TOKEN_INDICATOR

            a, b = model(coords, feats, padding_mask=padding_mask, do_ssl=True)

            loss = torch.nn.MSELoss()(a[:, masked], preserved_feats[:, masked])

            print(a.shape, b.shape)
            # this used to be return
            print({"loss": loss, "padding_fraction": 0})
            return loss



        A_pred = model(coords, feats, padding_mask=padding_mask)
        # remove inf values that might happen due to float16 numerics
        A_pred.clamp_(torch.finfo(torch.float16).min, torch.finfo(torch.float16).max)

        mask_invalid = torch.logical_or(
            padding_mask.unsqueeze(1), padding_mask.unsqueeze(2)
        )

        A_pred[mask_invalid] = 0
        loss = criterion(A_pred, A)

        if default_transformer_kwargs["causal_norm"] != "none":
            # TODO speedup: I could softmax only the part of the matrix (upper triangular) that is not masked out
            A_pred_soft = torch.stack(
                [
                    blockwise_causal_norm(
                        _A, _t, mode=default_transformer_kwargs["causal_norm"], mask_invalid=_m
                    )
                    for _A, _t, _m in zip(A_pred, timepoints, mask_invalid)
                ]
            )
            with torch.cuda.amp.autocast(enabled=False):
                if len(A) > 0:
                    # debug
                    if torch.any(torch.isnan(A_pred_soft)):
                        print(A_pred)
                        print(
                            "AAAA pred",
                            A_pred_soft.min().item(),
                            A_pred_soft.max().item(),
                        )
                        print("AAAA pred", A_pred_soft.shape)
                        print("AAAA pred", A_pred_soft.dtype)
                        print("AAAA", A.min().item(), A.max().item())
                        print("AAAA", A.shape)
                        print("AAAA", A.dtype)
                        print("A_pred_soft has nan")
                        np.savez(
                            "runs/nan.npz",
                            A_pred=A_pred.detach().cpu().numpy(),
                            timepoints=timepoints.detach().cpu().numpy(),
                        )

                # Keep the non-softmaxed loss for numerical stability
                loss = 0.01 * loss + criterion_softmax(A_pred_soft, A)

        # Reweighting does not need gradients
        with torch.no_grad():
            block_sum1 = torch.stack(
                [blockwise_sum(A, t, dim=-1) for A, t in zip(A, timepoints)], 0
            )
            block_sum2 = torch.stack(
                [blockwise_sum(A, t, dim=-2) for A, t in zip(A, timepoints)], 0
            )
            block_sum = A * (block_sum1 + block_sum2)

            normal_tracks = block_sum == 2
            division_tracks = block_sum > 2

            # upweight normal (not starting or ending) tracks and division tracks
            loss_weight = 1 + 1.0 * normal_tracks + div_upweight * division_tracks

        loss = loss * loss_weight

        mask_valid = ~mask_invalid
        dt = timepoints.unsqueeze(1) - timepoints.unsqueeze(2)
        mask_time = torch.logical_and(dt > 0, dt <= delta_cutoff)

        mask = mask_time * mask_valid
        mask = mask.float()

        loss_before_reduce = loss * mask
        # Normalized by number of valid entry for each sample
        # Here I get a loss that is normalized by the number of connections to predict
        loss_normalized = loss_before_reduce / (
            mask.sum(dim=(1, 2), keepdim=True) + eps
        )
        loss_per_sample = loss_normalized.sum(dim=(1, 2))

        # Hack: weight larger samples a little more...
        prefactor = torch.pow(mask.sum(dim=(1, 2)), 0.2)

        loss = loss_per_sample * prefactor / (prefactor.sum() + eps)
        loss = loss.sum()

        return loss



i = 0

best_val_loss = float('inf')
model_save_path = f"models/{args.ssl_size}_{args.ssl_fraction}_{args.ssl_lr}/"
Path("models").mkdir(exist_ok=True)
Path(model_save_path).mkdir(exist_ok=True)


for epoch in range(30):
    
    if epoch == 100:
        print("resetting lr scheduler")
        lr_scheduler=WarmupCosineLRScheduler(
                optimizer, 10, 100
            )

    model.train()
    for batch, ssl_batch in zip(loader_train, ssl_loader_train):

        optimizer.zero_grad()
        optimizer_ssl.zero_grad()

        if random.random() < args.ssl_fraction and epoch < 100:
            print("doing self supervision")
            loss = do_training_step(ssl_batch, True)
            run.track(loss.item(), name='train_loss_ssl', step=i, epoch=epoch, context={ "subset":"train" })

            loss.backward()
            optimizer_ssl.step()
        else:
            print("doing regular train")
            loss = do_training_step(batch)
            run.track(loss.item(), name='train_loss', step=i, epoch=epoch, context={ "subset":"train" })
    
            loss.backward()
            optimizer.step()

        #run.track(loss.item(), name='train_loss', step=i, epoch=epoch, context={ "subset":"train" })
        #print(loss.item())
        #loss.backward()
        #optimizer.step()

        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        run.track(current_lr, name='learning_rate', step=i, epoch=epoch)
        i+=1

    model.eval()    
    with torch.no_grad():
        losses = []
        for batch in loader_val:
            loss = do_training_step(batch)
            losses.append(loss.item())

        run.track(sum(losses) / len(losses), name='val_loss', step=i, epoch=epoch, context={ "subset":"val" })
        avg_val_loss = sum(losses) / len(losses)

    # Save model if it's the best so far
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        #torch.save(model.state_dict(), model_save_path)
        model.save(model_save_path)
        print(f"New best val loss: {best_val_loss:.4f} - model saved to {model_save_path}")


