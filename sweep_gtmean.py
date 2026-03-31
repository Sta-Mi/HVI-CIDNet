import argparse
import csv
import glob
import os
import re
import shutil

import torch
from torch.utils.data import DataLoader

from data.data import get_eval_set
from eval import eval as run_eval
from measure import metrics
from net.CIDNet import CIDNet


def parse_epoch(weight_path):
    name = os.path.basename(weight_path)
    match = re.search(r"epoch_(\d+)\.pth$", name)
    if match:
        return int(match.group(1))
    return -1


def prepare_lol_eval_data(num_workers=1):
    eval_data = DataLoader(
        dataset=get_eval_set("./datasets/LOLdataset/eval15/low"),
        num_workers=num_workers,
        batch_size=1,
        shuffle=False,
    )
    label_dir = "./datasets/LOLdataset/eval15/high/"
    return eval_data, label_dir


def main():
    parser = argparse.ArgumentParser(description="Sweep all weights and measure GT_mean metrics.")
    parser.add_argument("--dataset", type=str, default="lol_v1")
    parser.add_argument("--weights_glob", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default="./results/gtmean_sweep_lolv1.csv")
    parser.add_argument("--output_dir", type=str, default="./output/LOLv1_sweep/")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()
    if args.weights_glob is None:
        args.weights_glob = os.path.join("./weights/train", args.dataset, "epoch_*.pth")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    weight_paths = sorted(glob.glob(args.weights_glob), key=parse_epoch)
    if not weight_paths:
        raise FileNotFoundError(f"No checkpoints matched: {args.weights_glob}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for eval/measure in this repo.")

    eval_data, label_dir = prepare_lol_eval_data(num_workers=args.num_workers)
    rows = []

    for weight_path in weight_paths:
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)

        model = CIDNet().cuda()
        run_eval(
            model,
            eval_data,
            weight_path,
            args.output_dir,
            norm_size=True,
            LOL=True,
            v2=False,
            unpaired=False,
            alpha=1.0,
            gamma=args.gamma,
        )

        psnr, ssim, lpips = metrics(os.path.join(args.output_dir, "*.png"), label_dir, use_GT_mean=True)
        rows.append(
            {
                "weight": weight_path,
                "epoch": parse_epoch(weight_path),
                "psnr": float(psnr),
                "ssim": float(ssim),
                "lpips": float(lpips),
            }
        )
        print(
            f"[GT_mean] {os.path.basename(weight_path)} | "
            f"PSNR={psnr:.4f} SSIM={ssim:.4f} LPIPS={lpips:.4f}"
        )

    rows_sorted = sorted(rows, key=lambda x: (-x["psnr"], -x["ssim"], x["lpips"]))
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["weight", "epoch", "psnr", "ssim", "lpips"])
        writer.writeheader()
        writer.writerows(rows_sorted)

    print("\n=== Top checkpoints by GT_mean (PSNR desc, SSIM desc, LPIPS asc) ===")
    for row in rows_sorted[: args.topk]:
        print(
            f"epoch={row['epoch']:>4} | PSNR={row['psnr']:.4f} | "
            f"SSIM={row['ssim']:.4f} | LPIPS={row['lpips']:.4f} | {row['weight']}"
        )
    print(f"\nSaved full ranking to: {args.output_csv}")


if __name__ == "__main__":
    main()