import argparse
import os
from typing import List, Optional, Tuple

import gradio as gr
import imquality.brisque as brisque
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from loss.niqe_utils import calculate_niqe
from net.CIDNet import CIDNet


def parse_args():
    parser = argparse.ArgumentParser(description="IADNet Ablation Demo")
    parser.add_argument("--cpu", action="store_true", help="Run inference on CPU only")
    parser.add_argument("--server_port", type=int, default=7863, help="Gradio server port")
    parser.add_argument("--weights_dir", type=str, default="weights", help="Weights directory")
    return parser.parse_args()


def find_pth_files(directory: str) -> List[str]:
    pth_files = []
    for root, _, files in os.walk(directory):
        if "train" in root.split(os.sep):
            continue
        for file in files:
            if file.endswith(".pth"):
                rel_path = os.path.relpath(os.path.join(root, file), directory)
                pth_files.append(rel_path)
    pth_files.sort()
    return pth_files


class Enhancer:
    def __init__(self, weights_dir: str, use_cpu: bool):
        self.weights_dir = weights_dir
        self.device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")
        self.model: Optional[CIDNet] = None
        self.current_weight: Optional[str] = None
        self.current_ablation: Optional[str] = None
        self.pil2tensor = transforms.Compose([transforms.ToTensor()])

    @staticmethod
    def _unwrap_state_dict(ckpt_obj):
        if isinstance(ckpt_obj, dict):
            for key in ["state_dict", "model", "params", "net", "network"]:
                if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                    ckpt_obj = ckpt_obj[key]
                    break
        if not isinstance(ckpt_obj, dict):
            raise ValueError("Checkpoint format is not a state_dict dictionary.")

        cleaned = {}
        for k, v in ckpt_obj.items():
            new_k = k[7:] if k.startswith("module.") else k
            cleaned[new_k] = v
        return cleaned

    def _ensure_rgb_uint8(self, img_arr: np.ndarray) -> np.ndarray:
        """确保图像是RGB格式的uint8数组"""
        # 如果是2D（灰度），转换为3通道
        if img_arr.ndim == 2:
            img_arr = np.stack([img_arr, img_arr, img_arr], axis=-1)
        # 如果是单通道，转换为3通道
        elif img_arr.ndim == 3 and img_arr.shape[2] == 1:
            img_arr = np.concatenate([img_arr, img_arr, img_arr], axis=2)
        # 如果是多通道，只取前3个通道
        elif img_arr.ndim == 3 and img_arr.shape[2] > 3:
            img_arr = img_arr[:, :, :3]
        
        # 确保数据类型是uint8
        if img_arr.dtype != np.uint8:
            if img_arr.max() <= 1.0:
                img_arr = (img_arr * 255).astype(np.uint8)
            else:
                img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
        
        return img_arr

    def _load_model(self, weight_rel_path: str, hdp_ablation: str):
        if self.model is None or self.current_ablation != hdp_ablation:
            self.model = CIDNet(hdp_ablation=hdp_ablation).to(self.device)
            self.model.trans.gated = True
            self.model.trans.gated2 = True
            self.current_weight = None
            self.current_ablation = hdp_ablation

        if self.current_weight != weight_rel_path:
            weight_path = os.path.join(self.weights_dir, weight_rel_path)
            ckpt_obj = torch.load(weight_path, map_location=self.device)
            state_dict = self._unwrap_state_dict(ckpt_obj)
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            self.current_weight = weight_rel_path

    def infer(
        self,
        input_img: Optional[Image.Image],
        model_path: str,
        hdp_ablation: str,
        score_mode: str,
        gamma: float,
        alpha_s: float,
        alpha_i: float,
    ) -> Tuple[Optional[Image.Image], str, str]:
        try:
            if input_img is None:
                return None, "N/A (disabled in this demo)", "N/A (disabled in this demo)"
            if not model_path:
                return None, "请选择权重文件", "请选择权重文件"

            self._load_model(model_path, hdp_ablation)
            assert self.model is not None

            tensor = self.pil2tensor(input_img)
            factor = 8
            h, w = tensor.shape[1], tensor.shape[2]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            tensor = F.pad(tensor.unsqueeze(0), (0, padw, 0, padh), "reflect").to(self.device)

            with torch.no_grad():
                self.model.trans.alpha_s = alpha_s
                self.model.trans.alpha = alpha_i
                output = self.model(tensor ** gamma)
                output = torch.clamp(output, 0, 1)

            output = output[:, :, :h, :w].cpu()
            enhanced_img = transforms.ToPILImage()(output.squeeze(0))
            if score_mode == "Yes":
                # 确保图像是RGB格式
                if enhanced_img.mode != "RGB":
                    enhanced_img = enhanced_img.convert("RGB")
                
                # 转换为numpy数组并确保格式正确
                img_arr = np.array(enhanced_img)
                img_arr = self._ensure_rgb_uint8(img_arr)
                
                # BRISQUE需要PIL Image对象
                rgb_img = Image.fromarray(img_arr, mode="RGB")
                
                # NIQE可以直接使用numpy数组
                score_brisque = brisque.score(rgb_img)
                score_niqe = calculate_niqe(img_arr, input_order="HWC", convert_to="gray")
                
                return enhanced_img, f"{score_niqe:.6f}", f"{score_brisque:.6f}"

            return enhanced_img, "N/A (score disabled)", "N/A (score disabled)"
        except Exception as e:
            err_msg = f"推理失败: {type(e).__name__}: {e}"
            return None, err_msg, err_msg


def build_demo(weights_dir: str, use_cpu: bool):
    pth_files = find_pth_files(weights_dir)
    default_weight = pth_files[0] if pth_files else None
    enhancer = Enhancer(weights_dir=weights_dir, use_cpu=use_cpu)

    with gr.Blocks(title="IADNet Ablation Demo") as demo:
        gr.Markdown("## IADNet Ablation Demo")

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(label="Low-light Image", type="pil")
                model_path = gr.Radio(
                    choices=pth_files,
                    value=default_weight,
                    label="Model Path",
                    info="权重来自 weights 文件夹（自动扫描 .pth）",
                )
                hdp_ablation = gr.Radio(
                    choices=["full", "zi_only", "zc_only"],
                    value="full",
                    label="Ablation Mode",
                )
                score_mode = gr.Radio(
                    choices=["No", "Yes"],
                    value="No",
                    label="Image Score",
                    info="Yes 时计算 NIQE 和 BRISQUE（耗时更长）",
                )
                gamma = gr.Slider(0.1, 5, step=0.01, value=1.0, label="gamma curve")
                alpha_s = gr.Slider(0.0, 2.0, step=0.01, value=1.0, label="Alpha-s")
                alpha_i = gr.Slider(0.1, 2.0, step=0.01, value=1.0, label="Alpha-i")
                with gr.Row():
                    clear_btn = gr.Button("Clear")
                    submit_btn = gr.Button("Submit", variant="primary")

            with gr.Column(scale=1):
                result = gr.Image(label="Result", type="pil")
                niqe_text = gr.Textbox(label="NIQE", value="N/A (score disabled)")
                brisque_text = gr.Textbox(label="BRISQUE", value="N/A (score disabled)")

        submit_btn.click(
            enhancer.infer,
            inputs=[input_img, model_path, hdp_ablation, score_mode, gamma, alpha_s, alpha_i],
            outputs=[result, niqe_text, brisque_text],
        )
        clear_btn.click(
            fn=lambda: (None, None, "N/A (score disabled)", "N/A (score disabled)"),
            inputs=None,
            outputs=[input_img, result, niqe_text, brisque_text],
        )

    return demo


if __name__ == "__main__":
    args = parse_args()
    app = build_demo(weights_dir=args.weights_dir, use_cpu=args.cpu)
    app.launch(server_port=args.server_port)
