import torch
import numpy as np
import importlib
import subprocess
import sys


def hijack_import(importname, installname):
    try:
        importlib.import_module(importname)
    except ModuleNotFoundError:
        print(f"Import failed for {importname}, Installing {installname}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", installname])


hijack_import("librosa", "librosa")

import librosa.effects


class JoinAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor_1": ("AUDIO", ),
                "tensor_2": ("AUDIO", ),
                "gap": ("INT", {"default": 0, "min": -1e9, "max": 1e9, "step": 1}),
                "overlap_method": (("overwrite", "linear", "sigmoid"), {"default": "sigmoid"})
                },
            "optional": {
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 1e9, "step": 1, "forceInput": True}),
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("tensor", "sample_rate")
    FUNCTION = "join_audio"

    CATEGORY = "Audio/Waveform"

    def join_audio(self, tensor_1, tensor_2, gap, overlap_method, sample_rate):
        joined_length = tensor_1.size(2) + tensor_2.size(2) + gap
        joined_tensor = torch.zeros((tensor_1.size(0), tensor_1.size(1), joined_length), device=tensor_1.device)
        tensor_1_masked = tensor_1.clone()
        tensor_2_masked = tensor_2.clone()

        # Overlapping
        if gap < 0:
            mask = np.zeros(abs(gap))
            if overlap_method == 'linear':
                mask = np.linspace(0.0, 1.0, num=abs(gap))
            elif overlap_method == 'sigmoid':
                k = 6
                mask = np.linspace(-1.0, 1.0, num=abs(gap))
                mask = 1 / (1 + np.exp(-mask * k))
            mask = torch.from_numpy(mask).to(device=tensor_1.device)
            tensor_1_masked[:, :, -abs(gap):] *= 1.0 - mask
            tensor_2_masked[:, :, :abs(gap)] *= mask

        joined_tensor[:, :, :tensor_1.size(2)] += tensor_1_masked
        joined_tensor[:, :, tensor_1.size(2) + gap:] += tensor_2_masked

        return joined_tensor, sample_rate


class StretchAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("AUDIO", ),
                "rate": ("FLOAT", {"default": 1.0, "min": 1e-9, "max": 1e9, "step": 0.1})
                },
            "optional": {
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 1e9, "step": 1, "forceInput": True}),
                },
            }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("tensor", "sample_rate")
    FUNCTION = "stretch_audio"

    CATEGORY = "Audio/Waveform"

    def stretch_audio(self, tensor, rate, sample_rate):
        y = tensor.cpu().numpy()
        y = librosa.effects.time_stretch(y, rate=rate)
        tensor_out = torch.from_numpy(y).to(device=tensor.device)

        return tensor_out, sample_rate


class ReverseAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("AUDIO",),
            },
            "optional": {
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 1e9, "step": 1, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("tensor", "sample_rate")
    FUNCTION = "reverse_audio"

    CATEGORY = "Audio/Waveform"

    def reverse_audio(self, tensor, sample_rate):
        return torch.flip(tensor.clone(), (2,)), sample_rate


NODE_CLASS_MAPPINGS = {
    'JoinAudio': JoinAudio,
    'StretchAudio': StretchAudio,
    'ReverseAudio': ReverseAudio
}





