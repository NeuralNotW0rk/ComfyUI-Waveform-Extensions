import torch
import numpy as np
import importlib, subprocess, sys


class JoinAudio():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor_1": ("AUDIO", ),
                "tensor_2": ("AUDIO", ),
                "gap": ("INT", {"default": 0, "min": -1000000000, "max": 1000000000, "step": 1}),
                "overlap_method": (("overwrite", "linear", "sigmoid"), {"default": "sigmoid"})
                },
            "optional": {
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 10000000000, "step": 1, "forceInput": True}),
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


NODE_CLASS_MAPPINGS = {
    'JoinAudio': JoinAudio
}





