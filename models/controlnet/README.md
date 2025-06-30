# 🧠 ControlNet Weights for Industrial OCV

This folder provides links to the fine-tuned ControlNet checkpoint hosted on Hugging Face.

- **Model Weights:**  
  [`juliagartor/controlnet-industrial-ocv`](https://huggingface.co/juliagartor/controlnet-industrial-ocv)

To download via 🤗 `diffusers`:
```python
from diffusers import ControlNetModel
controlnet = ControlNetModel.from_pretrained("juliagartor/controlnet-industrial-ocv", torch_dtype=torch.float16)
