# AttentionModules

A collection of attention mechanism modules for deep learning, implemented in PyTorch. Designed for easy plug-and-play integration into CNN and Transformer architectures.

In recent years, attention-based modules have been proposed that can be inserted into one's own visual model.

This repository can also be used by beginners to learn the attention mechanism module and become familiar with PyTorch syntax, providing more options when building their own models.
## Modules

| Module | Description |
|--------|-------------|
| A2 | Double Attention Network |
| ACmix | ACmix: Self-Attention and Convolution |
| BAM | Bottleneck Attention Module |
| CA | Coordinate Attention |
| CBAM | Convolutional Block Attention Module |
| CCAM | Criss-Cross Attention Module |
| ECA | Efficient Channel Attention |
| ELA | Efficient Local Attention |
| EMA | Exponential Moving Average Attention |
| GAM | Global Attention Module |
| SCSA | Semantic-Channel-Spatial Attention |
| SE | Squeeze-and-Excitation Networks |
| SimAM | Simple Attention Module |
| SK | Selective Kernel Networks |
| SLAM | Spatial-LSTM Attention Module |
| TripletAttention | Triplet Attention |

## Usage

```python
from AttentionModules import CA, SE, CBAM, ECA  # import any module you need

attn = SE(channel=64)
out = attn(x)
```
Alternatively, you can clone it locally and rewrite the `defward` function in the module for your own use.

## Requirements

- Python 3.8 or above
- PyTorch 2.x

---

> This repository is actively maintained and will be continuously updated with more attention modules. We welcome your module submissions.
