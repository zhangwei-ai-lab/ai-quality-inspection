# -*- coding: utf-8 -*-
"""YOLOv12 AAttn 兼容补丁（独立模块）。

`models/best.pt` 是用 YOLOv12 源码版（AAttn 拆成 qk + v）训练的，而 pip 版 ultralytics
的 AAttn 用合并的 qkv。凡是直接 `YOLO(权重)` 的入口都必须先 import 本模块打补丁，
否则加载权重时报 `AttributeError: AAttn object has no attribute qkv`。
"""

# ========================================================================
# YOLOv12 AAttn 兼容性补丁
# 不同版本的 ultralytics 中 AAttn 存在两种实现:
#   - pip 官方版: 使用 self.qkv (合并的 QKV)
#   - YOLOv12 源码版: 使用 self.qk + self.v (分离的 QK 和 V)
# 这里对 forward 方法做兼容, 使两种权重格式都能正常加载和推理
# ========================================================================
def _patch_aattn_forward():
    """修补 AAttn.forward, 兼容 qkv 和 qk+v 两种权重格式"""
    try:
        from ultralytics.nn.modules.block import AAttn
    except ImportError:
        return

    _original_forward = AAttn.forward

    def _compatible_forward(self, x):
        # 如果有 qkv 属性, 走原始逻辑
        if hasattr(self, 'qkv'):
            return _original_forward(self, x)

        # 兼容 qk + v 分离格式
        import torch
        B, C, H, W = x.shape
        N = H * W

        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x).flatten(2).transpose(1, 2)

        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape

        q, k = (
            qk.view(B, N, self.num_heads, self.head_dim * 2)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim], dim=2)
        )
        v = v.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 3, 1)

        attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)

    AAttn.forward = _compatible_forward

_patch_aattn_forward()


PATCHED = True
