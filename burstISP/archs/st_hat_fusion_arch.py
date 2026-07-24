import torch
import torch.nn as nn
from einops import rearrange
from burstISP.utils.registry import ARCH_REGISTRY
from burstISP.archs.arch_util import to_2tuple, trunc_normal_

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows

def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x

class WindowAttention(nn.Module):
    r"""
    Shifted Window-based Multi-head Self-Attention

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.proj = nn.Linear(dim, dim)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, rpi, mask=None):
        r"""
        Args:
            qkv: Input query, key, and value tokens with shape of (num_windows*b, n, c*3)
            rpi: Relative position index
            mask (0/-inf):  Mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c3 = qkv.shape
        c = c3 // 3
        qkv = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}, qkv_bias={self.qkv_bias}'

class WindowAttention3D(nn.Module):
    r"""
    Modified WindowAttention module to accept tensor with multiple frames

    Shifted Window-based Multi-head Self-Attention

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_frames (int): Number of frames in the tensor
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, window_size, num_frames, num_heads, qkv_bias=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_frames = num_frames
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * num_frames - 1), num_heads))  # 2*Wh-1 * 2*Ww-1 * 2*N-1, nH

        self.proj = nn.Linear(dim, dim)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, rpi, mask=None):
        r"""
        Args:
            qkv: Input query, key, and value tokens with shape of (num_windows*b, n, c*3), n = Wh*Ww*N
            rpi: Relative position index
            mask (0/-inf):  Mask with shape of (num_windows, Wh*Ww*N, Wh*Ww*N) or None
        """
        assert self.window_size[0] == self.window_size[1], "Window size must be square, else current rpi_3d breaks"

        b_, n, c3 = qkv.shape
        c = c3 // 3
        qkv = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(n, n, -1)  # Wh*Ww*N,Wh*Ww*N,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww*N, Wh*Ww*N
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}, qkv_bias={self.qkv_bias}'
    
class HAB(nn.Module):
    r""" Hybrid Attention Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'

        self.norm1 = norm_layer(dim)
        self.wqkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias)

        self.conv_scale = conv_scale
        self.conv_block = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_size, rpi_sa, attn_mask):
        h, w = x_size
        b, _, c = x.shape
        # assert seq_len == h * w, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        # Conv_X
        conv_x = self.conv_block(x.permute(0, 3, 1, 2))
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = attn_mask
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        qkv_windows = self.wqkv(x_windows)
        attn_windows = self.attn(qkv_windows, rpi=rpi_sa, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c

        # reverse cyclic shift
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        attn_x = attn_x.view(b, h * w, c)

        # FFN
        x = shortcut + self.drop_path(attn_x) + conv_x * self.conv_scale
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class OCAB(nn.Module):
    # overlapping cross-attention block

    def __init__(self, dim,
                input_resolution,
                window_size,
                overlap_ratio,
                num_heads,
                qkv_bias=True,
                qk_scale=None,
                mlp_ratio=2,
                norm_layer=nn.LayerNorm
                ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size

        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3,  bias=qkv_bias)
        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size, padding=(self.overlap_win_size-window_size)//2)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((window_size + self.overlap_win_size - 1) * (window_size + self.overlap_win_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.proj = nn.Linear(dim,dim)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)

    def forward(self, x, x_size, rpi):
        h, w = x_size
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        qkv = self.qkv(x).reshape(b, h, w, 3, c).permute(3, 0, 4, 1, 2) # 3, b, c, h, w
        q = qkv[0].permute(0, 2, 3, 1) # b, h, w, c
        kv = torch.cat((qkv[1], qkv[2]), dim=1) # b, 2*c, h, w

        # partition windows
        q_windows = window_partition(q, self.window_size)  # nw*b, window_size, window_size, c
        q_windows = q_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        kv_windows = self.unfold(kv) # b, c*w*w, nw
        kv_windows = rearrange(kv_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=2, ch=c, owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous() # 2, nw*b, ow*ow, c
        k_windows, v_windows = kv_windows[0], kv_windows[1] # nw*b, ow*ow, c

        b_, nq, _ = q_windows.shape
        _, n, _ = k_windows.shape
        d = self.dim // self.num_heads
        q = q_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, nq, d
        k = k_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, n, d
        v = v_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, n, d

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size * self.window_size, self.overlap_win_size * self.overlap_win_size, -1)  # ws*ws, wse*wse, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, ws*ws, wse*wse
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.dim)
        x = window_reverse(attn_windows, self.window_size, h, w)  # b h w c
        x = x.view(b, h * w, self.dim)

        x = self.proj(x) + shortcut

        x = x + self.mlp(self.norm2(x))
        return x

class SpatialBlock(nn.Module):
    def __init__(self,
                 window_size,
                 num_feat,
                 num_heads,
                 mlp_ratio,
                 qkvbias=True):
        super(SpatialBlock, self).__init__()
        self.window_size = window_size
        self.num_feat = num_feat
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(num_feat)
        self.norm2 = nn.LayerNorm(num_feat)

        self.wqkv = nn.Linear(num_feat, 3 * num_feat, bias=qkvbias)
        self.mlp = nn.Sequential(
            nn.Linear(num_feat, mlp_ratio * num_feat),
            nn.GELU(),
            nn.Linear(mlp_ratio * num_feat, num_feat))

        self.mhsa = WindowAttention(dim=num_feat, window_size=to_2tuple(window_size),
                                    num_heads=num_heads, qkv_bias=qkvbias)

    def forward(self, x, rpi_sa):
        B, N, C, H, W = x.shape
        ws = self.window_size

        wh, ww = H // ws, W // ws

        assert H % ws == 0 and W % ws == 0

        # window mhsa
        x_win = rearrange(x, 'b n c (wh ws1) (ww ws2) -> (b n wh ww) (ws1 ws2) c', ws1=ws, ws2=ws)
        x_norm = self.norm1(x_win)
        qkv = self.wqkv(x_norm)
        attn_win = self.mhsa(qkv, rpi=rpi_sa, mask=None)

        # residual
        x_res1 = x_win + attn_win

        # mlp
        x_norm2 = self.norm2(x_res1)
        x_mlp = self.mlp(x_norm2)
        x_res2 = x_mlp + x_res1

        # reverse windowing
        x_out = rearrange(x_res2, '(b n wh ww) (ws1 ws2) c -> b n c (wh ws1) (ww ws2)', 
                          ws1=ws, ws2=ws, b=B, n=N, wh=wh, ww=ww)

        return x_out
    
class TemporalBlock(nn.Module):
    def __init__(self, 
                 num_feat,
                 num_frames,
                 num_heads,
                 mlp_ratio,
                 qkvbias=True):
        super(TemporalBlock, self).__init__()
        self.num_feat = num_feat
        self.num_heads = num_heads

        self.temporal_pe = nn.Parameter(torch.zeros(1, num_frames, num_feat))
        trunc_normal_(self.temporal_pe, std=.02)

        self.norm1 = nn.LayerNorm(num_feat)
        self.norm2 = nn.LayerNorm(num_feat)

        self.mlp = nn.Sequential(
            nn.Linear(num_feat, mlp_ratio * num_feat),
            nn.GELU(),
            nn.Linear(mlp_ratio * num_feat, num_feat))
        
        self.mhsa = nn.MultiheadAttention(embed_dim=num_feat, num_heads=num_heads, batch_first=True, bias=qkvbias)

    def forward(self, x):
        B, N, C, H, W = x.shape

        # mhsa
        x_win = rearrange(x, 'b n c h w -> (b h w) n c')
        x_norm = self.norm1(x_win + self.temporal_pe)
        attn, _ = self.mhsa(query=x_norm, key=x_norm, value=x_norm, need_weights=False)

        # residual
        x_res1 = x_win + attn

        # mlp
        x_norm2 = self.norm2(x_res1)
        x_mlp = self.mlp(x_norm2)
        x_res2 = x_mlp + x_res1

        # reverse partition
        x_out = rearrange(x_res2, '(b h w) n c -> b n c h w', b=B, h=H, w=W)

        return x_out

class SpatioTemporalBlock(nn.Module):
    def __init__(self,
                 window_size,
                 num_frames,
                 num_feat,
                 num_heads,
                 mlp_ratio,
                 qkvbias=True):
        super(SpatioTemporalBlock, self).__init__()
        self.window_size = window_size
        self.num_feat = num_feat
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(num_feat)
        self.norm2 = nn.LayerNorm(num_feat)

        self.wqkv = nn.Linear(num_feat, 3 * num_feat, bias=qkvbias)
        self.mlp = nn.Sequential(
            nn.Linear(num_feat, mlp_ratio * num_feat),
            nn.GELU(),
            nn.Linear(mlp_ratio * num_feat, num_feat))

        self.mhsa = WindowAttention3D(dim=num_feat, window_size=to_2tuple(window_size),
                                    num_frames=num_frames, num_heads=num_heads, qkv_bias=qkvbias)

    def forward(self, x, rpi_sa):
        B, N, C, H, W = x.shape
        ws = self.window_size

        wh, ww = H // ws, W // ws

        assert H % ws == 0 and W % ws == 0

        # window mhsta
        x_win = rearrange(x, 'b n c (wh ws1) (ww ws2) -> (b wh ww) (n ws1 ws2) c', ws1=ws, ws2=ws)
        x_norm = self.norm1(x_win)
        qkv = self.wqkv(x_norm)
        attn_win = self.mhsa(qkv, rpi=rpi_sa, mask=None)

        # residual
        x_res1 = x_win + attn_win

        # mlp
        x_norm2 = self.norm2(x_res1)
        x_mlp = self.mlp(x_norm2)
        x_res2 = x_mlp + x_res1

        # reverse windowing
        x_out = rearrange(x_res2, '(b wh ww) (n ws1 ws2) c -> b n c (wh ws1) (ww ws2)', 
                          ws1=ws, ws2=ws, b=B, n=N, wh=wh, ww=ww)

        return x_out
    
class FusionBlock(nn.Module):
    def __init__(self, 
                 window_size,
                 num_frames,
                 num_feat,
                 num_heads,
                 mlp_ratio,
                 qkvbias=True):
        super(FusionBlock, self).__init__()
        self.window_size = window_size
        self.num_feat = num_feat
        self.num_heads = num_heads

        head_dim = num_feat // num_heads
        self.scale = head_dim ** -0.5

        self.norm1 = nn.LayerNorm(num_feat)
        self.norm2 = nn.LayerNorm(num_feat)

        self.wq = nn.Linear(num_feat, num_feat, bias=qkvbias)
        self.wk = nn.Linear(num_feat, num_feat, bias=qkvbias)
        self.wv = nn.Linear(num_feat, num_feat, bias=qkvbias)
        self.proj = nn.Linear(num_feat, num_feat)
        self.softmax = nn.Softmax(dim=-1)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1) * (2 * num_frames - 1), num_heads))  # 2*Wh-1 * 2*Ww-1 * 2*N-1, nH
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.mlp = nn.Sequential(
            nn.Linear(num_feat, mlp_ratio * num_feat),
            nn.GELU(),
            nn.Linear(mlp_ratio * num_feat, num_feat))
        
    def forward(self, x, rpi_sa):
        B, N, C, H, W = x.shape
        ws = self.window_size

        wh, ww = H // ws, W // ws
        ref = N // 2
        P = ws * ws

        assert H % ws == 0 and W % ws == 0
        assert ww == wh, "Window size must be square, else current rpi_3d breaks"

        # window mh cross attn
        x_win = rearrange(x, 'b n c (wh ws1) (ww ws2) -> (b wh ww) n (ws1 ws2) c', ws1=ws, ws2=ws)
        x_norm = self.norm1(x_win)

        x_q = x_norm[:, ref, :, :]
        x_kv = rearrange(x_norm, 'b n p c -> b (n p) c')

        q = rearrange(self.wq(x_q), 'b p (h d) -> b h p d', h=self.num_heads)
        k = rearrange(self.wk(x_kv), 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(self.wv(x_kv), 'b n (h d) -> b h n d', h=self.num_heads)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # rpi encoding
        rpi_ref = rpi_sa[ref * P : (ref + 1) * P, :]
        rpb = self.relative_position_bias_table[rpi_ref.view(-1)].view(P, N * P, self.num_heads) # ws*ws, ws*ws*N, nH
        rpb = rearrange(rpb, 'p np h -> () h p np')

        attn = attn + rpb
        attn = self.softmax(attn)

        x_attn = (attn @ v)
        x_attn = rearrange(x_attn, 'b h p d -> b p (h d)')
        attn_win = self.proj(x_attn)

        # residual
        x_res1 = attn_win + x_win[:, ref]

        # mlp
        x_norm2 = self.norm2(x_res1)
        x_mlp = self.mlp(x_norm2)
        x_res2 = x_mlp + x_res1

        # reverse windowing
        x_out = rearrange(x_res2, '(b wh ww) (ws1 ws2) c -> b c (wh ws1) (ww ws2)',
                          ws1=ws, ws2=ws, b=B, wh=wh, ww=ww)

        return x_out

class RefinementBlock(nn.Module):
    def __init__(self,
                dim,
                window_size,
                num_heads,
                mlp_ratio,
                overlap_ratio):
        super(RefinementBlock, self).__init__()
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio

        self.s1 = OCAB(
            dim=dim, 
            input_resolution=None,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio)
        
        self.s2 = HAB(
            dim=dim, 
            input_resolution=(window_size * 2, window_size * 2),
            window_size=window_size,
            shift_size=0,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio)
        
        self.s3 = OCAB(
            dim=dim, 
            input_resolution=None,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio)
        
    def forward(self, x, rpi_sa, rpi_oca):
        B, C, H, W = x.shape
        x_size = (H, W)

        assert H == W and H % self.window_size == 0, f"Image size {H} must be a multiple of window size {self.window_size}"

        # attention
        x_seq = rearrange(x, 'b c h w -> b (h w) c')
        x_s1 = self.s1(x_seq, x_size, rpi_oca)
        x_s2 = self.s2(x_s1, x_size, rpi_sa, None)
        x_s3 = self.s3(x_s2, x_size, rpi_oca)

        x_out = rearrange(x_s3, 'b (h w) c -> b c h w', h=H, w=W)

        return x_out

@ARCH_REGISTRY.register()
class ST_HAT(nn.Module):
    """
    Spatio-Temporal Hybrid Attention Transformer (ST-HAT) Fusion:

    Given burst of aligned features size [B, N, C, H, W], where 
        N = Number of frames
        C = input channels
        H = height
        W = width
    Window size: 8x8
        
    3 stages:
    1. Deep feature extraction, 3 transformer blocks
    block 1: spatial attention only, self attention is calculated for
    each pixel with keys/vals all other pixels in same frame as q
    block 2: temporal attention only, self attention is calculated for
    each pixel with keys/vals being that same pixel in all other frames
    block 3: spatio-temporal attention, self attention is calculated for
    each pixel in each frame with k/v being all other pixels in tensor window

    2. Dimension collapse, 2 transformer blocks
    block 1: cross attention, only pixels in reference frame are queries,
    all other pixels in tensor are k/v, collapses burst into single feature map
    block 2: self attention across now-collapsed feature map for refinement and
    any aggregation

    After stage 2, the reference frame features output from stage 1 are passed
    through a 1x1 conv for projection and added as a residual to the output from
    stage 2.

    3. Refinement, 3 transformer blocks, HAT-style
    Alternating window and overlapping window attention plus channel attention to
    refine and reweight the output from stage 2 and remove any windowing artifacts
    """
    def __init__(self,
                 num_frames,
                 window_size,
                 in_feat,
                 num_feat,
                 mlp_ratio=4,
                 num_heads=4,
                 overlap_ratio=0.25,
                 depth_stage1=3,
                 depth_stage3=3):
        super(ST_HAT, self).__init__()
        self.num_frames = num_frames
        self.window_size = window_size
        self.num_feat = num_feat
        self.num_heads = num_heads
        self.overlap_ratio = overlap_ratio

        self.proj_in = nn.Conv2d(in_feat, num_feat, kernel_size=1)
        self.proj_out = nn.Conv2d(num_feat, in_feat, kernel_size=1)

        self.register_buffer('rpi_sa', self.calculate_rpi_sa())
        self.register_buffer('rpi_sa_3d', self.calculate_rpi_sa_3d())
        self.register_buffer('rpi_sa_oca', self.calculate_rpi_oca())

        self.stage1 = nn.ModuleList([
            nn.ModuleDict({
                'spatial': SpatialBlock(
                            window_size=window_size,
                            num_feat=num_feat,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio),
                'temporal': TemporalBlock(
                            num_feat=num_feat,
                            num_frames=num_frames,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio),
                'spatiotemporal': SpatioTemporalBlock(
                            window_size=window_size,
                            num_frames=num_frames,
                            num_feat=num_feat,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio)
            }) for _ in range(depth_stage1)
        ])
        
        self.fusion_block = FusionBlock(
            window_size=window_size,
            num_frames=num_frames,
            num_feat=num_feat,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio)
        
        self.s2_spatial = SpatialBlock(
            window_size=window_size,
            num_feat=num_feat,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio)
        
        self.proj = nn.Conv2d(num_feat, num_feat, kernel_size=1, stride=1, padding=0)

        self.stage3 = nn.ModuleList([
            nn.ModuleDict({
                'refinement': RefinementBlock(
                            dim=num_feat,
                            window_size=window_size,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            overlap_ratio=overlap_ratio)
            }) for _ in range(depth_stage3)
        ])

    def forward(self, x):
        B, N, C, H, W = x.shape
        ref = self.num_frames // 2

        # ---- Stage 0: Feature Projection ----
        x_wide = rearrange(x, 'b n c h w -> (b n) c h w')
        x_wide = self.proj_in(x_wide)
        x_wide = rearrange(x_wide, '(b n) c h w -> b n c h w', b=B, n=N)

        # ---- Stage 1: Deep Feature Extraction ----
        x_s1 = x_wide
        for blocks in self.stage1:
            x_s1 = blocks['spatial'](x_s1, self.rpi_sa)
            x_s1 = blocks['temporal'](x_s1)
            x_s1 = blocks['spatiotemporal'](x_s1, self.rpi_sa_3d)

        # ---- Stage 2: Dimension Collapse ----
        x_s2b1 = self.fusion_block(x_s1, self.rpi_sa_3d)
        x_s2b2 = self.s2_spatial(x_s2b1.unsqueeze(1), self.rpi_sa).squeeze(1)

        x_s1b3_ref = x_s1[:, ref]
        x_s1b3_proj = self.proj(x_s1b3_ref)
        x_s2b2 = x_s2b2 + x_s1b3_proj

        # ---- Stage 3: Refinement ----
        x_s3 = x_s2b2
        for blocks in self.stage3:
            x_s3 = blocks['refinement'](x_s3, self.rpi_sa, self.rpi_sa_oca)

        x_out = self.proj_out(x_s3)

        return x_out

    # For window attention
    def calculate_rpi_sa(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index
    
    # For 3d window attention
    def calculate_rpi_sa_3d(self):
        coords_t = torch.arange(self.num_frames)
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_t, coords_h, coords_w], indexing='ij'))  # 3, N, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, N*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, N*Wh*Ww, N*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # N*Wh*Ww, N*Wh*Ww, 3
        relative_coords[:, :, 0] += self.num_frames - 1 # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 2] += self.window_size - 1
        relative_coords[:, :, 0] *= (2 * self.window_size - 1) * (2 * self.window_size - 1) # scale N by Wh*Ww
        relative_coords[:, :, 1] *= (2 * self.window_size - 1) # scale Wh by Ww
        relative_position_index = relative_coords.sum(-1)  # N*Wh*Ww, N*Wh*Ww
        return relative_position_index
    
    # calculate relative position index for OCA
    def calculate_rpi_oca(self):
        window_size_ori = self.window_size
        window_size_ext = self.window_size + int(self.overlap_ratio * self.window_size)

        coords_h = torch.arange(window_size_ori)
        coords_w = torch.arange(window_size_ori)
        coords_ori = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, ws, ws
        coords_ori_flatten = torch.flatten(coords_ori, 1)  # 2, ws*ws

        coords_h = torch.arange(window_size_ext)
        coords_w = torch.arange(window_size_ext)
        coords_ext = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, wse, wse
        coords_ext_flatten = torch.flatten(coords_ext, 1)  # 2, wse*wse

        relative_coords = coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]   # 2, ws*ws, wse*wse

        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # ws*ws, wse*wse, 2
        relative_coords[:, :, 0] += window_size_ori - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size_ori - 1

        relative_coords[:, :, 0] *= window_size_ori + window_size_ext - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

# Test blocks
if __name__ == '__main__':
    print("--- Testing Full ST-HAT Architecture ---")
    
    # Hyperparameters
    B, N, C, H, W = 2, 5, 64, 80, 80
    window_size = 8
    num_heads = 4
    num_feat = 96
    mlp_ratio = 4
    overlap_ratio = 0.25
    s1_depth = 3
    s3_depth = 3
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")
    
    # Initialize Model
    try:
        model = ST_HAT(
            num_frames=N,
            window_size=window_size,
            in_feat=C,
            num_feat=num_feat,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            overlap_ratio=overlap_ratio,
            depth_stage1=s1_depth,
            depth_stage3=s3_depth
        ).to(device)
        print("ST_HAT successfully initialized.")
    except Exception as e:
        print(f"Initialization failed: {e}")
        exit()

    # Model Size Calculation
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params / 1e6:.3f}M")
    print(f"Trainable Parameters: {trainable_params / 1e6:.3f}M")

    # Forward Pass Test
    _input = torch.randn(B, N, C, H, W, device=device, requires_grad=True)
    
    try:
        output = model(_input)
        
        # Expected output shape: [B, C, H, W]
        expected_shape = (B, C, H, W)
        assert output.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
        print(f"Forward pass successful. Output shape: {output.shape}")
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        exit()

    # Backward Pass Test
    try:
        loss = output.sum()
        loss.backward()
        print("Backward pass successful. Gradients computed.")
        
        # Check if the input tensor received gradients (verifies graph is connected)
        assert _input.grad is not None, "Input tensor did not receive gradients."
        print("Computational graph is fully connected.")
        
    except Exception as e:
        print(f"Backward pass failed: {e}")