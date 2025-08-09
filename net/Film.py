import jittor as jt
from jittor import nn
import math

jt.flags.use_cuda = 0  # 如果有 GPU 并想用，请设置为 1

# ---------------------------
# Helper: safe interpolate wrapper (use nn.interpolate if available, else nn.Upsample)
# ---------------------------
def upsample_nearest(x, size):
    # size: (H, W)
    if hasattr(nn, 'interpolate'):
        return nn.interpolate(x, size=size, mode='nearest')
    else:
        return nn.Upsample(size=size, mode='nearest')(x)


# ---------------------------
# Simple multi-head cross-attention implementation (B, T, C) inputs
# ---------------------------
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def execute(self, query, key, value):
        # query: (B, Tq, C); key/value: (B, Tk, C)
        B, Tq, C = query.shape
        _, Tk, _ = key.shape

        q = self.q_proj(query).reshape(B, Tq, self.num_heads, self.head_dim).permute(0,2,1,3)  # (B, heads, Tq, head_dim)
        k = self.k_proj(key).reshape(B, Tk, self.num_heads, self.head_dim).permute(0,2,3,1)    # (B, heads, head_dim, Tk)
        v = self.v_proj(value).reshape(B, Tk, self.num_heads, self.head_dim).permute(0,2,1,3)  # (B, heads, Tk, head_dim)

        attn = jt.matmul(q, k) * self.scale  # (B, heads, Tq, Tk)
        # softmax along last dim
        attn = nn.Softmax(dim=-1)(attn)
        out = jt.matmul(attn, v)  # (B, heads, Tq, head_dim)
        out = out.permute(0,2,1,3).reshape(B, Tq, C)
        out = self.out_proj(out)
        return out


# ---------------------------
# Self-attention wrapper used inside Restormer block
# ---------------------------
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.cross = CrossAttention(embed_dim, num_heads)

    def execute(self, x):
        # self-attn for x (B, N, C)
        return self.cross(x, x, x)


# ---------------------------
# Simplified Restormer-like TransformerBlock (works on (B,C,H,W))
# - This is a lightweight replacement to make the file self-contained.
# - It tokenizes spatial dims -> applies self-attn + FFN -> reshapes back.
# ---------------------------
class Restormer(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_expansion=4, bias=False, LayerNorm_type='WithBias'):
        super(Restormer, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads if num_heads > 0 else 1

        # token-level normalization & modules
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_expansion),
            nn.ReLU(),
            nn.Linear(dim * ffn_expansion, dim)
        )

    def execute(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        tokens = x.reshape(B, C, H * W).permute(0, 2, 1)  # (B, N, C)
        # attn block
        t = self.norm1(tokens)
        t = self.attn(t)  # (B, N, C)
        tokens = tokens + t
        # ffn
        u = self.norm2(tokens)
        u = self.ffn(u)
        tokens = tokens + u
        out = tokens.permute(0, 2, 1).reshape(B, C, H, W)
        return out


# ---------------------------
# ChannelAttention (keeps API/behavior close于 PyTorch 原实现)
# ---------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Conv1d(in_planes, max(in_planes // 16, 1), 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(max(in_planes // 16, 1), in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def execute(self, x):
        # x: (B, T, C) in our usage (note original code permuted)
        # We'll mimic original permute shape handling where needed
        x_perm = x.permute(0, 2, 1)  # (B, C, T)
        avg_out = self.fc(self.avg_pool(x_perm)).permute(0, 2, 1)
        max_out = self.fc(self.max_pool(x_perm)).permute(0, 2, 1)
        out = avg_out + max_out
        return self.sigmoid(out)


# ---------------------------
# imagefeature2textfeature: conv -> upsample to fixed size -> flatten into tokens of hidden_dim
# ---------------------------
class imagefeature2textfeature(nn.Module):
    def __init__(self, in_channel, mid_channel, hidden_dim, target_size=(288, 384)):
        super(imagefeature2textfeature, self).__init__()
        self.conv = nn.Conv2d(in_channel, mid_channel, kernel_size=1)
        self.hidden_dim = hidden_dim
        self.target_size = target_size

    def execute(self, x):
        # x: (B, C, H, W)
        x = self.conv(x)
        x = upsample_nearest(x, size=self.target_size)  # (B, mid_channel, 288, 384)
        B, C, H, W = x.shape
        total = C * H * W
        assert total % self.hidden_dim == 0, "C*H*W must be divisible by hidden_dim"
        N = total // self.hidden_dim
        # flatten per batch and split into tokens of size hidden_dim
        x_flat = x.reshape(B, -1)  # (B, total)
        x_tokens = x_flat.reshape(B, N, self.hidden_dim)  # (B, N, hidden_dim)
        return x_tokens


# ---------------------------
# restormer_cablock: main cross-attention block (converted)
# ---------------------------
class restormer_cablock(nn.Module):
    def __init__(
            self,
            input_channel=1,
            restormerdim=32,
            restormerhead=8,
            image2text_dim=10,
            ffn_expansion_factor=4,
            bias=False,
            LayerNorm_type='WithBias',
            hidden_dim=768,
            pooling='avg',
            normalization='l1'
    ):
        super().__init__()
        # convs
        self.convA1 = nn.Conv2d(input_channel, restormerdim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.preluA1 = nn.PReLU()
        self.convA2 = nn.Conv2d(image2text_dim, restormerdim, kernel_size=1)
        self.preluA2 = nn.PReLU()
        self.convA3 = nn.Conv2d(2 * restormerdim, restormerdim, kernel_size=1)
        self.preluA3 = nn.PReLU()

        self.convB1 = nn.Conv2d(input_channel, restormerdim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.preluB1 = nn.PReLU()
        self.convB2 = nn.Conv2d(image2text_dim, restormerdim, kernel_size=1)
        self.preluB2 = nn.PReLU()
        self.convB3 = nn.Conv2d(2 * restormerdim, restormerdim, kernel_size=1)
        self.preluB3 = nn.PReLU()

        self.image2text_dim = image2text_dim
        # Use our simplified Restormer block
        self.restormerA1 = Restormer(restormerdim, restormerhead, ffn_expansion_factor, bias, LayerNorm_type)
        self.restormerB1 = Restormer(restormerdim, restormerhead, ffn_expansion_factor, bias, LayerNorm_type)
        self.cross_attentionA1 = CrossAttention(embed_dim=hidden_dim, num_heads=8)
        self.cross_attentionA2 = CrossAttention(embed_dim=hidden_dim, num_heads=8)
        self.imagef2textfA1 = imagefeature2textfeature(restormerdim, image2text_dim, hidden_dim)
        self.imagef2textfB1 = imagefeature2textfeature(restormerdim, image2text_dim, hidden_dim)

    def execute(self, imageA, imageB, text):
        # imageA/B: (B, C, H, W) ; text: (B, T_text, hidden_dim)
        if len(imageA.shape) == 3:
            # mimic original behavior if image given as HWC
            imageA = imageA.unsqueeze(0).permute(0, 3, 1, 2)
            imageB = imageB.unsqueeze(0).permute(0, 3, 1, 2)

        b, _, H, W = imageA.shape

        imageA = self.restormerA1(self.preluA1(self.convA1(imageA)))
        imageAtotext = self.imagef2textfA1(imageA)  # (B, N, hidden_dim)

        imageB = self.restormerB1(self.preluB1(self.convB1(imageB)))
        imageBtotext = self.imagef2textfB1(imageB)  # (B, N, hidden_dim)

        # cross-attention: query=text (B, T_text, hidden_dim), key/value = image tokens (B, N, hidden_dim)
        ca_A = self.cross_attentionA1(text, imageAtotext, imageAtotext)  # (B, T_text, hidden_dim)
        imageA_sideout = imageA
        # pool to one token per query (mimic adaptive_avg_pool1d over tokens)
        # ba shape -> (B, T_text, hidden_dim) -> we want (B, 1, hidden_dim) per original code -> then normalize
        ca_A_pool = ca_A.permute(0, 2, 1)  # (B, hidden, T_text)
       # 对第2维取平均值并保持维度，再进行维度交换
        ca_A_pool = jt.mean(ca_A_pool, dim=2, keepdim=True).permute(0, 2, 1)  # (B,1,hidden_dim)
        # normalize p=1 along hidden dim original used dim=2 after permuting; mimic original normalization
        ca_A_norm = ca_A_pool / (ca_A_pool.sum(dim=2, keepdims=True) + 1e-8)  # (B,1,hidden_dim)
        # broadcast multiply with imageAtotext (B,N,hidden_dim) -> need same dim order as original
        # original did: (imageAtotext * ca_A).view(...) -> so we broadcast ca_A_norm across tokens dimension
        ca_A_scaled = imageAtotext * ca_A_norm  # (B,N,hidden_dim)
        # reshape back to spatial map: first make sure N * hidden_dim equals image2text_dim*288*384
        # We will reconstruct (B, image2text_dim, 288, 384)
        B, N, hidden = ca_A_scaled.shape
        total = N * hidden
        # compute expected channel count
        c = self.image2text_dim
        h_t, w_t = 288, 384
        assert c * h_t * w_t == total, f"Mismatch during reshape: {c}*{h_t}*{w_t} != {total}"
        ca_A_map = ca_A_scaled.reshape(B, c, h_t, w_t)
        imageA_sideout = upsample_nearest(imageA_sideout, size=(H, W))
        ca_A_map = upsample_nearest(ca_A_map, size=(H, W))
        ca_A = self.preluA3(self.convA3(jt.concat((upsample_nearest(imageA, size=(H, W)),
                                                   self.preluA2(self.convA2(ca_A_map)) + imageA_sideout), dim=1)))

        # same for B
        ca_B = self.cross_attentionA2(text, imageBtotext, imageBtotext)
        imageB_sideout = imageB
        ca_B_pool = ca_B.permute(0, 2, 1)
        # 对第2维取平均值并保持维度，再进行维度交换
        ca_B_pool = jt.mean(ca_B_pool, dim=2, keepdim=True).permute(0, 2, 1)
        ca_B_norm = ca_B_pool / (ca_B_pool.sum(dim=2, keepdims=True) + 1e-8)
        ca_B_scaled = imageBtotext * ca_B_norm
        B, N, hidden = ca_B_scaled.shape
        total = N * hidden
        c = self.image2text_dim
        assert c * 288 * 384 == total
        ca_B_map = ca_B_scaled.reshape(B, c, 288, 384)
        imageB_sideout = upsample_nearest(imageB_sideout, size=(H, W))
        ca_B_map = upsample_nearest(ca_B_map, size=(H, W))
        ca_B = self.preluB3(self.convB3(jt.concat((upsample_nearest(imageB, size=(H, W)),
                                                   self.preluB2(self.convB2(ca_B_map)) + imageB_sideout), dim=1)))

        return ca_A, ca_B


# ---------------------------
# text preprocessing conv1d
# ---------------------------
class text_preprocess(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(text_preprocess, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, 1, 1, 0)

    def execute(self, x):
        # x: (B, T, C_in)
        x = x.permute(0, 2, 1)  # (B, C_in, T)
        x = self.conv(x)
        return x.permute(0, 2, 1)  # (B, T, C_out)


# ---------------------------
# Final Net (converted)
# ---------------------------
class Net(nn.Module):
    def __init__(
            self,
            mid_channel=32,
            decoder_num_heads=8,
            ffn_factor=4,
            bias=False,
            LayerNorm_type='WithBias',
            out_channel=1,
            hidden_dim=256,
            image2text_dim=32,
            pooling='avg',
            normalization='l1'
    ):
        super().__init__()
        self.text_process = text_preprocess(768, hidden_dim)
        self.restormerca1 = restormer_cablock(hidden_dim=hidden_dim, image2text_dim=image2text_dim)
        self.restormerca2 = restormer_cablock(input_channel=mid_channel, hidden_dim=hidden_dim,
                                              image2text_dim=image2text_dim)
        self.restormerca3 = restormer_cablock(input_channel=mid_channel, hidden_dim=hidden_dim,
                                              image2text_dim=image2text_dim)
        # simplified Restormer blocks (work on 4D tensors)
        self.restormer1 = Restormer(2 * mid_channel, decoder_num_heads, ffn_factor, bias, LayerNorm_type)
        self.restormer2 = Restormer(mid_channel, decoder_num_heads, ffn_factor, bias, LayerNorm_type)
        self.restormer3 = Restormer(mid_channel, decoder_num_heads, ffn_factor, bias, LayerNorm_type)
        self.conv1 = nn.Conv2d(2 * mid_channel, mid_channel, kernel_size=1)
        self.conv2 = nn.Conv2d(mid_channel, out_channel, kernel_size=1)
        self.softmax = nn.Sigmoid()
        self.mid_channel = mid_channel

    def execute(self, imageA, imageB, text):
        # text: (B, T_text, 768) -> processed to hidden_dim
        text = self.text_process(text)  # (B, T_text, hidden_dim)
        featureA, featureB = self.restormerca1(imageA, imageB, text)
        featureA, featureB = self.restormerca2(featureA, featureB, text)
        featureA, featureB = self.restormerca3(featureA, featureB, text)

        fusionfeature = jt.concat((featureA, featureB), dim=1)  # (B, 2*mid_channel, H, W)
        fusionfeature = self.restormer1(fusionfeature)
        fusionfeature = self.conv1(fusionfeature)
        fusionfeature = self.restormer2(fusionfeature)
        fusionfeature = self.restormer3(fusionfeature)
        fusionfeature = self.conv2(fusionfeature)
        fusionfeature = self.softmax(fusionfeature)
        return fusionfeature
