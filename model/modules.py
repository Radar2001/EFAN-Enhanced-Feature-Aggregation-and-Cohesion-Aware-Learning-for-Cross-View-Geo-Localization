"""Building blocks of the Scale-Unified Semantic-Spatial Feature Enhancement
Module (S3EM) used in EFAN (Enhanced Feature Aggregation Network).

The S3EM enhances multi-scale backbone features through two parallel paths:

* Semantic Guidance Enhancement (SGE) path: a top-down cross-attention pathway
  in which deep features act as Queries (Q) and sequentially guide shallow
  features (Keys/Values), distilling high-level semantics from abstract to
  concrete levels.
* Linear Spatial Enhancement (LSE) path: a linear-attention pathway that
  aggregates multi-scale features to capture cross-scale spatial dependencies.

The two paths are adaptively fused by a learnable gate (:class:`GatedFusion`).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEncodingSine(nn.Module):
    """Sinusoidal position encoding generalized to 2-D images."""

    def __init__(self, d_model, max_shape=(256, 256), temp_bug_fix=True):
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if temp_bug_fix:
            div_term = torch.exp(
                torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / (d_model // 2))
            )
        else:
            # a buggy implementation kept only for backward compatibility
            div_term = torch.exp(
                torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / d_model // 2)
            )
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """Args:
        x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]


def elu_feature_map(x):
    return F.elu(x) + 1


class LinearAttention(nn.Module):
    """Multi-head linear attention proposed in "Transformers are RNNs"."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Args:
        queries: [N, L, H, D]
        keys:    [N, S, H, D]
        values:  [N, S, H, D]
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class LinearSpatialEnhancementLayer(nn.Module):
    """Transformer layer with linear attention, used by the LSE path."""

    def __init__(self, d_model, nheads=8, dim_feedforward=2048, dropout=0.1,
                 activation=nn.GELU()):
        super().__init__()
        self.d_model = d_model
        self.nheads = nheads
        self.dim = d_model // nheads

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.merge = nn.Linear(d_model, d_model)

        self.attention = LinearAttention(eps=1e-6)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        for proj in [self.query_proj, self.key_proj, self.value_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.constant_(proj.bias, 0.)

    def forward(self, q_src, kv_src):
        residual = q_src

        # the query bypasses linear projection (asymmetric design)
        query = q_src
        key = self.key_proj(kv_src)
        value = self.value_proj(kv_src)

        b, l, c = query.shape
        query = query.view(b, l, self.nheads, self.dim)
        key = key.view(b, -1, self.nheads, self.dim)
        value = value.view(b, -1, self.nheads, self.dim)

        attn_out = self.attention(query, key, value)
        attn_out = attn_out.contiguous().view(b, l, c)
        attn_out = self.merge(attn_out)

        out = self.norm1(residual + self.dropout1(attn_out))

        residual = out
        out = self.mlp(out)
        out = self.norm2(residual + self.dropout2(out))

        return out


class LinearSpatialEnhancement(nn.Module):
    """Linear Spatial Enhancement (LSE) path of the S3EM.

    Concatenates the unified multi-scale feature sequences along the spatial
    dimension and processes the aggregated sequence with linear attention.
    The channel dimension is interpreted as the sequence length.
    """

    def __init__(self, output_channels, spatial_size=12, ffn_dim_ratio=2, dropout=0.1):
        super().__init__()
        self.output_channels = output_channels
        self.spatial_size = spatial_size
        self.total_channels = 3 * spatial_size * spatial_size

        self.nheads = 3
        if self.total_channels % self.nheads != 0:
            self.nheads = 4

        ffn_dim = self.total_channels * ffn_dim_ratio
        self.unified_transformer = LinearSpatialEnhancementLayer(
            d_model=self.total_channels,
            nheads=self.nheads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
        )

    def forward(self, features_list):
        sequence_features = []
        for feat in features_list:
            B, C, H, W = feat.shape
            sequence_features.append(feat.view(B, C, H * W))  # [B, C, H*W]

        concatenated_seq = torch.cat(sequence_features, dim=2)  # [B, C, 3*H*W]
        enhanced_seq = self.unified_transformer(concatenated_seq, concatenated_seq)

        enhanced_features = []
        start_channel = 0
        for orig_channels in self.output_channels:
            end_channel = start_channel + self.spatial_size * self.spatial_size
            enhanced_features.append(enhanced_seq[:, :, start_channel:end_channel])
            start_channel = end_channel

        return enhanced_features


class SemanticGuidanceLayer(nn.Module):
    """Cross-attention layer of the SGE path.

    High-level features act as Queries (Q) and shallow features act as Keys (K)
    and Values (V).
    """

    def __init__(self, query_dim, key_dim, nheads=8, dropout=0.1):
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim

        self.query_proj = nn.Linear(query_dim, query_dim)
        self.key_proj = nn.Linear(key_dim, query_dim)
        self.value_proj = nn.Linear(key_dim, query_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=nheads,
            dropout=dropout,
            batch_first=True,
        )

        self.out_proj = nn.Linear(query_dim, query_dim)
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim * 4, query_dim),
        )

    def forward(self, query, key, value):
        residual = query

        q = self.query_proj(query)
        k = self.key_proj(key)
        v = self.value_proj(value)

        attn_output, _ = self.attention(q, k, v)
        out = self.norm1(residual + self.dropout(attn_output))

        ffn_output = self.ffn(out)
        out = self.norm2(out + self.dropout(ffn_output))

        return out


class SemanticGuidanceEnhancement(nn.Module):
    """Semantic Guidance Enhancement (SGE) path of the S3EM.

    Top-down hierarchical semantic propagation: L3 first guides L2, then the
    enhanced L2 further guides L1, and the direct L3->L1 guidance is fused with
    the indirect guidance through L2.
    """

    def __init__(self, output_channels, spatial_size=12, dropout=0.1):
        super().__init__()
        self.output_channels = output_channels
        self.spatial_size = spatial_size

        self.semantic_guidance_layers = nn.ModuleList([
            # L3 -> L1
            SemanticGuidanceLayer(
                query_dim=output_channels[2], key_dim=output_channels[0], dropout=dropout
            ),
            # L3 -> L2
            SemanticGuidanceLayer(
                query_dim=output_channels[2], key_dim=output_channels[1], dropout=dropout
            ),
            # L2 -> L1
            SemanticGuidanceLayer(
                query_dim=output_channels[1], key_dim=output_channels[0], dropout=dropout
            ),
        ])

    def forward(self, channel_features, spatial_features):
        enhanced_features = [None] * 3

        l1_feat, l2_feat, l3_feat = channel_features
        _, _, l3_spatial = spatial_features

        B, C1, H1, W1 = l1_feat.shape
        _, C2, H2, W2 = l2_feat.shape
        _, C3, H3, W3 = l3_feat.shape

        l1_seq = l1_feat.view(B, C1, H1 * W1).transpose(1, 2)  # [B, L1, C1]
        l2_seq = l2_feat.view(B, C2, H2 * W2).transpose(1, 2)  # [B, L2, C2]
        l3_spatial_seq = l3_spatial.view(B, C3, H3 * W3).transpose(1, 2)  # [B, L3, C3]

        # 1. L3 guides L2
        guided_l2 = self.semantic_guidance_layers[1](
            query=l3_spatial_seq, key=l2_seq, value=l2_seq
        )

        # 2. L3 guides L1
        guided_l1_from_l3 = self.semantic_guidance_layers[0](
            query=l3_spatial_seq, key=l1_seq, value=l1_seq
        )

        # 3. enhanced L2 guides L1
        guided_l1_from_l2 = self.semantic_guidance_layers[2](
            query=guided_l2, key=l1_seq, value=l1_seq
        )

        enhanced_l1 = guided_l1_from_l3 + guided_l1_from_l2
        enhanced_l2 = guided_l2

        enhanced_features[0] = enhanced_l1.transpose(1, 2).view(B, C1, H1, W1)
        enhanced_features[1] = enhanced_l2.transpose(1, 2).view(B, C2, H2, W2)
        enhanced_features[2] = l3_feat  # L3 stays unchanged

        return enhanced_features


class GatedFusion(nn.Module):
    """Stage-wise learnable gate that adaptively fuses the SGE and LSE outputs."""

    def __init__(self, output_channels):
        super().__init__()
        self.output_channels = output_channels

        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid(),
            )
            for d_model in output_channels
        ])

    def forward(self, cross_scale_features, spatial_features):
        fused_features = []

        for i, (cross_feat, sp_feat) in enumerate(zip(cross_scale_features, spatial_features)):
            B, C, H, W = cross_feat.shape

            cross_seq = cross_feat.view(B, C, H * W).transpose(1, 2)
            sp_seq = sp_feat.view(B, C, H * W).transpose(1, 2)

            fusion_gate = self.fusion_layers[i](
                torch.cat([cross_seq, sp_seq], dim=-1)
            )

            fused_seq = fusion_gate * cross_seq + (1 - fusion_gate) * sp_seq
            fused_feat = fused_seq.transpose(1, 2).view(B, C, H, W)
            fused_features.append(fused_feat)

        return fused_features


class S3EM(nn.Module):
    """Scale-Unified Semantic-Spatial Feature Enhancement Module (S3EM).

    Enhances multi-scale features through the parallel LSE (linear spatial
    enhancement) and SGE (semantic guidance enhancement) paths, then fuses the
    two pathway outputs with a learnable gate.
    """

    def __init__(self, output_channels, spatial_size=12, ffn_dim_ratio=2, dropout=0.1):
        super().__init__()
        self.output_channels = output_channels
        self.spatial_size = spatial_size

        self.spatial_stream = LinearSpatialEnhancement(
            output_channels=output_channels,
            spatial_size=spatial_size,
            dropout=dropout,
        )

        self.cross_scale_enhance = SemanticGuidanceEnhancement(
            output_channels=output_channels,
            spatial_size=spatial_size,
            dropout=dropout,
        )

        self.feature_fusion = GatedFusion(output_channels=output_channels)

    def forward(self, features_list):
        spatial_enhanced = self.spatial_stream(features_list)
        final_enhanced = self.cross_scale_enhance(features_list, features_list)
        fused_features = self.feature_fusion(final_enhanced, spatial_enhanced)
        return fused_features
