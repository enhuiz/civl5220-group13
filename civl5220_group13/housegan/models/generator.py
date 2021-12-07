import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def conv_block(
    in_channels,
    out_channels,
    k,
    s,
    p,
    act=None,
    upsample=False,
    spec_norm=False,
):
    block = []

    if upsample:
        if spec_norm:
            block.append(
                spectral_norm(
                    torch.nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel_size=k,
                        stride=s,
                        padding=p,
                        bias=True,
                    )
                )
            )
        else:
            block.append(
                torch.nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    bias=True,
                )
            )
    else:
        if spec_norm:
            block.append(
                spectral_norm(
                    torch.nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=k,
                        stride=s,
                        padding=p,
                        bias=True,
                    )
                )
            )
        else:
            block.append(
                torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    bias=True,
                )
            )
    if "leaky" in act:
        block.append(torch.nn.LeakyReLU(0.1, inplace=True))
    elif "relu" in act:
        block.append(torch.nn.ReLU(True))
    elif "tanh":
        block.append(torch.nn.Tanh())
    return block


class CMP(nn.Module):
    def __init__(self, in_channels):
        super(CMP, self).__init__()
        self.in_channels = in_channels
        self.encoder = nn.Sequential(
            *conv_block(3 * in_channels, 2 * in_channels, 3, 1, 1, act="leaky"),
            *conv_block(2 * in_channels, 2 * in_channels, 3, 1, 1, act="leaky"),
            *conv_block(2 * in_channels, in_channels, 3, 1, 1, act="leaky")
        )

    def forward(self, feats, edges=None):

        # allocate memory
        dtype, device = feats.dtype, feats.device
        edges = edges.view(-1, 3)
        V, E = feats.size(0), edges.size(0)
        pooled_v_pos = torch.zeros(
            V,
            feats.shape[-3],
            feats.shape[-1],
            feats.shape[-1],
            dtype=dtype,
            device=device,
        )
        pooled_v_neg = torch.zeros(
            V,
            feats.shape[-3],
            feats.shape[-1],
            feats.shape[-1],
            dtype=dtype,
            device=device,
        )

        # pool positive edges
        pos_inds = torch.where(edges[:, 1] > 0)
        pos_v_src = torch.cat([edges[pos_inds[0], 0], edges[pos_inds[0], 2]]).long()
        pos_v_dst = torch.cat([edges[pos_inds[0], 2], edges[pos_inds[0], 0]]).long()
        pos_vecs_src = feats[pos_v_src.contiguous()]
        pos_v_dst = pos_v_dst.view(-1, 1, 1, 1).expand_as(pos_vecs_src).to(device)
        pooled_v_pos = pooled_v_pos.scatter_add(0, pos_v_dst, pos_vecs_src)

        # pool negative edges
        neg_inds = torch.where(edges[:, 1] < 0)
        neg_v_src = torch.cat([edges[neg_inds[0], 0], edges[neg_inds[0], 2]]).long()
        neg_v_dst = torch.cat([edges[neg_inds[0], 2], edges[neg_inds[0], 0]]).long()
        neg_vecs_src = feats[neg_v_src.contiguous()]
        neg_v_dst = neg_v_dst.view(-1, 1, 1, 1).expand_as(neg_vecs_src).to(device)
        pooled_v_neg = pooled_v_neg.scatter_add(0, neg_v_dst, neg_vecs_src)

        # update nodes features
        enc_in = torch.cat([feats, pooled_v_pos, pooled_v_neg], 1)
        out = self.encoder(enc_in)
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = 32 // 4
        self.l1 = nn.Sequential(nn.Linear(138, 16 * self.init_size ** 2))
        self.upsample_1 = nn.Sequential(
            *conv_block(16, 16, 4, 2, 1, act="leaky", upsample=True)
        )
        self.upsample_2 = nn.Sequential(
            *conv_block(16, 16, 4, 2, 1, act="leaky", upsample=True)
        )
        self.cmp_1 = CMP(in_channels=16)
        self.cmp_2 = CMP(in_channels=16)
        self.decoder = nn.Sequential(
            *conv_block(16, 256, 3, 1, 1, act="leaky"),
            *conv_block(256, 128, 3, 1, 1, act="leaky"),
            *conv_block(128, 1, 3, 1, 1, act="tanh")
        )

    @property
    def latent_dim(self):
        return 128

    def forward(self, z, given_y=None, given_w=None):
        z = z.view(-1, 128)
        # include nodes
        if True:
            y = given_y.view(-1, 10)
            z = torch.cat([z, y], 1)
        x = self.l1(z)
        x = x.view(-1, 16, self.init_size, self.init_size)
        x = self.cmp_1(x, given_w).view(-1, *x.shape[1:])
        x = self.upsample_1(x)
        x = self.cmp_2(x, given_w).view(-1, *x.shape[1:])
        x = self.upsample_2(x)
        x = self.decoder(x.view(-1, x.shape[1], *x.shape[2:]))
        x = x.view(-1, *x.shape[2:])
        return x
