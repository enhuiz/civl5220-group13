import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from einops import rearrange, repeat


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
                    nn.ConvTranspose2d(
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
                nn.ConvTranspose2d(
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
                    nn.Conv2d(
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
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    bias=True,
                )
            )
    if "leaky" in act:
        block.append(nn.LeakyReLU(0.1, inplace=True))
    elif "relu" in act:
        block.append(nn.ReLU(True))
    elif "tanh":
        block.append(nn.Tanh())
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

    def pool_edges(self, feats, edges, positive=False):
        """
        Args:
            feats: (b k c h w)
            edges: (ck2 3)
        Returns:
            feats: (b k c h w)
        """
        assert (feats.dim(), edges.dim()) == (5, 2)

        if positive:
            edges = edges[edges[..., 1] > 0]
        else:
            edges = edges[edges[..., 1] < 0]

        b, _, c, h, w = feats.shape
        pooled = torch.zeros_like(feats)

        dst = torch.cat([edges[..., 0], edges[..., 2]], dim=-1)  # (2l)
        dst = repeat(dst, "l2 -> b l2 c h w", b=b, c=c, h=h, w=w)
        dst = dst.long()

        src = torch.cat([edges[..., 2], edges[..., 0]], dim=-1)  # (2l)
        src = repeat(src, "l2 -> b l2 c h w", b=b, c=c, h=h, w=w)
        src = src.long()
        src = feats.gather(1, src)

        # dst: (b 2l d h w)
        # src: (b 2l d h w)
        pooled = pooled.scatter_add(1, dst, src)

        return pooled

    def forward(self, feats, edges):
        """
        Args:
            feats: (b k c h w)
            edges: (ck2 3)
        """
        b, k = feats.shape[:2]

        pooled_v_pos = torch.zeros_like(feats)
        pooled_v_neg = torch.zeros_like(feats)

        pooled_v_pos = self.pool_edges(feats, edges, positive=True)
        pooled_v_neg = self.pool_edges(feats, edges, positive=False)

        enc_in = torch.cat([feats, pooled_v_pos, pooled_v_neg], -3)
        out = self.encoder(enc_in.flatten(0, 1))
        out = out.view(b, k, *out.shape[1:])

        return out


def flat_forward(m, x, first_dim=0, last_dim=1):
    dims = x.shape[first_dim : last_dim + 1]
    x = m(x.flatten(first_dim, last_dim))
    x = x.view(*dims, *x.shape[1:])
    return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_size = 32 // 4
        self.l1 = nn.Sequential(
            nn.Linear(138, 16 * self.init_size ** 2),
        )
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

    def forward(self, z, onehot_nodes, edges):
        """This forward supports batched z.
        We don't support batched graph here as graph may have different number of nodes and edges,
        which can be tricky.
        Args:
            z: (b k d)
            onehot_nodes: (k d)
            edges: (ck2 d)
        Returns:
            pass
        """
        assert (z.dim(), onehot_nodes.dim(), edges.dim()) == (3, 2, 2)

        onehot_nodes = repeat(onehot_nodes, "k d -> b k d", b=len(z))
        z = torch.cat([z, onehot_nodes], dim=-1)

        x = self.l1(z)

        s = self.init_size
        x = rearrange(x, "b k (c h w) -> b k c h w", c=16, h=s, w=s)

        x = self.cmp_1(x, edges)
        x = flat_forward(self.upsample_1, x, 0, 1)

        x = self.cmp_2(x, edges)
        x = flat_forward(self.upsample_2, x, 0, 1)

        x = flat_forward(self.decoder, x, 0, 1)
        x = x.squeeze(dim=2)  # (b k 1 h w) -> (b k h w)

        return x


if __name__ == "__main__":
    import tqdm
    import numpy as np
    from .generator import Generator as OfficialGenerator
    from ..extract_edges import extract_edges, load_data, ROOM_CLASS

    device = "cuda"

    generator = OfficialGenerator().to(device)
    batched_generator = Generator().to(device)
    batched_generator.load_state_dict(generator.state_dict())

    generator.eval()
    batched_generator.eval()

    data = load_data("3rdparty/housegan/FloorplanDataset/eval_data.npy")

    for nodes, boxes in tqdm.tqdm(data):
        edges = extract_edges(boxes)

        if len(edges) == 0:
            continue

        onehot_nodes = np.eye(len(ROOM_CLASS))[nodes - 1]

        onehot_nodes = torch.from_numpy(onehot_nodes).to(device).float()
        edges = torch.from_numpy(edges).to(device).float()

        z = torch.randn(len(onehot_nodes), generator.latent_dim, device=device)

        ref = generator(
            z,
            onehot_nodes,
            edges,
        )

        impl = batched_generator(
            torch.stack([torch.rand_like(z), z]),
            onehot_nodes,
            edges,
        )[1]

        print((ref - impl).abs().max().item())
        assert impl.isclose(ref).all()
