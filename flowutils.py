import pdb

import normflows as nf
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from normflows.distributions import BaseDistribution


def sanitize_locals(args_dict, ignore_keys=None):

    if ignore_keys is None:
        ignore_keys = []

    if not isinstance(ignore_keys, list):
        ignore_keys = [ignore_keys]

    _dict = args_dict.copy()
    _dict.pop("self")
    class_name = _dict.pop("__class__").__name__
    class_params = {k: v for k, v in _dict.items() if k not in ignore_keys}

    return {class_name: class_params}


def build_flows(
    latent_size, num_flows=4, num_blocks_per_flow=2, hidden_units=128, context_size=64
):
    # Define flows

    flows = []

    flows.append(
        nf.flows.MaskedAffineAutoregressive(
            latent_size,
            hidden_features=hidden_units,
            num_blocks=num_blocks_per_flow,
            context_features=context_size,
        )
    )

    for i in range(num_flows):
        flows += [
            nf.flows.CoupledRationalQuadraticSpline(
                latent_size,
                num_blocks=num_blocks_per_flow,
                num_hidden_channels=hidden_units,
                num_context_channels=context_size,
            )
        ]
        flows += [nf.flows.LULinearPermute(latent_size)]

    # Set base distribution

    context_encoder = nn.Sequential(
        nn.Linear(context_size, context_size),
        nn.SiLU(),
        # output mean and scales for K=latent_size dimensions
        nn.Linear(context_size, latent_size * 2),
    )

    q0 = ConditionalDiagGaussian(latent_size, context_encoder)

    # Construct flow model
    model = nf.ConditionalNormalizingFlow(q0, flows)

    return model


class ConditionalDiagGaussian(BaseDistribution):
    """
    Conditional multivariate Gaussian distribution with diagonal
    covariance matrix, parameters are obtained by a context encoder,
    context meaning the variable to condition on
    """

    def __init__(self, shape, context_encoder):
        """Constructor

        Args:
          shape: Tuple with shape of data, if int shape has one dimension
          context_encoder: Computes mean and log of the standard deviation
          of the Gaussian, mean is the first half of the last dimension
          of the encoder output, log of the standard deviation the second
          half
        """
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)
        self.context_encoder = context_encoder

    def forward(self, num_samples=1, context=None):
        encoder_output = self.context_encoder(context)
        split_ind = encoder_output.shape[-1] // 2
        mean = encoder_output[..., :split_ind]
        log_scale = encoder_output[..., split_ind:]
        eps = torch.randn(
            (num_samples,) + self.shape, dtype=mean.dtype, device=mean.device
        )
        z = mean + torch.exp(log_scale) * eps
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow(eps, 2), list(range(1, self.n_dim + 1))
        )
        return z, log_p

    def log_prob(self, z, context=None):
        encoder_output = self.context_encoder(context)
        split_ind = encoder_output.shape[-1] // 2
        mean = encoder_output[..., :split_ind]
        log_scale = encoder_output[..., split_ind:]
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow((z - mean) / torch.exp(log_scale), 2),
            list(range(1, self.n_dim + 1)),
        )
        return log_p


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if (
            self.cached_penc is not None
            and self.cached_penc.shape[:2] == tensor.shape[1:3]
        ):
            return self.cached_penc

        self.cached_penc = None
        batch_size, orig_ch, x, y = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros(
            (x, y, self.channels * 2),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb

        return self.cached_penc


class SpatialNormer(nn.Module):
    def __init__(
        self,
        in_channels,  # channels will be number of sigma scales in input
        kernel_size=3,
        stride=2,
        padding=1,
    ):
        """
        Note that the convolution will reduce the channel dimension
        So (b, num_sigmas, c, h, w) -> (b, num_sigmas, new_h , new_w)
        """
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size,
            # This is the real trick that ensures each
            # sigma dimension is normed separately
            groups=in_channels,
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False,
        )
        self.conv.weight.data.fill_(1)  # all ones weights
        self.conv.weight.requires_grad = False  # freeze weights

    @torch.no_grad()
    def forward(self, x):
        return self.conv(x.square()).pow_(0.5).squeeze(2)


class PatchFlow(torch.nn.Module):
    def __init__(
        self,
        input_size,
        patch_size=3,
        context_embedding_size=128,
        num_flows=4,
        num_blocks_per_flow=2,
        hidden_units=128,
    ):
        super().__init__()

        self.config = sanitize_locals(locals(), ignore_keys="input_size")

        num_sigmas, c, h, w = input_size
        self.local_pooler = SpatialNormer(
            in_channels=num_sigmas, kernel_size=patch_size
        )
        self.flows = build_flows(
            latent_size=num_sigmas,
            context_size=context_embedding_size,
            num_flows=num_flows,
            num_blocks_per_flow=num_blocks_per_flow,
            hidden_units=hidden_units,
        )
        self.position_encoding = PositionalEncoding2D(channels=context_embedding_size)

        # caching pos encs
        _, _, ctx_h, ctw_w = self.local_pooler(
            torch.empty((1, num_sigmas, c, h, w))
        ).shape
        self.position_encoding(torch.empty(1, 1, ctx_h, ctw_w))
        assert self.position_encoding.cached_penc.shape[-1] == context_embedding_size

    def init_weights(self):
        # Initialize weights with Xavier
        linear_modules = list(
            filter(lambda m: isinstance(m, nn.Linear), self.flows.modules())
        )
        total = len(linear_modules)

        for idx, m in enumerate(linear_modules):
            # Last layer gets init w/ zeros
            if idx == total - 1:
                nn.init.zeros_(m.weight.data)
            else:
                nn.init.xavier_uniform_(m.weight.data)

            if m.bias is not None:
                nn.init.zeros_(m.bias.data)

    def forward(self, x, chunk_size=32):
        b, s, c, h, w = x.shape
        x_norm = self.local_pooler(x)
        _, _, new_h, new_w = x_norm.shape
        context = self.position_encoding(x_norm)

        # (Patches * batch) x channels
        local_ctx = rearrange(context, "h w c -> (h w) c")
        patches = rearrange(x_norm, "b c h w -> (h w) b c")

        nchunks = (patches.shape[0] + chunk_size - 1) // chunk_size
        patches = patches.chunk(nchunks, dim=0)
        ctx_chunks = local_ctx.chunk(nchunks, dim=0)
        patch_logpx = []

        # gc = repeat(global_ctx, "b c -> (n b) c", n=self.patch_batch_size)

        for p, ctx in zip(patches, ctx_chunks):

            # num patches in chunk (<= chunk_size)
            n = p.shape[0]
            ctx = repeat(ctx, "n c -> (n b) c", b=b)
            p = rearrange(p, "n b c -> (n b) c")

            # Compute log densities for each patch
            logpx = self.flows.log_prob(p, context=ctx)
            logpx = rearrange(logpx, "(n b) -> n b", n=n, b=b)
            patch_logpx.append(logpx)

        # Convert back to image
        logpx = torch.cat(patch_logpx, dim=0)
        logpx = rearrange(logpx, "(h w) b -> b 1 h w", b=b, h=new_h, w=new_w)

        return logpx.contiguous()

    @staticmethod
    def stochastic_step(
        scores, x_batch, flow_model, opt=None, train=False, n_patches=32, device="cpu"
    ):
        if train:
            flow_model.train()
            opt.zero_grad(set_to_none=True)
        else:
            flow_model.eval()

        patches, context = PatchFlow.get_random_patches(
            scores, x_batch, flow_model, n_patches
        )

        patch_feature = patches.to(device)
        context_vector = context.to(device)
        patch_feature = rearrange(patch_feature, "n b c -> (n b) c")
        context_vector = rearrange(context_vector, "n b c -> (n b) c")

        # global_pooled_image = flow_model.global_pooler(x_batch)
        # global_context = flow_model.global_attention(global_pooled_image)
        # gctx = repeat(global_context, "b c -> (n b) c", n=n_patches)

        # # Concatenate global context to local context
        # context_vector = torch.cat([context_vector, gctx], dim=1)

        # z, ldj = flow_model.flows.inverse_and_log_det(
        #     patch_feature,
        #     context=context_vector,
        # )

        loss = flow_model.flows.forward_kld(patch_feature, context_vector)
        loss *= n_patches

        if train:
            loss.backward()
            opt.step()

        return loss.item() / n_patches

    @staticmethod
    def get_random_patches(scores, x_batch, flow_model, n_patches):
        b = scores.shape[0]
        h = flow_model.local_pooler(scores)
        patches = rearrange(h, "b c h w -> (h w) b c")

        context = flow_model.position_encoding(h)
        context = rearrange(context, "h w c -> (h w) c")
        context = repeat(context, "n c -> n b c", b=b)

        # conserve gpu memory
        patches = patches.cpu()
        context = context.cpu()

        # Get random patches
        total_patches = patches.shape[0]
        shuffled_idx = torch.randperm(total_patches)
        rand_idx_batch = shuffled_idx[:n_patches]

        return patches[rand_idx_batch], context[rand_idx_batch]
