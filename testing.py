import torch
from x_transformers import ViTransformerWrapper, TransformerWrapper, Encoder, Decoder
from torchviz import make_dot, make_dot_from_trace
import hiddenlayer
from torch.utils.tensorboard import SummaryWriter

encoder = ViTransformerWrapper(
    image_size = 256,
    patch_size = 32,
    attn_layers = Encoder(
        dim = 512,
        depth = 6,
        heads = 8
    )
)

decoder = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 1024,
    attn_layers = Encoder(
        dim = 512,
        depth = 2,
        heads = 8,
        cross_attend = True
    )
)

img = torch.randn(1, 3, 256, 256)
caption = torch.randint(0, 20000, (1, 1024))

encoded = encoder(img, return_embeddings = True)
writer = SummaryWriter(f'logs/net')
with SummaryWriter(comment='MultipleInput') as w:
    w.add_graph(decoder, (caption, encoded))