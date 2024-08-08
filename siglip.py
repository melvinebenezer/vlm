import torch
import torch.nn as nn
from typing import Tuple

class SiglipVisionConfig:

    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 16,
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        num_image_tokens: int = None,
        **kwargs):

        super().__init__()

        self.hidden_size = hidden_size # size of the embeddings
        self.intermediate_size = intermediate_size # size of the linear layer in the feedforward network
        self.num_hidden_layers = num_hidden_layers # number of layers of the vision transformer stacked
        self.num_attention_heads = num_attention_heads # number of attention heads in the multi attention layer
        self.num_channels = num_channels # number of the channels in the input image (RGB = 3)
        self.patch_size = patch_size # size of the patch of the input image like 16x16
        self.image_size = image_size # size of the input image
        self.attention_drioout =  attention_dropout 
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens # number of image tokens that the vision transformer will output


class SiglipVisionEmbeddings(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid", # no padding
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embeddings = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False
        )
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [Batch_size, Channels, Height, Width]
        # [Batch_size, Channels, Height, Width] -> [Batch_size, Num_Patches, hidden_size]
        embeddings = self.patch_embedding(pixel_values)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        embeddings = embeddings + self.position_embeddings(self.position_ids)

        #[Batch_size, Num_Patches, hidden_size]
        return embeddings

class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_size, Num_Patches, hidden_size] -> [Batch_size, Num_Patches, intermediate_size]
        hidden_state = self.fc1(hidden_states)
        hidden_state = nn.functional.gelu(hidden_state, approximate="tanh")
        # [Batch_size, Num_Patches, intermediate_size] -> [Batch_size, Num_Patches, hidden_size]
        hidden_state = self.fc2(hidden_state)

        return hidden_state


class SiglipAttention(nn.Module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5 # scale factor
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden states: [Batch_size, Num_Patches, hidden_size]
        batch_size, seq_len, _ = hidden_states.size()

        # eg: [4, 1024] x [1024, 8, 128] -> [4, 8, 128]
        # query states: [Batch_size, Num_Patches, hidden_size]
        query_states = self.q_proj(hidden_states)
        # key states: [Batch_size, Num_Patches, hidden_size]
        key_states = self.k_proj(hidden_states)
        # value states: [Batch_size, Num_Patches, hidden_size]
        value_states = self.v_proj(hidden_states)

        # This makes the hidden states to be divided into num_heads and then transposed 
        # to get the shape [Batch_size, Num_Heads, Num_Patches, Head_dim]
        # This makes the heads work in parallel in part of the embedding/hidden states
        # eg. [2, 4, 1024] -> [2, 4, 8, 128] -> [2, 8, 4, 128] 
        # That is each of the 8 heads will work on the same parts of the embedding across the 4 patches/ tokens 
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # [Batch_size, Num_Heads, Num_Patches, Head_dim] x [Batch_size, Num_Heads, Head_dim, Num_Patches]
        #  -> [Batch_size, Num_Heads, Num_Patches, Num_Patches]
        # in matrix multiplication, the last dimension of the 
        # first matrix should be equal to the second dimension of the second matrix

        # calculate the attention weights using the formula QK^T/ sqrt(d_k)
        # [Batch_size, Num_Heads, Num_Patches, Num_Patches]
        # eg [2, 8, 4, 128] x [2, 8, 128, 4] -> [2, 8, 4, 4]
        # This matrix materializes the attention weights for each head

        attn_weights = (torch.matmul(query_states, key_states.transpose(-2,-1)) * self.scale)

        #lets verify the shape of the attention weights

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should have the shapa {batch_size, self.num_heads, seq_len, seq_len}, but is"
                f" {attn_weights.size()}"
            )
        
        # apply the softmax function to the attention weights
        # dims = -1 means the last dimension
        # that means apply it on the row of the matrix
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        #apply dropout to the attention weights
        # used only during training to reduce overfitting
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # lets multiply the attention weights with the value states
        # [Batch_size, Num_Heads, Num_Patches, Num_Patches] x [Batch_size, Num_Heads, Num_Patches, Head_dim]
        # -> [Batch_size, Num_Heads, Num_Patches, Head_dim]
        # eg. [2, 8, 4, 4] x [2, 8, 4, 128] -> [2, 8, 4, 128]

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"Attention output should have the shape {batch_size, self.num_heads, seq_len, self.head_dim}, but is"
                f" {attn_output.size()}"
            )

        # before concating the heads, we need to transpose the attn_output
        # [Batch_size, Num_Heads, Num_Patches, Head_dim] -> [Batch_size, Num_Patches, Num_Heads, Head_dim]
        # eg. [2, 8, 4, 128] -> [2, 4, 8, 128]
        attn_output = attn_output.transpose(1,2).contiguous()
        # contuguous is used to make sure that the tensor is stored in a contiguous block of memory
        # this helps in the reshaping of the tensor 
        # Note: reshape is used to concatenate the heads
        # eg. [2, 4, 8, 128] -> [2, 4, 1024]

        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        # by merely concatenating the heads , there is no mechanism to learn the relationship between the heads
        # so we use a linear layer to learn the relationship between the heads
        # [Batch_size, Num_Patches, hidden_size] x [hidden_size, hidden_size] -> [Batch_size, Num_Patches, hidden_size]
        # eg. [2, 4, 1024] x [1024, 1024] -> [2, 4, 1024]
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

class SiglipEncoderLayer(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size # size of the embeddings
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        #[Batch_size, Num_Patches, hidden_size] -> [Batch_size, Num_Patches, hidden_size]
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        # [Batch_size, Num_Patches, hidden_size] -> [Batch_size, Num_Patches, hidden_size]
        hidden_states, _ = self.self_attn(hidden_states)
        # [Batch_size, Num_Patches, hidden_size]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        # [Batch_size, Num_Patches, hidden_size]
        return hidden_states

class SiglipEncoder(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_size, Num_Patches, hidden_size] -> [Batch_size, Num_Patches, hidden_size]
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return hidden_states
        

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [Batch_size, Channels, Height, Width] -> [Batch_size, Num_Patches, hidden_size]

        hidden_states = self.embeddings(pixel_values)

        last_hidden_state = self.encoder(hidden_states)

        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state

    

class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # [Batch_size, Channels, Height, Width] -> [Batch_size, Num_Patches, hidden_size]
        return self.vision_model(pixel_values = pixel_values)
