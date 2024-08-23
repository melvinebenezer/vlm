import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from siglip import SiglipVisionConfig, SiglipVisionModel

class KVCache:

    def __init__(self, ) -> None:
        self.key_cache = List[torch.Tensor] = []
        self.value_cache = List[torch.Tensor] = []

    def num_items(self):
        if len(self.key_cache) == 0:
            return 0
        else:
            # [batch_size, num_heads, seq_len, head_dim] hence the -2 for the seq_len
            return self.key_cache[0].shape[-2]
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) == layer_idx:
            # if we never added anything to the kv cache of this layer, let's create it
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # otherwise we concatenate the new keys and values with the existing ones
            # each tensor has shape [batch_size, num_heads, seq_len, head_dim]
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        # we can return the existing keys + the new ones
        return self.key_cache[layer_idx], self.value_cache[layer_idx]



class GemmaConfig():

    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim # this is set to the head_dim, i,e each head has its own rotary embedding
        self.max_position_embeddings = max_position_embeddings # tells us the max sequence length
        self.base = base

        # Calculate the theta according to the formula theta_i = base^(2i/dim) where i = 0, 1, 2, 3, ... dim//2
        # this is slightly different from the paper, where the theta is calculated as base^(2i/dim) where i = 0, 1, 2, 3, ... dim//2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, presistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_dim]
        self.inv_freq.to(x.device)
        #Copy the inv_freq tensor for batch in the sequence
        # inv_freq_expanded: [Batch_Size, Head_Dim // 2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # position_ids_expanded: [Batch_Size, 1, Seq_Len]
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type_as
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # Multiply each theta by the position (which is the argument of the sin and cos functions)
            # freqs: [Batch_Size, Head_Dim//2, Seq_len] @ [Batch_Size, 1, Seq_len] --> [Batch_Size, Seq_Len, Head_Dim //2]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # emb: [Batch_Size, Seq_len, Head_Dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            # cos, sin: [Batch_Size, Seq_len, Head_Dim]
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def rotate_half(x):
        # slightly different from the paper
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

        
    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim) # Add the head dimension
        sin = sin.unsqueeze(unsqueeze_dim) # Add the head dimension
        # Apply the formula9340 of the Rotary Encoding paper
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
    


class GemmaAttention(nn.Module):

    def __init__(self, config: GemmaConfig, layer_idx: Optiona[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0, "Hidden size must be divisible by the number of heads"

        # NUmber of heads = 8
        # Hidden_Size = 1024
        # Head_Dim = 1024 / 8 = 128
        # Wq = [1024, 8 * 128] = [ 1024, 1024]
        # in grouped query attention
        # the KV are compressed to save memory transfer. There is slight change in accuracy

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_state)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q-len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(keys_states, value_states, self.layer_idx)
    
        # here we are using the naive implementation of the attention
        # in fact reversing the grouped query attention optimization
        # the flash attention can leverage from the grouped query attention
        # or any kernel

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Q * K^T / sqrt(head_dim) 
        # [Batch_Size, Num_Heads, Seq_Len, Head_Dim] * [Batch_Size, Num_Heads, Head_Dim, Seq_Len_KV] -> [Batch_Size, Num_Heads, Seq_Len, Seq_Len_KV]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        assert attention_mask is not None, "Not implemented Yet!"
        attn_weights = attn_weights + attention_mask


        # Apply the softmax
        # [Batch_Size, Num_Heads, Seq_Len, Seq_Len_KV]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtypes)
        # Apply the dropout only on training
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropput, training=self.training)
        # multiply the attention weights with the values states
        # [Batch_Size, Num_Heads, Seq_Len, Seq_Len_KV] * [Batch_Size, Num_Heads, Seq_Len_KV, Head_Dim] -> [Batch_Size, Num_Heads, Seq_Len, Head_Dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size()  != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"attn_output should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f"{attn_output.size()}"
            )
        
        # Make sure the sequence length is the second dimension
        # [Batch_size, Num_Heads_Q, Seq_Len_Q, Head_Dim] -> [Batch_Size, Seq_Len_Q, Num_Heads_Q, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # Concatenate all the heads together
        attn_output = attn_output.view(bsz, q_len, -1)

        # We need to mix the heads together than just concatenation of all head dimensions
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights 







        




class GemmaMLP(nn.Module):

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    
    def forward(self, x):
        # Equivalent to 
        # y = self.gate_proj(x)
        # y = torch.gelu(y, approximate="tanh")
        # j = self.up_proj(x)
        # z = y * j # [Batch_Size, Seq_Len, Intermediate_Size]
        # z = self.down_proj(z) # [Batch_Size, Seq_Len, Intermediate_Size] -> [Batch_Size, Seq_Len, Hidden_Size]
        # return z

        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class GemmaDecoderLayer(nn.Module):

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        #[ Batch_Size, Seq_len]
        hidden_states = self.input_layernorm(hidden_states)

        #[Batch_Size, Seq_Len, Hidden_Size]
        hidden_states, _, = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )

        #[Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = residual + hidden_states

        #[Batch_Size, Seq_Len, Hidden_Size]
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)

        #[Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.mlp(hidden_states)

        #[Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = residual + hidden_states

        return hidden_states


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        # this  calculating the 1 / sqrt(mean(x^2)) and then multiply it with x. same as ai/RMS(a) in the paper
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x)
        # llama does x.to(float16) * w while Gemma does (x*w).to(float16)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)



class GemmaModel(nn.Module):

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # tie the weights of the embeddings and the logits layer
        # this layer is [vocab_size, hidden_size] can be shared with the logits layer [hidden_size, vocab_size]
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_id) for layer_id in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = inputs_embeds
        #[Batch_Size, Seq_len, Hidden_Size]
        normalizer = torch.tensor(self.config.hidden_size ** 0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        hidden_states = self.norm(hidden_states)

        #[Batch_Size, Seq_Len, Hidden_Size]
        return hidden_states



class GemmaForCausalLM(nn.Module):

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    # we share the weights between the embeddings and the logits layer
    def tie_weights(self):
        self.lm_head.weight = self.model.get_input_embeddings().weight
    
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None
        ) -> Tuple:
        outputs = self.model(
            attention_mask=attention_mask,
            postion_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache
        
        return return_data




# convert the image features dims to the hidden size of the LLM
class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        #[Batch_Size, Num_Patches, Hidden_Size] -> [Batch_Size, Num_Patches, Projection_Dim]
        hidden_states = self.linear(image_features)
        return hidden_states


class PaliGemmaConfig:

    def __init__(
        self,
        vision_config: SiglipVisionConfig,
        text_config=None,
        ignore_index=-100,
        image_token_index=2560000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim



class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_encoder = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
    
    def tie_weights(self):
        self.language_model.tie_weights()
    

    def _merge_input_ids_with_images_features(
        self, images_features, inputs_embeds, input_ids, attention_mask, kv_cache
    ):
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        scaled_image_features = image_features / (self.config.hidden_size ** 0.5)

        # combine the embeddings of the image tokens, text tokens and mask out all the padding tokens
        final_embeddings = torch.zeros(batch_size, sequence_length, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        # Shape: [Batch_Size, Seq_Len]. True for text tokens
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        # Shape: [Batch_Size, Seq_Len]. True for image tokens
        image_mask = input_ids == self.config.image_token_index
        # Shape: [Batch_Size, Seq_Len]. True for padding tokens
        pad_mask = input_ids == self.pad_token_id

        # We need to expand the masks to the embedding dims other wise we can not use torch.Where
        # eg. [Batch_Size, Seq_Len] -> [Batch_Size, Seq_Len, Embed_Dim]
        # lets say <image_token> is 567, <BOS> is 1, \n is 2
        # input_ids = [ 567, 567, 567, 1, 65, 78, 99, 2]
        # text_mask = [ 0, 0, 0, 1, 1, 1, 1, 0]
        # image_mask = [ 1, 1, 1, 0, 0, 0, 0, 0]
        # pad_mask = [ 0, 0, 0, 0, 0, 0, 0, 0] -- we dont have any padding tokens in this example
        text_mask_expanded = text_mask.unseqeeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        final_embeddings = torch.where(text_mask_expanded, inputs_embeds, final_embeddings)
        #Insert image embeddings. We cant use torch.where because the sequence length of the scaled_image_features
        # is not equal the sequence length of the final_embeddings
        # but does the same as torch.where
        final_embeddings = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        final_embeddings = torch.where(pad_mask_expanded, torch.zeros_like(final_embeddings), final_embeddings)

        ### CREATE THE ATTENTION MASK

        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            # Do not mask any token, because we're in the prefill phase
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Since we are generating tokens, the query must be a single token
            assert q_len == 1, "Query length must be 1 during generation"
            kv_len = kv_cache.num_items()
            # when using the kv cache, we dont need to mask out anything because we are generating only 
            # one token at a time
            causal_mask = torch.full((batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device)

        # add the head dimension
        # [Batch_Size, Seq_Len, Seq_Len] -> [Batch_Size, Num_Heads, Seq_Len, Seq_Len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids =  attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            #create a position_ids based on the size of the attention mask
            # For masked tokens, use the number 1 as position.
            position_ids = (attention_mask.cumsum(-1).masked_fill_((attention_mask == 0), 1) - 1)


        return final_embeddings, causal_mask, position_ids


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:

        # 1. Extract the input embeddings
        # shape: (Batch_size, Seq_len, Hidden_size)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 2. Merge the text and images
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Size]
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        # [Batch_Size, Num_Patches, Embed_Size] -> [Batch_Size, Num_Patches, Hidden_Size]
        # here the Embed_Size is from the the vision tower, to make the dimensions match
        # we need to project the image features to the hidden size of the LLM
        images_features = self.multi_modal_encoder(selected_image_feature)


        # 3. Merge the embeddings of the text and the image tokens
        inputs_embeds, attention_mask, position_ids = \
            self._merge_input_ids_with_images_features(
                images_features,
                inputs_embeds,
                input_ids,
                attention_mask,
                kv_cache
            )

        #4. Forward pass through the language model

        outputs = self.language_model(
            attention_mask= attention_mask,
            position_ids= position_ids,
            inputs_embeds= inputs_embeds,
            kv_cache= kv_cache
        )
        
        return outputs



