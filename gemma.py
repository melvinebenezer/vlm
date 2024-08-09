import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from siglip import SiglipVisionConfig, SiglipVisionModel

clas GemmaConfig():

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



