from typing import Dict, List, Optional, Union
import numpy as np
from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5] # mean value for each channel
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5] # standard deviation value for each channel

def add_image_tokens_to_prompt(
    prefix_prompt: str,
    bos_token: str, 
    image_seq_len, 
    image_token):
    # The input text is tokenized normally
    # A <bos> token is added at the beginning and an additional \n is appended
    # the newline is essential as the model was trained with it. 
    # the tokenized text is then prefixed with a fixed number of image tokens
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"

def resize(
    image: Image.Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None,
    ) -> Image.Image:
    height, width = size
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image

def rescale(
    image: np.ndarray,
    scale: float,
    dtype: np.dtype = np.float32,
    ) -> np.ndarray:
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image

def normalize(
    image: np.ndarray,
    mean: Union[float, List[float]],
    std: Union[float, List[float]],
    ) -> np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image

def process_images(
    images: List[Image.Image],
    size: Dict[str, int],
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
    ) -> List[np.ndarray]:

    height, width = size[0], size[1]
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]

    #Convert each image to a numpy array
    images = [np.array(image) for image in images]
    # Rescale the pixel values to be in the range of [0, 1]
    images = [rescale(image, scale=rescale_factor) for image in images]
    # Normalize the pixel values
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    # Move the channel dimension to the first dimension. The model expects images in the format [ Channels, Height, Width]
    images = [np.transpose(2,0,1) for image in images]

    return images



class PaliGemmaProcessor:

    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int , image_size:int ):
        super().__init__()

        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        tokenizer_to_add = {
            "additional_special_tokens": [self.IMAGE_TOKEN]
        }
        tokenizer.add_special_tokens(tokens_to_add)

        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ] #these tokens are for object detection (bounding box)

        EXTRA_TOKENS += [
            f"<img{i:03d}>" for i in range(128)
        ] # these are tokens for segmentation
        tokenizer.add_special_tokens(EXTRA_TOKENS)

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer 

    def __call__(
        self, 
        text: List[str], 
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
        ) -> dict:
        assert len(images) == 1 and len(text) == 1, f" Received {len(images)} images and {len(text)} prompts."

        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor = 1/255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )

        # convert the list of numpy arrays to a single numpy array of shape (Batch_Size, Channels, Height, Width)
        pixel_values = np.stack(pixel_values, axis=0) # stack them along the first axis

        # Convert the numpy array to a Pytorch Tensor
        pixel_values = torch.tensor(pixel_values)

        # Prepend  a "self.image_seq_length" number of image tokens to the input text

        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_length=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # Return the input_ids and attention_mask as PyTorch Tensors
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )
        
        return_data = { "pixel_values": pixel_values, **inputs }
        
        return return_data
