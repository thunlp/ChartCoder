from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle

from llava.model import *
import torch

import os
from PIL import Image

class ChartCoder:
    def __init__(
        self,
        temperature=0.1,
        max_tokens=2048,
        top_p=0.95,
        context_length=2048,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.context_length = context_length

        # Note: change to you path
        pretrained = "/mnt/afs/chartcoder"
        model_name = "llava_deepseekcoder"
        device_map = "auto"
        self.system_message = ""

        tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)
        model.eval()

        self.tokenizer = tokenizer   
        self.model = model
        self.image_processor = image_processor
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX

    def generate(self, instruction, image_path):
        image = Image.open(image_path).convert('RGB')
        prompt = self.system_message + f"### Instruction:\n{DEFAULT_IMAGE_TOKEN}\n{instruction}\n### Response:\n"
        input_ids = tokenizer_image_token(prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_tokens,
                use_cache=True)
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs

