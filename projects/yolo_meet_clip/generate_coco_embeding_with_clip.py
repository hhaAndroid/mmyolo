from transformers import AutoTokenizer, CLIPTextModelWithProjection
import torch

model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

with open('coco_cls_name.txt') as f:
    coco_cls_str = f.read()
text_cls = coco_cls_str.strip().split('\n')

text_prompts = []
for cls in text_cls:
    text_prompts.append(f'a photo of a {cls}')

with torch.no_grad():
    inputs = tokenizer(text_prompts, padding=True, return_tensors="pt")

    outputs = model(**inputs)
    text_embeds = outputs.text_embeds
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    print(text_embeds.shape)  # torch.Size([2, 512]
    torch.save(text_embeds, 'text_embeds.pth')

# from PIL import Image
# import requests
#
from transformers import CLIPProcessor, CLIPModel
#
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
#
# inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
#
# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
# probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
