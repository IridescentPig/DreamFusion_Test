from diffusion.sd_utils import StableDiffusion
import torch
import os
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm

model_key = './models/diffusion/stable-diffusion-v1-5'

guidance = StableDiffusion(model_key, 'cuda', False, True)

prompt = 'grasp the cup'
negative = ''
uncond_z = guidance.get_text_embeds(negative)
pos_z = guidance.get_text_embeds(prompt)
text_z = torch.cat([uncond_z, pos_z], dim=0)

img_base_dir = './figs/grasp_mug_4/'
transform = T.ToTensor()
total = 120
rounds = 20
losses = []

with torch.no_grad():
    for i in tqdm(range(total + 1)):
        img_path = img_base_dir + f'{i}.png'
        img = Image.open(img_path)
        w = img.width
        h = img.height
        img = transform(img) # (3, h, w)
        img = torch.unsqueeze(img, 0) # (1, 3, h, w)
        img = img.to('cuda')
        avg_loss = 0.
        for j in range(rounds):
            loss = guidance.train_step(text_z, img)
            avg_loss += loss.item()
        
        avg_loss /= rounds
        losses.append(avg_loss)

plt.plot(range(total + 1), losses)
plt.savefig('./sds_result_4.png')