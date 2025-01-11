# %load_ext autoreload
# %autoreload 2
import gradio as gr
import numpy as np
import torch
import requests
import random
import os
import sys
import pickle
from PIL import Image
from tqdm.auto import tqdm
from datetime import datetime
from utils.gradio_utils import is_torch2_available
if is_torch2_available(): # 判断是否是pytorch 2.x
    from utils.gradio_utils import \
        AttnProcessor2_0 as AttnProcessor
else:
    from utils.gradio_utils  import AttnProcessor

import diffusers
from diffusers import StableDiffusionXLPipeline
from diffusers import DDIMScheduler
import torch.nn.functional as F
from utils.gradio_utils import cal_attn_mask_xl
import copy
import os
from diffusers.utils import load_image
from utils.utils import get_comic
from utils.style_template import styles  # 引入字典
import argparse
import json

# 创建解析器
parser = argparse.ArgumentParser(description="记录外部传入的参数")
# 添加参数
parser.add_argument('--seed', type=int, default=2047, help='种子')
parser.add_argument('--id_length', type=int, default=4, help='注意力关注区间')
parser.add_argument('--para', type=str, default="", help='区分语句')
# 解析参数
args = parser.parse_args()

device = "cuda"
# 初始化 Accelerator
# accelerator = Accelerator(mixed_precision="fp16")  # 使用 fp16 混合精度
# device = accelerator.device

# Set Config  -------------------------------------------------------------------------------------------------------------------------

## Global
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "(No style)"  # 无风格？
MAX_SEED = np.iinfo(np.int32).max
global models_dict
use_va = False
models_dict = {  # 模型路径
   "Juggernaut":"RunDiffusion/Juggernaut-XL-v8",
   "RealVision":"/mnt/d/models/RealVisXL_V4.0",
   "SDXL":"/mnt/d/models/stable-diffusion-xl-base-1.0",
   "SD1.5":"/mnt/d/models/stable-diffusion-v1-5",  # sd1.5不兼容
   "Unstable": "stablediffusionapi/sdxl-unstable-diffusers-y"
}

print(torch.cuda.is_available())

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


#################################################
########Consistent Self-Attention################
#################################################
class SpatialAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        text_context_len (`int`, defaults to 77):
            The context length of the text features.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(self, hidden_size=None, cross_attention_dim=None, id_length=4, device="cuda", dtype=torch.float16):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim  # 文本特征的通道数
        self.total_length = id_length + 1
        self.id_length = id_length
        self.id_bank = {} # 列表？字典？

    def __call__(
            self,
            attn,
            hidden_states,      # 图像特征
            encoder_hidden_states=None,   # 文本特征
            attention_mask=None,
            temb=None):
        global total_count, attn_count, cur_step, mask1024, mask4096
        global sa32, sa64
        global write   # 是否写入某些东西？？？
        global height, width

        if write:
            # print(f"white:{cur_step}")
            self.id_bank[cur_step] = [hidden_states[:self.id_length], hidden_states[self.id_length:]]
        else:
            encoder_hidden_states = torch.cat((self.id_bank[cur_step][0].to(self.device), hidden_states[:1],
                                               self.id_bank[cur_step][1].to(self.device), hidden_states[1:])) # ???
        # skip in early step
        if cur_step < 5:  # 如果cur_step小于5，则调用__call2__
            hidden_states = self.__call2__(attn, hidden_states, encoder_hidden_states, attention_mask, temb) # 有encoder隐藏层

        else:  # 256 1024 4096  如果cur_step大于等于5，则根据一定概率切换使用__call1__或__call2__
            random_number = random.random()
            if cur_step < 20:
                rand_num = 0.3
            else:
                rand_num = 0.1
            if random_number > rand_num:
                if not write:
                    if hidden_states.shape[1] == (height // 32) * (width // 32):
                        attention_mask = mask1024[mask1024.shape[0] // self.total_length * self.id_length:]
                    else:
                        attention_mask = mask4096[mask4096.shape[0] // self.total_length * self.id_length:]
                else:
                    if hidden_states.shape[1] == (height // 32) * (width // 32):
                        attention_mask = mask1024[:mask1024.shape[0] // self.total_length * self.id_length,
                                         :mask1024.shape[0] // self.total_length * self.id_length]
                    else:
                        attention_mask = mask4096[:mask4096.shape[0] // self.total_length * self.id_length,
                                         :mask4096.shape[0] // self.total_length * self.id_length]

                hidden_states = self.__call1__(attn, hidden_states, encoder_hidden_states, attention_mask, temb) # 有encoder隐藏层
            else:
                hidden_states = self.__call2__(attn, hidden_states, None, attention_mask, temb)

        attn_count += 1
        if attn_count == total_count:
            attn_count = 0
            cur_step += 1
            mask1024, mask4096 = cal_attn_mask_xl(self.total_length, self.id_length, sa32, sa64, height, width,
                                                  device=self.device, dtype=self.dtype)

        return hidden_states

    def __call1__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            total_batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(total_batch_size, channel, height * width).transpose(1, 2)

        total_batch_size, nums_token, channel = hidden_states.shape
        img_nums = total_batch_size // 2
        hidden_states = hidden_states.view(-1, img_nums, nums_token, channel).reshape(-1, img_nums * nums_token,
                                                                                      channel)

        batch_size, sequence_length, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1, self.id_length + 1, nums_token, channel).reshape(-1, (self.id_length + 1) * nums_token, channel) # self.id_length + 1 图什么

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(total_batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(total_batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        # print(hidden_states.shape)
        return hidden_states

    def __call2__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, channel = (
            hidden_states.shape
        )
        # print(hidden_states.shape)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1, self.id_length + 1, sequence_length,channel).reshape(-1, (self.id_length + 1) * sequence_length, channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def set_attention_processor(unet, id_length):
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            if name.startswith("up_blocks"):
                attn_procs[name] = SpatialAttnProcessor2_0(id_length=id_length)
            else:
                attn_procs[name] = AttnProcessor()
        else:
            attn_procs[name] = AttnProcessor()

    unet.set_attn_processor(attn_procs)

# Load Pipeline ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————


global attn_count, total_count, id_length, total_length,cur_step, cur_model_type
global write
global sa32, sa64
global height,width
attn_count = 0
total_count = 0
cur_step = 0
id_length = args.id_length                  # 一次多少个prompt注入
total_length = 5                            # 主体一致性关注的滑动窗口大小！！！！！！！！！！
cur_model_type = ""
batch_size = 1
global attn_procs,unet
attn_procs = {}
###
write = False
### strength of consistent self-attention: the larger, the stronger
sa32 = 0.5
sa64 = 0.5
### Res. of the Generated Comics. Please Note: SDXL models may do worse in a low-resolution!
height = 512
width = 512
### 768

global pipe
global sd_model_path

model_version = "RealVision"
sd_model_path = models_dict[model_version] #"SG161222/RealVisXL_V4.0" 用这个模型效果就很好，很能一致性，但SDXL模型就很差
### LOAD Stable Diffusion Pipeline


pipe = StableDiffusionXLPipeline.from_pretrained(sd_model_path, torch_dtype=torch.float16, use_safetensors=False)
pipe = pipe.to(device)
pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2) # SDXL 管道的 FreeU 机制，调整相关的参数（s1、s2、b1、b2）以优化性能

scheduler_name = pipe.scheduler.__class__.__name__
print(scheduler_name, "看看原始scheduler")

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.set_timesteps(50)
unet = pipe.unet

### Insert PairedAttention
for name in unet.attn_processors.keys(): # ["up_blocks.1", "up_blocks.2", "up_blocks.3", "down_blocks.1", "down_blocks.2", ...]
    cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
    if name.startswith("mid_block"):
        hidden_size = unet.config.block_out_channels[-1]  # 这逼 hidden_size 记录下来又不传进去？

    elif name.startswith("up_blocks"):
        block_id = int(name[len("up_blocks.")]) # 提取"up_blocks."后面跟的块ID
        hidden_size = list(reversed(unet.config.block_out_channels))[block_id]

    elif name.startswith("down_blocks"):
        block_id = int(name[len("down_blocks.")])
        hidden_size = unet.config.block_out_channels[block_id]

    if cross_attention_dim is None and (name.startswith("up_blocks")): # 只换自注意力层和上采样层
        attn_procs[name] = SpatialAttnProcessor2_0(id_length = id_length)
        # 记录每个自注意力块

        total_count +=1
    else:
        attn_procs[name] = AttnProcessor()

print("successsfully load consistent self-attention")
print(f"number of the processor : {total_count}")
unet.set_attn_processor(copy.deepcopy(attn_procs)) # 拷贝一份attn_procs

global mask1024,mask4096
mask1024, mask4096 = cal_attn_mask_xl(total_length,id_length,sa32,sa64,height,width,device=device,dtype= torch.float16)


# Create the text description for the comics -----------------------------------------------------------------------------------

guidance_scale = 5.0
seed = args.seed  # 2047
sa32 = 0.5
sa64 = 0.5
id_length = args.id_length
num_steps = 30   # 50 - 30
# 步数太大爆内存

def create_folder_and_save_prompts(model, style_name, general_prompt, prompts, id_prompts, negative_prompt, para=args.para):
    # 组合路径：用 style_name 和 general_prompt 创建文件夹名
    folder_name = f"images/{style_name}_{para}_{general_prompt}"

    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # 定义 prompt.json 文件的路径
    json_file_path = os.path.join(folder_name, 'prompt.json')

    # 将 id_prompts 和 negative_prompt 写入字典
    data = {
        "model": model,
        "style_name": style_name,
        "general_prompt": general_prompt,
        "all_prompts": prompts,
        "negative_prompt": negative_prompt,
        "change_prompts": id_prompts
    }

    # 将数据写入 JSON 文件
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)  # 使用缩进使 JSON 文件更易读

    print(f"文件已成功写入到 {json_file_path}")
    return folder_name

general_prompt = "A playful small dog with a wagging tail, looking amused"
prompt_array = [
    "chasing its own tail in circles, barking excitedly",
    "rolling over on the grass, kicking its legs in joy",
    "jumping up to catch soap bubbles blown by children",
    "hopping between puddles on a rainy day, splashing around",
    "tugging on a squeaky toy with its owner, refusing to let go",
    "sneaking a playful nibble on the owner's sock, then running away",
]


negative_prompt = "deformed, bad anatomy, disfigured, poorly drawn face, mutation, extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, blurry, watermarks, oversaturated, distorted hands, amputation"

### Set the generated Style
style_name = "Japanese Anime" # 定义风格

def apply_style_positive(style_name: str, positive: str):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive)
def apply_style(style_name: str, positives: list, negative: str = ""):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME]) # 后者是默认风格
    return [p.replace("{prompt}", positive) for positive in positives], n + ',' + negative  # 加入消极抑制 prompt

setup_seed(seed)
generator = torch.Generator(device=device).manual_seed(seed)

prompts = [general_prompt+","+prompt+"," for prompt in prompt_array]

id_prompts = prompts[0:id_length]
real_prompts = prompts[id_length:] #为啥分两段？

torch.cuda.empty_cache()
write = True
cur_step = 0
attn_count = 0
id_prompts, negative_prompt = apply_style(style_name, id_prompts, negative_prompt)

print(id_prompts) # 要一次性注入多个prompt，对内存需求大
################################################################################################ gannima
image_path = create_folder_and_save_prompts(model_version, style_name, general_prompt, prompt_array, id_prompts, negative_prompt)

id_images = pipe(
        id_prompts,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        negative_prompt=negative_prompt, # 忘记抑制了
        generator=generator
    ).images


for i, id_image in enumerate(id_images):
    id_image.save(f"{image_path}/sd={seed:06d}_gc={guidance_scale}_id_images{i}.png")

del id_images
torch.cuda.empty_cache()

write = False

# exit()         #——————————————————————————————————————————————————————————————————————————————————————————————————————————

real_images = []
for real_prompt in real_prompts:
    cur_step = 0
    real_prompt = apply_style_positive(style_name, real_prompt)  # 不加风格负面抑制？单图生成无一致性？

    image = pipe(
        real_prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        generator=generator
    ).images[0]

    real_images.append(image)

for i, real_image in enumerate(real_images):
    real_image.save(f"{image_path}/sd={seed:06d}_gc={guidance_scale}_real_images{i}.png")

del real_images
torch.cuda.empty_cache()

exit()

## Continued Creation
### From now on, you can create endless stories about this character without worrying about memory constraints.

new_prompt_array = ["siting on the sofa", "on the bed, at night "]
new_prompts = [general_prompt+","+prompt for prompt in new_prompt_array]
new_images = []
for new_prompt in new_prompts :
    cur_step = 0
    new_prompt = apply_style_positive(style_name, new_prompt)

    image = pipe(
        new_prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        height=height, width=width,
        negative_prompt=negative_prompt,
        generator=generator
    ).images[0]
    new_images.append(image)

for i, new_image in enumerate(new_images):
    new_image.save(f"images/new_images_seed={seed:06d}_gc={guidance_scale}_{i}.png")


### Make pictures into comics
###
total_images = id_images + real_images + new_images
from PIL import Image,ImageOps,ImageDraw, ImageFont
#### LOAD Fonts, can also replace with any Fonts you have!
font = ImageFont.truetype("./fonts/Inkfree.ttf", 30)

# import importlib
# import utils.utils
# importlib.reload(utils)
from utils.utils import get_row_image
from utils.utils import get_row_image
from utils.utils import get_comic_4panel

comics = get_comic_4panel(total_images, captions = prompt_array+ new_prompts,font = font )
for comic in comics:
    display(comic)