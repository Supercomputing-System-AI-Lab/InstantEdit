from src.pipeline import EditPipeline
import os
import torch
from PIL import Image
import argparse
import torch.nn.functional as nnf
from typing import Optional
import utils_.ptp_utils as ptp_utils
import numpy as np
import utils_.seq_aligner as seq_aligner
from diffusers import  ControlNetModel
from src.scheduler_perflow import PeRFlowScheduler
import cv2
from src.sd_inversion_pipeline import SDDDIMPipelineControl
import copy
import json
from pie_evaluator.evaluate import MetricsCalculator, calculate_metric

MAX_NUM_WORDS = 77

### Initialized necessary components
device = "cuda" if torch.cuda.is_available() else "cpu"
controlnet = ControlNetModel.from_pretrained(
       "lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16
    )
pipe = EditPipeline.from_pretrained("hansyan/perflow-sd15-dreamshaper", torch_dtype=torch.float16, controlnet=controlnet, safety_checker=None).to(device)
pipe.scheduler = PeRFlowScheduler.from_config(pipe.scheduler.config, prediction_type="diff_eps", num_time_windows=4)
pipe_inv = SDDDIMPipelineControl
pipe_inversion = pipe_inv(**copy.deepcopy(pipe.components)).to(device)
pipe_inversion.scheduler = PeRFlowScheduler.from_config(pipe_inversion.scheduler.config)
tokenizer = pipe.tokenizer
encoder = pipe.text_encoder


def rle2mask(rle_mask):
    mask = np.zeros(512**2, dtype=np.uint8)
    # Apply the RLE to reconstruct the mask
    for i in range(0, len(rle_mask), 2):
        start_index = rle_mask[i]
        run_length = rle_mask[i + 1]
        mask[start_index:start_index + run_length] = 1
    
    return mask


class LocalBlend:
    
    def get_mask(self,x_t,maps,word_idx, thresh, i=None):
        maps = maps * word_idx.reshape(1,1,1,1,-1)
        maps = (maps[:,:,:,:,1:self.len-1]).mean(0,keepdim=True)
        maps = (maps).max(-1)[0]
        maps = nnf.interpolate(maps, size=(x_t.shape[2:]))
        maps = maps / maps.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
        mask = maps > thresh
        return mask


    def save_image(self,mask,i, caption):
        image = mask[0, 0, :, :]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.cpu().numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        if not os.path.exists(f"inter/{caption}"):
           os.makedirs(f"inter/{caption}", exist_ok=True) 
        ptp_utils.save_images(image, f"inter/{caption}/{i}.jpg")
        

    def __call__(self, i, x_s, x_t,  attention_store, alpha_prod, temperature=0.15):
        maps = attention_store["down_cross"] + attention_store["up_cross"]
        h,w = x_t.shape[2],x_t.shape[3]
        h , w = ((h+1)//2+1)//2, ((w+1)//2+1)//2
        maps = [item.reshape(2, -1, 1, h // int((h*w/item.shape[-2])**0.5),  w // int((h*w/item.shape[-2])**0.5), MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps_m = maps[1,:]
        thresh_e = temperature / alpha_prod ** (0.5)
        if thresh_e < self.thresh_e:
          thresh_e = self.thresh_e
        mask_e = self.get_mask(x_t, maps_m, self.alpha_e, thresh_e, i)
       
        if torch.sum(mask_e.to(torch.int)) == 0:
            x_t_out = x_t
        else:
            x_t_out = torch.where(mask_e, x_t, x_s)
        
        return x_t_out, mask_e

    def __init__(self,thresh_e=0.3):
        self.thresh_e = thresh_e
        
    def set_map(self, alpha_e, len):
        self.alpha_e = alpha_e
        self.len = len


class AttentionControlEdit:
    def init_blend(self, prompts, prompt_specifiers, tokenizer, encoder, device):
        alpha_e = seq_aligner.get_local_mapper(prompts, prompt_specifiers, tokenizer, encoder, device)
        t_len = len(tokenizer(prompts[1])["input_ids"])
        self.local_blend.set_map(alpha_e, t_len)

    def step_callback(self,i, t, x_s, x_t, alpha_prod):
        if (self.local_blend is not None) and (i>0):
            x_t, mask = self.local_blend(i, x_s, x_t, self.attention_store, alpha_prod)
        else:
            mask = None
        return x_t, mask
    
    def register_attention(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] == 16 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn
    
    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()
    
    def get_empty_store(self):
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        self.register_attention(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers//2:
            self.between_steps()
        return attn

    def __init__(self, local_blend: Optional[LocalBlend]):
        self.local_blend = local_blend
        self.cur_step = 0
        self.attention_store = {}
        self.step_store = self.get_empty_store()
        self.num_att_layers = -1
        self.cur_att_layer = 0



def inference(source_prompt, control_image, target_prompt, positive_prompt, negative_prompt, guidance_dpg, guidance_cfg, local=" ",num_inference_steps=4, controlnet_conditioning_scale=0.4,
              width=512, height=512, seed=0, img=None, thresh_e=0.3, latents=None, all_latents=None):

    torch.manual_seed(seed)
    ratio = min(height / img.height, width / img.width)
    img = img.resize((int(img.width * ratio), int(img.height * ratio)))
    local_blend = LocalBlend(thresh_e=thresh_e)
    controller = AttentionControlEdit(local_blend)
    controller.init_blend([source_prompt, target_prompt], [local], tokenizer, encoder, device)
    ptp_utils.register_attention_control(pipe, controller)

    results = pipe(prompt= target_prompt,
                   source_prompt=source_prompt,
                   positive_prompt=positive_prompt,
                   negative_prompt=negative_prompt,
                   control_image=control_image,
                   image = img,
                   latents = latents,
                   all_latents=all_latents,
                   num_inference_steps=num_inference_steps,
                   cfg_guidance_scale=guidance_cfg,
                   dpg_guidance_scale=guidance_dpg,
                   controlnet_conditioning_scale = args.controlnet_conditioning_scale,
                   callback = controller.step_callback
                   )
    
    return results.images[0]


def main(args):
    with open(f'{args.dataset_path}/mapping_file.json', 'r') as file:
        dataset = json.load(file)

    evaluator = MetricsCalculator(device)

    metrics = {
        "psnr":[0,0],
        "lpips":[0,0],
        "mse":[0,0],
        "ssim":[0,0],
        "structure_distance":[0,0],
        "psnr_unedit_part":[0,0],
        "lpips_unedit_part":[0,0],
        "mse_unedit_part":[0,0],
        "ssim_unedit_part":[0,0],
        "structure_distance_unedit_part":[0,0],
        "psnr_edit_part":[0,0],
        "lpips_edit_part":[0,0],
        "mse_edit_part":[0,0],
        "ssim_edit_part":[0,0],
        "structure_distance_edit_part":[0,0],
        "clip_similarity_source_image":[0,0],
        "clip_similarity_target_image":[0,0],
        "clip_similarity_target_image_edit_part":[0,0],
    }
    
    with torch.no_grad():
        for id,val in enumerate(dataset.values()):
            group = val["image_path"].split('_')[0]
            if group not in ['0']:
                continue


            img_path = f"{args.dataset_path}/annotation_images/{val['image_path']}"
            orig_prompt = val["original_prompt"].replace("[", "").replace("]", "")
            instruction = val["editing_instruction"]
            edit_prompt = f"{val['editing_prompt'].replace('[', '').replace(']', '')}"
            
            mask = rle2mask(val["mask"]).reshape((512,512))
            mask= mask[:, :, np.newaxis].repeat([3],axis=2)


            input_image = Image.open(img_path).convert("RGB").resize((512,512))
            image = np.array(input_image)

            image = cv2.Canny(image, 100, 200)

            res = pipe_inversion(prompt = [''],
                            negative_prompt = [''],
                            num_inversion_steps = args.num_inference_steps,
                            num_inference_steps = args.num_inference_steps,
                            image = input_image,
                            control_image= Image.fromarray(image),
                            guidance_scale = 1.01,
                            controlnet_conditioning_scale = args.controlnet_conditioning_scale,
                        )
                
            latents = res[0][0]
            all_latents = res[1]


            if val["blended_word"]!="":
                local = val["blended_word"].split(" ")[1]
            else:
                local = ""
           
            samples= inference(
                                img= input_image,
                                control_image = Image.fromarray(image),
                                source_prompt= orig_prompt,
                                target_prompt= edit_prompt,
                                guidance_dpg= args.dpg_weight,
                                guidance_cfg= args.cfg_weight,
                                local = local,
                                num_inference_steps= args.num_inference_steps,
                                controlnet_conditioning_scale = args.controlnet_conditioning_scale,
                                width= 512,
                                height= 512,
                                positive_prompt= '',
                                negative_prompt= '',
                                seed = 47,
                                thresh_e= args.mask_threshold,
                                latents = latents,
                                all_latents = all_latents
                            )
           
            
            samples_viz = samples

            pipe_inversion.scheduler.prev_v = None

            for keys in metrics.keys():
                value = calculate_metric(evaluator, keys, input_image, samples_viz, mask, mask, orig_prompt, edit_prompt)
                if value != 'nan':
                    metrics[keys][0] += value
                    metrics[keys][1] += 1

    for keys, val in metrics.items():
        print(f"{keys}:{val[0]/val[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./pie")
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--mask_threshold", type=float, default=1.0)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=0.0)
    parser.add_argument("--dpg_weight", type=float, default=1.0)
    parser.add_argument("--cfg_weight", type=float, default=1.1)
    args = parser.parse_args()
    main(args)


    