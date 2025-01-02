import os
import json
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from accelerate import PartialState
import torch
from diffusers import StableDiffusionInpaintPipeline, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, BlipProcessor, BlipForConditionalGeneration

def get_mask(height, width, polygons):
        mask = np.zeros([height, width])
        for polygon in polygons:
            polygon = np.array([polygon]).reshape(1, -1, 2)
            mask = cv2.fillPoly(
                mask, np.array(polygon), color=[255, 255, 255]
            )
            
        mask = mask.astype(np.uint8)
        return mask

def get_average_color(image, mask):
    selected_pixels = image[mask > 0]
    average_color = selected_pixels.mean(axis=0).astype(int)  # (R, G, B)
    return average_color
    
def fill_img(img, mask, last_col):
    img = np.array(img)
    average_color = get_average_color(img, mask)
    if last_col > 0:
        img[:, :last_col] = average_color
    else:
        img[:, last_col+1:] = average_color

    return img

def restore_from_mask(
    pipe,
    init_image,
    mask_image,
    prompt="",
    negative_prompt="",
    num_inference_steps=30,
    guidance_scale=7.5,
    denoise_strength=0.75,
    sampler="euler_a"
):
    """
    Restore an image using stable diffusion inpainting with customizable parameters.
    
    Args:
        cropped_image (PIL.Image): The input image to be restored
        mask_image (PIL.Image): The mask indicating areas to be inpainted
        prompt (str): Text prompt for guided generation
        negative_prompt (str): Text prompt for what to avoid in generation
        num_inference_steps (int): Number of denoising steps
        guidance_scale (float): How strongly the image should conform to prompt (CFG scale)
        denoise_strength (float): Strength of denoising, between 0.0 and 1.0
        sampler (str): Sampling method to use ('euler_a', 'euler', 'heun', 'dpm_2', 
                       'dpm_2_ancestral', 'lms', 'ddim', 'pndm')
    
    Returns:
        PIL.Image: The restored image
    """
    
    # Set device and optimize memory
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # pipe = pipe.to(device)
    torch.cuda.empty_cache()
    pipe.enable_attention_slicing()
    # pipe.enable_xformers_memory_efficient_attention()
    
    if sampler == "euler_a":
        from diffusers import EulerAncestralDiscreteScheduler
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler == "euler":
        from diffusers import EulerDiscreteScheduler
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler == "heun":
        from diffusers import HeunDiscreteScheduler
        pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler == "dpm_2":
        from diffusers import KDPM2DiscreteScheduler
        pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler == "dpm_2_ancestral":
        from diffusers import KDPM2AncestralDiscreteScheduler
        pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler == "lms":
        from diffusers import LMSDiscreteScheduler
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    elif sampler == "ddim":
        from diffusers import DDIMScheduler
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif sampler == "pndm":
        from diffusers import PNDMScheduler
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    elif sampler == "DPM":
        from diffusers import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,
                                                                    use_karras_sigmas=True,
                                                                algorithm_type="sde-dpmsolver++"
                                                                )
    
    target_size = (512, 512)
    images = []
    masks = []
    for img in init_image:
        images.append(img.resize(target_size))

    for mask in mask_image:
        masks.append(mask.resize(target_size))
    
    with torch.inference_mode():
        outputs = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=images,
            mask_image=masks,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            strength=denoise_strength,  # Using the denoise_strength parameter
        ).images
        
    torch.cuda.empty_cache()
    return outputs

def generate_image_caption(model, processor, images, device):
    inputs = processor(images, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
    )
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

def get_sd_pipeline(model_id, seed):
    # seed (int, optional): Random seed for reproducibility

    torch.cuda.empty_cache()
    if seed is not None:
        torch.manual_seed(seed)
        
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        safety_checker=None
    )
    return pipe

if __name__ == "__main__":
  save_path = "outpaint_result"
  imgs_path = "/kaggle/input/final-dataset/test"
  expand_mask_root = "/kaggle/input/expand-masks/kaggle/working/image_mask"
  json_path = "/kaggle/input/final-dataset/annotations/test.json"
  seed = 69
  #model_id = "Uminosachi/realisticVisionV51_v51VAE-inpainting"
  model_id = "stabilityai/stable-diffusion-2-inpainting"
  sampler = "euler"
  sample_step = 20
  batch_size = 4        #need to be even
  denoise_strength = 1

  if not os.path.exists(save_path):
      os.makedirs(save_path)


  with open(json_path, "r") as anno_file:
      data = json.load(anno_file)

  images_info = dict(
              [[img_info["id"], img_info] for img_info in data["images"]]
          )

  categories = [cate["id"] for cate in data["categories"]]

  annos_info = data["annotations"]

  bprocessor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
  bmodel = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
  pipeline = get_sd_pipeline(model_id, seed)

  distributed_state = PartialState()
  pipeline.to(distributed_state.device)
  bmodel.to(distributed_state.device)
  batch_info = {"images": [],
              "masks": [],
              "expand_masks_name": [],
              "imgs_shape": []}

  for anno in tqdm(annos_info):
      img_info = images_info[anno["image_id"]]
      img_name = img_info["file_name"].split(".")[0]
      img_width, img_height = img_info["width"], img_info["height"]
      
      init_mask = get_mask(
              img_height, img_width, anno["visible_segmentations"]
      )

      percent = str(int(anno["percent"] * 100))
      expand_mask_name = f"{img_name}_{anno['id']}_e.png"
      expand_mask_path = f"{expand_mask_root}/{percent}/{expand_mask_name}"
      expand_mask = Image.open(expand_mask_path)
      expand_mask = Image.fromarray(np.array(expand_mask) * 255.0).convert("RGB") 
      
      img_path = f"{imgs_path}/{img_info['file_name']}"
      img = Image.open(img_path)
      img_filled = fill_img(img, init_mask, anno["last_col"])
      
      if len(batch_info["images"]) == batch_size:
          with distributed_state.split_between_processes(batch_info) as batch_info:
              captions = []
              for image in batch_info["images"]:
                  caption = generate_image_caption(bmodel,
                                                  bprocessor,
                                                  (torch.as_tensor(np.array(image))
                                                      .permute(2, 0, 1)),
                                                  distributed_state.device)
                  captions.append(caption)

              negative_promts = ['' for i in range(len(captions))]
              
              images = restore_from_mask(pipe=pipeline,
                                        init_image=batch_info["images"],
                                        mask_image=batch_info["masks"],
                                        prompt=captions,
                                        negative_prompt=negative_promts,
                                        sampler=sampler,
                                        num_inference_steps=sample_step,
                                        denoise_strength=denoise_strength)
              
              for i, image in enumerate(images):
                  result_save_path = f"{save_path}/{batch_info['expand_masks_name'][i]}_re.png"
                  image.resize(batch_info["imgs_shape"][i]).save(result_save_path)
                  
          batch_info = {"images": [],
                      "masks": [],
                      "expand_masks_name": [],
                      "imgs_shape": []}
      else:
          batch_info["images"].append(Image.fromarray(img_filled.copy()))
          batch_info["masks"].append(expand_mask)
          batch_info["expand_masks_name"].append(expand_mask_name.split(".")[0])
          batch_info["imgs_shape"].append([img_height, img_width])