import os
import json
import argparse
import numpy as np
import cv2
from tqdm.auto import tqdm
from accelerate import PartialState
import torch
from diffusers import StableDiffusionInpaintPipeline, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, BlipProcessor, BlipForConditionalGeneration

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, default="outpaint_result")
    parser.add_argument("--imgs-root", type=str, required=True)
    parser.add_argument("--expand-mask-root", type=str, required=True)
    parser.add_argument("--json-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--model-id", type=str, default="stabilityai/stable-diffusion-2-inpainting")
    parser.add_argument("--sampler", type=str, default="euler")
    parser.add_argument("--sample-step", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--denoise-strength", type=float, default=1.0)
    parser.add_argument("--disable-progress-bar", action="store_true")
    parser.add_argument("--negative-prompt", type=str, default="")
    args = parser.parse_args()
    return args

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

def resize(img, new_shape, is_shrink=False):
    if is_shrink:
        return cv2.resize(img, (new_shape[1], new_shape[0]), cv2.INTER_AREA)
    else:
        return cv2.resize(img, (new_shape[1], new_shape[0]), cv2.INTER_LINEAR)

def restore_from_mask(
    pipe,
    init_images,
    mask_images,
    prompts=[],
    negative_prompts=[],
    num_inference_steps=30,
    guidance_scale=7.5,
    seed=None,
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
        seed (int, optional): Random seed for reproducibility
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
    
    with torch.inference_mode():
        outputs = pipe(
            prompt=prompts,
            negative_prompt=negative_prompts,
            image=init_images,
            mask_image=mask_images,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            strength=denoise_strength,
            output_type="np"
        ).images
    torch.cuda.empty_cache()
    
    images = []
    for image in outputs:
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        images.append(image)
        
    return images

def generate_image_caption(model, processor, images, device):
    inputs = processor(images, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
    )
    captions = processor.batch_decode(outputs, skip_special_tokens=True)
    return captions

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
        torch_dtype=torch.float16,
        safety_checker=None
    )
    return pipe

if __name__ == "__main__":
    args = get_args()
    target_size = (512, 512)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)


    with open(args.json_path, "r") as anno_file:
        data = json.load(anno_file)

    images_info = dict(
                [[img_info["id"], img_info] for img_info in data["images"]]
            )

    categories = [cate["id"] for cate in data["categories"]]

    annos_info = data["annotations"]

    bprocessor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    bmodel = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    pipeline = get_sd_pipeline(args.model_id, args.seed)

    distributed_state = PartialState()
    if args.disable_progress_bar:
        pipeline.set_progress_bar_config(disable=True)

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

        is_shrink = ((img_width > target_size[1]) or (img_height > target_size[0]))
        percent = str(int(anno["percent"] * 100))
        expand_mask_name = f"{img_name}_{anno['id']}_e.png"
        expand_mask_path = f"{args.expand_mask_root}/{percent}/{expand_mask_name}"
        expand_mask = cv2.imread(expand_mask_path, cv2.IMREAD_GRAYSCALE)
        expand_mask = expand_mask * 255.0
        expand_mask = resize(expand_mask, target_size, is_shrink)
        _, expand_mask = cv2.threshold(expand_mask, 128, 255, 0)
        
        img_path = f"{args.imgs_root}/{img_info['file_name']}"
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_filled = fill_img(img, init_mask, anno["last_col"])
        img_filled = resize(img_filled, target_size, is_shrink)
        
        if len(batch_info["images"]) == args.batch_size:
            with distributed_state.split_between_processes(batch_info) as batch_info:
                images = np.array(batch_info["images"])
                captions = generate_image_caption(bmodel,
                                                bprocessor,
                                                (torch.as_tensor(images)
                                                    .permute(0, 3, 1, 2)),
                                                distributed_state.device)


                negative_promts = [args.negative_prompt for i in range(len(captions))]
                images = images / 255
                
                result_images = restore_from_mask(pipe=pipeline,
                                            init_images=images,
                                            mask_images=batch_info["masks"],
                                            prompts=captions,
                                            negative_prompts=negative_promts,
                                            sampler=args.sampler,
                                            num_inference_steps=args.sample_step,
                                            denoise_strength=args.denoise_strength)
                
                for i, image in enumerate(result_images):
                    result_save_path = f"{args.save_path}/{batch_info['expand_masks_name'][i]}_re.png"
                    img_h, img_w = batch_info['imgs_shape'][i]
                    is_shrink = (((img_w < target_size[1]) or (img_h < target_size[0])))
                    image = resize(image, batch_info["imgs_shape"][i], is_shrink)
                    cv2.imwrite(result_save_path, image)
                    
            batch_info = {"images": [],
                        "masks": [],
                        "expand_masks_name": [],
                        "imgs_shape": []}
        else:
            batch_info["images"].append(img_filled)
            batch_info["masks"].append(expand_mask)
            batch_info["expand_masks_name"].append(expand_mask_name.split(".")[0])
            batch_info["imgs_shape"].append([img_height, img_width])