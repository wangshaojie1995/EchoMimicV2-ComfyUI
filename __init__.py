import sys,os
import folder_paths
import os.path as osp
now_dir = osp.dirname(__file__)
aifsh_dir = osp.join(folder_paths.models_dir,"AIFSH")
sys.path.append(now_dir)
from huggingface_hub import snapshot_download
echomimicv2_models_dir = osp.join(aifsh_dir,"EchoMimicV2")
vae_dir = osp.join(echomimicv2_models_dir,"sd-vae-ft-mse")
image_encoder_dir = osp.join(echomimicv2_models_dir,"sd-image-variations-diffusers")

import torch
import torchaudio
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf
from diffusers import AutoencoderKL,DDIMScheduler
from echomimicv2.src.models.unet_2d_condition import UNet2DConditionModel
from echomimicv2.src.models.unet_3d_emo import EMOUNet3DConditionModel
from echomimicv2.src.models.whisper.audio2feature import load_audio_model
from echomimicv2.src.pipelines.pipeline_echomimicv2 import EchoMimicV2Pipeline
from echomimicv2.src.utils.util import save_videos_grid
from echomimicv2.src.models.pose_encoder import PoseEncoder
from echomimicv2.src.utils.dwpose_util import draw_pose_select_v2
from moviepy import VideoFileClip, AudioFileClip

class EchoMimicV2Node:
    def __init__(self) -> None:
        self.pipe = None
        if not osp.exists(osp.join(echomimicv2_models_dir,"denoising_unet.pth")):
            snapshot_download(repo_id="BadToBest/EchoMimicV2",
                              local_dir=echomimicv2_models_dir)
        if not osp.exists(osp.join(vae_dir,"diffusion_pytorch_model.safetensors")):
            snapshot_download(repo_id="stabilityai/sd-vae-ft-mse",
                              local_dir=vae_dir,
                              allow_patterns=["*json","*safetensors"])
            
        if not osp.exists(osp.join(image_encoder_dir,"unet/diffusion_pytorch_model.bin")):
            snapshot_download(repo_id="lambdalabs/sd-image-variations-diffusers",
                              local_dir=image_encoder_dir,)
        snapshot_download(repo_id="lambdalabs/sd-image-variations-diffusers",
                              local_dir=image_encoder_dir,)
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "refimg":("IMAGE",),
                "driving_audio":("AUDIO",),
                "steps":("INT",{
                    "default":30
                }),
                "cfg":("FLOAT",{
                    "default":2.5
                }),
                "context_frames":("INT",{
                    "default":12
                }),
                "context_overlap":("INT",{
                    "default":3
                }),
                "fps":("INT",{
                    "default":24
                }),
                "store_in_varm":("BOOLEAN",{
                    "default":True
                }),
                "seed":("INT",{
                    "default":42
                })
            }
        }
    
    RETURN_TYPES = ("VIDEO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "gen_video"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_EchoMimicV2"

    def pad(self,source_img,target_size=(768,768)):
        iw,ih = source_img.size
        if iw == ih:
           return source_img.resize(target_size)
        w,h = target_size
        scale = min(w/iw,h/ih)
        nw,nh = int(iw*scale),int(ih*scale)
        image = source_img.resize((nw,nh))
        new_image = Image.new("RGB",target_size,color=(124,252,0))
        new_image.paste(image,box=((w-nw)//2,(h-nh)//2),mask=image)
        return new_image

    def gen_video(self,refimg,driving_audio,steps,cfg,
                  context_frames,context_overlap,fps,store_in_varm,seed):
        weight_dtype = torch.float16
        device = "cuda" if torch.cuda.is_available() else "cpu"
        infer_config = OmegaConf.load(osp.join(now_dir,"echomimicv2/configs/inference/inference_v2.yaml"))
        out_dir = folder_paths.get_output_directory()
        save_dir = osp.join(out_dir,"EchoMimicV2")
        os.makedirs(save_dir,exist_ok=True)
        if self.pipe is None:
            ############# model_init started #############
            ## vae init
            vae = AutoencoderKL.from_pretrained(
                vae_dir,
            ).to(device, dtype=weight_dtype)

            ## reference net init
            reference_unet = UNet2DConditionModel.from_pretrained(
                image_encoder_dir,
                subfolder="unet",
            ).to(dtype=weight_dtype, device=device)
            reference_unet.load_state_dict(
                torch.load(osp.join(echomimicv2_models_dir,"reference_unet.pth"), map_location="cpu"),
            )
            ## denoising net init
            denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
                image_encoder_dir,
                osp.join(echomimicv2_models_dir,"motion_module.pth"),
                subfolder="unet",
                unet_additional_kwargs=infer_config.unet_additional_kwargs,
            ).to(dtype=weight_dtype, device=device)
        
            denoising_unet.load_state_dict(
                torch.load(osp.join(echomimicv2_models_dir,"denoising_unet.pth"), map_location="cpu"),
                strict=False
            )

            # pose net init
            pose_net = PoseEncoder(320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)).to(
                dtype=weight_dtype, device=device
            )
            pose_net.load_state_dict(torch.load(osp.join(echomimicv2_models_dir,"pose_encoder.pth")))

            ### load audio processor params
            audio_processor = load_audio_model(model_path=osp.join(echomimicv2_models_dir,"whisper_tiny.pt"), device=device)
        
            ############# model_init finished #############
            sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
            scheduler = DDIMScheduler(**sched_kwargs)

            self.pipe = EchoMimicV2Pipeline(
                vae=vae,
                reference_unet=reference_unet,
                denoising_unet=denoising_unet,
                audio_guider=audio_processor,
                pose_encoder=pose_net,
                scheduler=scheduler,
            )
            self.pose_encoder = pose_net

            self.pipe = self.pipe.to(device, dtype=weight_dtype)
            
        generator = torch.manual_seed(seed)
        W,H = 768,768
        with tempfile.NamedTemporaryFile(suffix=".png",delete=False,dir=save_dir) as img:
            img_np = refimg.numpy()[0] * 255
            img_np = img_np.astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            img_pil = self.pad(img_pil,(W,H))
            img_pil.save(img.name)
        
        with tempfile.NamedTemporaryFile(suffix='.wav',delete=False,dir=save_dir) as audio:
            torchaudio.save(audio.name,driving_audio["waveform"][0],driving_audio["sample_rate"])
        
        inputs_dict = {
            "refimg":img.name,
            "audio":audio.name,
            "pose":osp.join(now_dir,"echomimicv2/pose/01")
        }
        start_idx = 0
        print('Pose:', inputs_dict['pose'])
        print('Reference:', inputs_dict['refimg'])
        print('Audio:', inputs_dict['audio'])
        audio_clip = AudioFileClip(inputs_dict['audio'])
        L = min(int(audio_clip.duration * fps), len(os.listdir(inputs_dict['pose'])))
        pose_list = []
        for index in range(start_idx, start_idx + L):
            tgt_musk = np.zeros((W, H, 3)).astype('uint8')
            tgt_musk_path = os.path.join(inputs_dict['pose'], "{}.npy".format(index))
            detected_pose = np.load(tgt_musk_path, allow_pickle=True).tolist()
            imh_new, imw_new, rb, re, cb, ce = detected_pose['draw_pose_params']
            im = draw_pose_select_v2(detected_pose, imh_new, imw_new, ref_w=800)
            im = np.transpose(np.array(im),(1, 2, 0))
            tgt_musk[rb:re,cb:ce,:] = im

            tgt_musk_pil = Image.fromarray(np.array(tgt_musk)).convert('RGB')
            pose_list.append(torch.Tensor(np.array(tgt_musk_pil)).to(dtype=weight_dtype, device=device).permute(2,0,1) / 255.0)
        
        poses_tensor = torch.stack(pose_list, dim=1).unsqueeze(0)
        audio_clip = audio_clip.with_subclip(0,L / fps)
        video = self.pipe(
            img_pil,
            inputs_dict['audio'],
            poses_tensor[:,:,:L,...],
            W,
            H,
            L,
            steps,
            cfg,
            generator=generator,
            context_frames=context_frames,
            fps=fps,
            context_overlap=context_overlap,
            start_idx=start_idx,
        ).videos
        video_sig = video[:, :, :L, :, :]
        tmp_file = osp.join(save_dir,Path(img.name).stem+".mp4")
        save_videos_grid(
            video_sig,
            tmp_file,
            n_rows=1,
            fps=fps,
        )
        video_clip_sig = VideoFileClip(tmp_file)
        video_clip_sig = video_clip_sig.set_audio(audio_clip)
        outfile = osp.join(save_dir,Path(img.name).stem+"_"+Path(audio.name).stem+".mp4")
        video_clip_sig.write_videofile(outfile, codec="libx264", audio_codec="aac", threads=2)
        if not store_in_varm:
            self.pipe = None
            torch.cuda.empty_cache()
        return (outfile,)

NODE_CLASS_MAPPINGS = {
    "EchoMimicV2Node": EchoMimicV2Node
}
