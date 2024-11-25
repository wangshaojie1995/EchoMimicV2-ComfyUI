import sys,os
import folder_paths
import os.path as osp
now_dir = osp.dirname(__file__)
aifsh_dir = osp.join(folder_paths.models_dir,"AIFSH")
sys.path.append(now_dir)
from huggingface_hub import snapshot_download
echomimicv2_models_dir = osp.join(aifsh_dir,"EchoMimicV2")
vae_dir = osp.join(echomimicv2_models_dir,"sd-vae-ft-mse")
base_model_dir = osp.join(echomimicv2_models_dir,"sd-image-variations-diffusers")
image_encoder_dir = osp.join(base_model_dir,"image_encoder")

import cv2
import decord
import torch
import torchaudio
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
if sys.platform != "win32":
    from torchao.quantization import quantize_,int8_weight_only
else:
    from .fp8_optimization import convert_fp8_linear
from diffusers import AutoencoderKL,DDIMScheduler
from multiprocessing.pool import ThreadPool
from echomimicv2.dwpose import DWposeDetector
from echomimicv2.src.models.unet_2d_condition import UNet2DConditionModel
from echomimicv2.src.models.unet_3d_emo import EMOUNet3DConditionModel
from echomimicv2.src.models.whisper.audio2feature import load_audio_model
from echomimicv2.src.pipelines.pipeline_echomimicv2 import EchoMimicV2Pipeline
from echomimicv2.src.utils.util import save_videos_grid
from echomimicv2.src.models.pose_encoder import PoseEncoder
from echomimicv2.src.utils.img_utils import save_video_from_cv2_list
from echomimicv2.src.utils.dwpose_util import draw_pose_select_v2
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    from moviepy import VideoFileClip, AudioFileClip
except:
    from moviepy.editor import VideoFileClip, AudioFileClip

out_dir = folder_paths.get_output_directory()
save_dir = osp.join(out_dir,"EchoMimicV2")

def convert_fps(src_path, tgt_path, tgt_fps=24, tgt_sr=16000):
    clip = VideoFileClip(src_path)
    try:
        new_clip = clip.set_fps(tgt_fps)
    except:
        new_clip = clip.with_fps(tgt_fps)
    if tgt_fps is not None:
        audio = new_clip.audio
        try:
            audio = audio.set_fps(tgt_sr)
            new_clip = new_clip.set_audio(audio)
        except:
            audio = audio.with_fps(tgt_sr)
            new_clip = new_clip.with_audio(audio)
    
    new_clip.write_videofile(tgt_path, codec='libx264', audio_codec='aac')
  
def get_video_pose(
        video_path: str, 
        sample_stride: int=1,
        max_frame=None):

    # read input video
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    sample_stride *= max(1, int(vr.get_avg_fps() / 24))

    frames = vr.get_batch(list(range(0, len(vr), sample_stride))).asnumpy()
    if max_frame is not None:
        frames = frames[0:max_frame,:,:]
    height, width, _ = frames[0].shape
    # detected_poses = [dwprocessor(frm) for frm in tqdm(frames, desc="DWPose")]
    dwp_dir = osp.join(echomimicv2_models_dir,"DWPose")
    dwprocessor = DWposeDetector(det_ckpt=osp.join(dwp_dir,"yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth"),
                                 pose_ckpt=osp.join(dwp_dir,"dw-ll_ucoco_384.pth"),device=device)
    detected_poses = [dwprocessor(frm) for frm in tqdm(frames, desc="DWPose")]
    dwprocessor = None
    torch.cuda.empty_cache()
    return detected_poses, height, width, frames


def resize_and_pad_param(imh, imw, max_size):
    half = max_size // 2
    if imh > imw:
        imh_new = max_size
        imw_new = int(round(imw/imh * imh_new))
        half_w = imw_new // 2
        rb, re = 0, max_size
        cb = half-half_w
        ce = cb + imw_new
    else:
        imw_new = max_size
        imh_new = int(round(imh/imw * imw_new))
        imh_new = max_size

        half_h = imh_new // 2
        cb, ce = 0, max_size
        rb = half-half_h
        re = rb + imh_new
        
    return imh_new, imw_new, rb, re, cb, ce

def get_pose_params(detected_poses, max_size,height, width):
    print('get_pose_params...')
    # pose rescale 
    w_min_all, w_max_all, h_min_all, h_max_all = [], [], [], []
    mid_all = []
    for num, detected_pose in enumerate(detected_poses):
        detected_poses[num]['num'] = num
        candidate_body = detected_pose['bodies']['candidate']
        score_body = detected_pose['bodies']['score']
        candidate_face = detected_pose['faces']
        score_face = detected_pose['faces_score']
        candidate_hand = detected_pose['hands']
        score_hand = detected_pose['hands_score']

        # 选取置信度最高的face
        if candidate_face.shape[0] > 1:
            index = 0
            candidate_face = candidate_face[index]
            score_face = score_face[index]
            detected_poses[num]['faces'] = candidate_face.reshape(1, candidate_face.shape[0], candidate_face.shape[1])
            detected_poses[num]['faces_score'] = score_face.reshape(1, score_face.shape[0])
        else:
            candidate_face = candidate_face[0]
            score_face = score_face[0]

        # 选取置信度最高的body
        if score_body.shape[0] > 1:
            tmp_score = []
            for k in range(0, score_body.shape[0]):
                tmp_score.append(score_body[k].mean())
            index = np.argmax(tmp_score)
            candidate_body = candidate_body[index*18:(index+1)*18,:]
            score_body = score_body[index]
            score_hand = score_hand[(index*2):(index*2+2),:]
            candidate_hand = candidate_hand[(index*2):(index*2+2),:,:]
        else:
            score_body = score_body[0]
        all_pose = np.concatenate((candidate_body, candidate_face))
        all_score = np.concatenate((score_body, score_face))
        all_pose = all_pose[all_score>0.8]


        body_pose = np.concatenate((candidate_body,))
        mid_ = body_pose[1, 0]


        face_pose = candidate_face
        hand_pose = candidate_hand


        h_min, h_max = np.min(face_pose[:,1]), np.max(body_pose[:7,1])

        h_ = h_max - h_min
        
        mid_w = mid_
        w_min = mid_w - h_ // 2
        w_max = mid_w + h_ // 2
        
        w_min_all.append(w_min)
        w_max_all.append(w_max)
        h_min_all.append(h_min)
        h_max_all.append(h_max)
        mid_all.append(mid_w)

    w_min = np.min(w_min_all)
    w_max = np.max(w_max_all)
    h_min = np.min(h_min_all)
    h_max = np.max(h_max_all)
    mid = np.mean(mid_all)
    print(mid)

    margin_ratio = 0.25
    h_margin = (h_max-h_min)*margin_ratio
    
    h_min = max(h_min-h_margin*0.65, 0)
    h_max = min(h_max+h_margin*0.5, 1)

    h_new = h_max - h_min
    
    h_min_real = int(h_min*height)
    h_max_real = int(h_max*height)
    mid_real = int(mid*width)
    
    
    height_new = h_max_real-h_min_real+1
    width_new = height_new
    w_min_real = mid_real - height_new // 2

    w_max_real = w_min_real + width_new
    w_min = w_min_real / width
    w_max = w_max_real / width

    print(width_new, height_new)

    imh_new, imw_new, rb, re, cb, ce = resize_and_pad_param(height_new, width_new, max_size)
    res = {'draw_pose_params': [imh_new, imw_new, rb, re, cb, ce], 
           'pose_params': [w_min, w_max, h_min, h_max],
           'video_params': [h_min_real, h_max_real, w_min_real, w_max_real],
           }
    return res

def save_pose_params_item(input_items):
    detected_pose, pose_params, draw_pose_params, save_dir = input_items
    w_min, w_max, h_min, h_max = pose_params
    num = detected_pose['num']
    candidate_body = detected_pose['bodies']['candidate']
    candidate_face = detected_pose['faces'][0]
    candidate_hand = detected_pose['hands']
    candidate_body[:,0] = (candidate_body[:,0]-w_min)/(w_max-w_min)
    candidate_body[:,1] = (candidate_body[:,1]-h_min)/(h_max-h_min)
    candidate_face[:,0] = (candidate_face[:,0]-w_min)/(w_max-w_min)
    candidate_face[:,1] = (candidate_face[:,1]-h_min)/(h_max-h_min)
    candidate_hand[:,:,0] = (candidate_hand[:,:,0]-w_min)/(w_max-w_min)
    candidate_hand[:,:,1] = (candidate_hand[:,:,1]-h_min)/(h_max-h_min)
    detected_pose['bodies']['candidate'] = candidate_body
    detected_pose['faces'] = candidate_face.reshape(1, candidate_face.shape[0], candidate_face.shape[1])
    detected_pose['hands'] = candidate_hand
    detected_pose['draw_pose_params'] = draw_pose_params
    np.save(save_dir+'/'+str(num)+'.npy', detected_pose)

def save_pose_params(detected_poses, pose_params, draw_pose_params, pose_dir):
    save_dir = pose_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    input_list = []
    for i, detected_pose in enumerate(detected_poses):
        input_list.append([detected_pose, pose_params, draw_pose_params, save_dir])

    pool = ThreadPool(8)
    pool.map(save_pose_params_item, input_list)
    pool.close()
    pool.join()

def resize_and_pad(img, max_size):
    img_new = np.zeros((max_size, max_size, 3)).astype('uint8')
    imh, imw = img.shape[0], img.shape[1]
    half = max_size // 2
    if imh > imw:
        imh_new = max_size
        imw_new = int(round(imw/imh * imh_new))
        half_w = imw_new // 2
        rb, re = 0, max_size
        cb = half-half_w
        ce = cb + imw_new
    else:
        imw_new = max_size
        imh_new = int(round(imh/imw * imw_new))
        half_h = imh_new // 2
        cb, ce = 0, max_size
        rb = half-half_h
        re = rb + imh_new

    img_resize = cv2.resize(img, (imw_new, imh_new))
    img_new[rb:re,cb:ce,:] = img_resize
    return img_new


def save_processed_video(ori_frames, video_params, save_path, max_size):
    h_min_real, h_max_real, w_min_real, w_max_real = video_params
    video_frame_crop = []
    for img in ori_frames:
        img = img[h_min_real:h_max_real,w_min_real:w_max_real,:]
        img = resize_and_pad(img, max_size=max_size)
        video_frame_crop.append(img)
    save_video_from_cv2_list(video_frame_crop, save_path, fps=24.0, rgb2bgr=True)
    return video_frame_crop

def draw_pose_video(pose_params_path, save_path, max_size, ori_frames=None):
    pose_files = os.listdir(pose_params_path)
     # 生成Pose图cd pro 
    output_pose_img = []
    for i in range(0, len(pose_files)):
        pose_params_path_tmp = pose_params_path + '/' + str(i) + '.npy'
        detected_pose = np.load(pose_params_path_tmp, allow_pickle=True).tolist()
        imh_new, imw_new, rb, re, cb, ce = detected_pose['draw_pose_params']
        im = draw_pose_select_v2(detected_pose, imh_new, imw_new, ref_w=800)
        im = np.transpose(np.array(im),(1,2,0))
        img_new = np.zeros((max_size, max_size, 3)).astype('uint8')
        img_new[rb:re,cb:ce,:] = im
        if ori_frames is not None:
            img_new = img_new * 0.6 + ori_frames[i] * 0.4
            img_new = img_new.astype('uint8')
        output_pose_img.append(img_new)

    output_pose_img = np.stack(output_pose_img)
    save_video_from_cv2_list(output_pose_img, save_path, fps=24.0, rgb2bgr=True)
    print('save to ' + save_path)


class EchoMimicV2PoseNode:
    @classmethod
    def INPUT_TYPES(s):
        try:
            pose_dir = ["default"] +os.listdir(osp.join(save_dir,"DWPose/pose"))
        except:
            pose_dir = ["default"]
        return {
            "required":{
                "pose":("STRING",{
                    "default":"default",
                    "tooltip":f"name your pose or use the pose propossed in {pose_dir}"
                }),
            },
            "optional":{
                "driving_pose":("VIDEO",),
            }
        }
    RETURN_TYPES = ("POSE","VIDEO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "gen_pose"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_EchoMimicV2"

    def gen_pose(self,pose,driving_pose=None):
        ## 
        if pose == "default":
           pose_dir =  osp.join(now_dir,"echomimicv2/pose/01")
           return (pose_dir, )
        os.makedirs(save_dir,exist_ok=True)
        pose_dir = osp.join(save_dir,"DWPose/pose",pose)
        save_path = osp.join(save_dir,"DWPose","pose_"+Path(driving_pose).name)
        if osp.exists(pose_dir) and len(os.listdir(pose_dir)) > 24:
            return (pose_dir,save_path,)
        with tempfile.NamedTemporaryFile(suffix=".mp4",delete=False,dir=save_dir) as f:
            convert_fps(driving_pose, f.name)
        
        # 提取Pose
        detected_poses, height, width, ori_frames = get_video_pose(f.name, max_frame=None)
        print(height, width)

        # 提取相关参数
        res_params = get_pose_params(detected_poses, 768,height, width)
        
        # 存储Pose参数
        save_pose_params(detected_poses, res_params['pose_params'], res_params['draw_pose_params'], pose_dir)
        
        # 存储截取视频
        processed_video_path = osp.join(save_dir,"DWPose",Path(driving_pose).name)
        video_frame_crop = save_processed_video(ori_frames, res_params['video_params'],processed_video_path, 768)
        
        draw_pose_video(pose_dir, save_path, 768,ori_frames=video_frame_crop)
        return (pose_dir,save_path,)


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
            
        if not osp.exists(osp.join(base_model_dir,"unet/diffusion_pytorch_model.bin")):
            snapshot_download(repo_id="lambdalabs/sd-image-variations-diffusers",
                              local_dir=base_model_dir,)
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "refimg":("IMAGE",),
                "driving_audio":("AUDIO",),
                "pose":("POSE",),
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
                "if_low_varm":("BOOLEAN",{
                    "default":False
                }),
                "store_in_varm":("BOOLEAN",{
                    "default":True
                }),
                "seed":("INT",{
                    "default":42
                })
            },
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
        new_image.paste(image,box=((w-nw)//2,(h-nh)//2))
        return new_image

    def gen_video(self,refimg,driving_audio,pose,steps,cfg,context_frames,
                  context_overlap,fps,if_low_varm,store_in_varm,seed):
        weight_dtype = torch.float16
        infer_config = OmegaConf.load(osp.join(now_dir,"echomimicv2/configs/inference/inference_v2.yaml"))
        os.makedirs(save_dir,exist_ok=True)
        if self.pipe is None:
            ############# model_init started #############
            ## vae init
            vae = AutoencoderKL.from_pretrained(
                vae_dir,
            ).to(device, dtype=weight_dtype)
            ## reference net init
            reference_unet = UNet2DConditionModel.from_pretrained(
                base_model_dir,
                subfolder="unet",
            ).to(dtype=weight_dtype, device=device)
            reference_unet.load_state_dict(
                torch.load(osp.join(echomimicv2_models_dir,"reference_unet.pth"), map_location="cpu"),
            )
            if if_low_varm:
                try:
                    quantize_(vae, int8_weight_only())
                    quantize_(reference_unet,int8_weight_only())
                except:
                    convert_fp8_linear(vae,torch.bfloat16)
                    convert_fp8_linear(reference_unet,torch.bfloat16)
            ## denoising net init
            denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
                base_model_dir,
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
            # self.pose_encoder = pose_net
            #if if_low_varm:
                # self.pipe.enable_vae_slicing()
                # self.pipe.enable_sequential_cpu_offload()
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
            "pose":pose
        }
        print('Pose:', inputs_dict['pose'])
        print('Reference:', inputs_dict['refimg'])
        print('Audio:', inputs_dict['audio'])
        audio_clip = AudioFileClip(inputs_dict['audio'])

        num_pose_files = len(os.listdir(inputs_dict['pose'])) # Total number of pose files (indices 0 to 335)
        L = int(audio_clip.duration * fps)
        print(f"the max frame num:{L}")
        pose_list = []
        for index in range(L):
            file_index = index % num_pose_files
            tgt_musk = np.zeros((W, H, 3)).astype('uint8')
            tgt_musk_path = os.path.join(inputs_dict['pose'], "{}.npy".format(file_index))
            detected_pose = np.load(tgt_musk_path, allow_pickle=True).tolist()
            imh_new, imw_new, rb, re, cb, ce = detected_pose['draw_pose_params']
            im = draw_pose_select_v2(detected_pose, imh_new, imw_new, ref_w=800)
            im = np.transpose(np.array(im),(1, 2, 0))
            tgt_musk[rb:re,cb:ce,:] = im

            tgt_musk_pil = Image.fromarray(np.array(tgt_musk)).convert('RGB')
            pose_list.append(torch.Tensor(np.array(tgt_musk_pil)).to(dtype=weight_dtype, device=device).permute(2,0,1) / 255.0)
        
        poses_tensor = torch.stack(pose_list, dim=1).unsqueeze(0)
        try:
            audio_clip = audio_clip.with_subclip(0,L / fps)
        except:
            audio_clip = audio_clip.set_duration(L / fps)
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
            context_overlap=context_overlap
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
        try:
            video_clip_sig = video_clip_sig.set_audio(audio_clip)
        except:
            video_clip_sig = video_clip_sig.with_audio(audio_clip)
        outfile = osp.join(out_dir,Path(img.name).stem+"_"+Path(audio.name).stem+".mp4")
        video_clip_sig.write_videofile(outfile,fps=fps,codec="libx264", audio_codec="aac", threads=2)
        if not store_in_varm:
            self.pipe = None
            torch.cuda.empty_cache()
        return (outfile,)

NODE_CLASS_MAPPINGS = {
    "EchoMimicV2PoseNode":EchoMimicV2PoseNode,
    "EchoMimicV2Node": EchoMimicV2Node
}
