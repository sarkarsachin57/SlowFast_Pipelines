import faulthandler
faulthandler.enable()

import os
import cv2
import torch
import slowfast
import imutils
from imutils.video import FPS
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm
import logging
import psutil
logging.basicConfig(format='%(asctime)s - p%(process)s {%(filename)s:%(lineno)d} - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

from ultralytics import YOLO
from slowfast.config.defaults import get_cfg, assert_and_infer_cfg
from slowfast.models import build_model
from slowfast.utils.checkpoint import load_checkpoint
from slowfast.utils.parser import load_config, parse_args
from slowfast.utils.misc import launch_job
from slowfast.visualization.utils import process_cv2_inputs
from slowfast.visualization.demo_loader import ThreadVideoManager, VideoManager
from slowfast.visualization.predictor import ActionPredictor
from slowfast.visualization.video_visualizer import VideoVisualizer
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.datasets.utils import get_sequence
from slowfast.datasets.cv2_transform import scale, scale_boxes

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)



ava_classes = ['bend/bow (at the waist)','crawl','crouch/kneel','dance','fall down',
 'get up','jump/leap','lie/sleep','martial art','run/jog','sit','stand','swim','walk',
 'answer phone','brush teeth','carry/hold (an object)','catch (an object)','chop','climb (e.g., a mountain)',
 'clink glass','close (e.g., a door, a box)','cook','cut','dig','dress/put on clothing','drink',
 'drive (e.g., a car, a truck)','eat','enter','exit','extract','fishing','hit (an object)','kick (an object)',
 'lift/pick up','listen (e.g., to music)','open (e.g., a window, a car door)','paint','play board game',
 'play musical instrument','play with pets','point to (an object)','press','pull (an object)',
 'push (an object)','put down','read','ride (e.g., a bike, a car, a horse)','row boat','sail boat','shoot',
 'shovel','smoke','stir','take a photo','text on/look at a cellphone','throw','touch (an object)',
 'turn (e.g., a screwdriver)','watch (e.g., TV)','work on a computer','write','fight/hit (a person)',
 'give/serve (an object) to (a person)','grab (a person)','hand clap','hand shake', 'hand wave',
 'hug (a person)','kick (a person)','kiss (a person)','lift (a person)','listen to (a person)',
 'play with kids','push (another person)','sing to (e.g., self, a person, a group)',
 'take (an object) from (a person)','talk to (e.g., self, a person, a group)','watch (a person)']

action_class_list = ['fall',
 'standing',
 'sitting',
 'walking',
 'crawling',
 'sleeping',
 'eating',
 'jumping',
 'others']


yolo_weights = "../yolov8m_no_augmentation.pt"
yolov8_model = YOLO(yolo_weights)
yolov8_model = yolov8_model.to('cuda' if torch.cuda.is_available() else 'cpu')

class Config:
    def __init__(self, shard_id=0, num_shards=1, init_method="tcp://localhost:9999", cfg_files=None, opts=None):
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.init_method = init_method
        self.cfg_files = cfg_files if cfg_files is not None else ["configs/Kinetics/SLOWFAST_4x16_R50.yaml"]
        self.opts = opts #if opts is not None else []

    def update(self, shard_id=None, num_shards=None, init_method=None, cfg_files=None, opts=None):
        if shard_id is not None:
            self.shard_id = shard_id
        if num_shards is not None:
            self.num_shards = num_shards
        if init_method is not None:
            self.init_method = init_method
        if cfg_files is not None:
            self.cfg_files = cfg_files
        if opts is not None:
            self.opts = opts

    def display(self):
        print("Configuration:")
        print(f"  Shard ID: {self.shard_id}")
        print(f"  Number of Shards: {self.num_shards}")
        print(f"  Initialization Method: {self.init_method}")
        print(f"  Configuration Files: {self.cfg_files}")
        print(f"  Additional Options: {self.opts}")

parse_config = Config()
parse_config.display()
args = parse_config

# cfg = load_config(args, "../slowfast_config_daycare_2.yaml")
# cfg.TEST.CHECKPOINT_FILE_PATH = "checkpoints/checkpoint_epoch_00050.pyth"
cfg = load_config(args, "configs/AVA/c2/SLOWFAST_32x2_R101_50_50_v2.1.yaml")
cfg.TEST.CHECKPOINT_FILE_PATH = "SLOWFAST_32x2_R101_50_50.pkl"
cfg.NUM_GPUS = 1
cfg.DETECTION.ENABLE = True
cfg.TEST.NUM_TEMPORAL_CLIPS = [cfg.TEST.NUM_ENSEMBLE_VIEWS]
cfg = assert_and_infer_cfg(cfg)
action_model = build_model(cfg)
action_model.eval()

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
# Set up environment.
du.init_distributed_training(cfg)
# Set random seed from configs.
np.random.seed(cfg.RNG_SEED)
torch.manual_seed(cfg.RNG_SEED)

flops, params = misc.log_model_info(action_model, cfg, use_train_input=False)
cu.load_test_checkpoint(cfg, action_model)



num_frames = 32
sampling_rate = 2


def draw_text(img, text,
          pos=(0, 0),
          font=cv2.FONT_HERSHEY_PLAIN,
          font_scale=3,
          text_color=(0, 255, 0),
          font_thickness=2,
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    # font_scale = 1
    # font = cv2.FONT_HERSHEY_PLAIN
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, (x, y - text_h - 10), (x + text_w + 10, y), text_color_bg, -1)
    cv2.putText(img, text, (x+5, y-5), font, font_scale, text_color, font_thickness)




def draw_bb_text(frame, text, bbox,
                 font=cv2.FONT_HERSHEY_PLAIN,
                 base_font_scale=3,
                 text_color=(0, 255, 0),
                 base_font_thickness=2,
                 text_color_bg=(255, 255, 255)):
    # Get frame dimensions
    frame_h, frame_w = frame.shape[:2]

    # Calculate scaling factors
    scale_factor = max(frame_w, frame_h) / 500  # Normalize scaling to a base size (500px)
    font_scale = base_font_scale * scale_factor
    font_thickness = max(1, int(base_font_thickness * scale_factor))
    padding = max(2, int(4 * scale_factor))  # Additional padding around text

    # Extract bounding box coordinates
    startX, startY, endX, endY = bbox

    # Get text size
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size

    # Adjust text box height and padding
    tboxh = text_h + padding * 2

    # Ensure coordinates stay within bounds
    startY = max(tboxh, startY)
    startX = max(1, startX)

    # Create background for text
    bg = np.ones_like(frame[startY-tboxh:startY, startX-1:startX+text_w+3]).astype('uint8') * 255
    bg[:, :] = text_color_bg
    frame[startY-tboxh:startY, startX-1:startX+text_w+3] = cv2.addWeighted(
        frame[startY-tboxh:startY, startX-1:startX+text_w+3],
        0.0, bg, 1.0, 1
    )

    # Draw the text
    # print(f"font ; {font}, font_scale : {font_scale}, font_thickness : {font_thickness}")
    cv2.putText(frame, text, (startX, startY-padding),
                font, font_scale, text_color, font_thickness)


def get_color_from_id(idx):
    idx = idx * 3
    color = (int((37 * idx) % 255), int((17 * idx) % 255), int((29 * idx) % 255))

    return color

def isLightOrDark(rgbColor=[0,128,255]):
    [r,g,b]=rgbColor
    hsp = np.sqrt(0.299 * (r * r) + 0.587 * (g * g) + 0.114 * (b * b))
    if (hsp>127.5):
        return [0, 0, 0]
    else:
        return [255, 255, 255]


def format_tracking_results(tracking_results, target_classes=None, classlist = ["adult","child","mobile","table","bookstand","chair","chairs", "hurdle", "crib", "book", "food"]):
    for tracking_result in tracking_results:
        bboxes = tracking_result.boxes.cpu().numpy()
        tracking_results = []
        for xyxy, tid, class_id in zip(bboxes.xyxy, bboxes.id, bboxes.cls):
            center_x = int((xyxy[0] + xyxy[2]) / 2)
            center_y = int((xyxy[1] + xyxy[3]) / 2)
            if target_classes is None:
                tracking_results.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), int(tid), classlist[int(class_id)], center_x, center_y])
            elif class_id in target_classes:
                tracking_results.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), int(tid), classlist[int(class_id)], center_x, center_y])
        return tracking_results

def format_detection_results(detection_results, target_classes=None, classlist = ["adult","child","mobile","table","bookstand","chair","chairs", "hurdle", "crib", "book", "food"]):
    for detection_result in detection_results:
        bboxes = detection_result.boxes.cpu().numpy()
        detection_results = []
        for xyxy, class_id in zip(bboxes.xyxy, bboxes.cls):
            center_x = int((xyxy[0] + xyxy[2]) / 2)
            center_y = int((xyxy[1] + xyxy[3]) / 2)
            if target_classes is None:
                detection_results.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), classlist[int(class_id)], center_x, center_y])
            elif class_id in target_classes:
                detection_results.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), classlist[int(class_id)], center_x, center_y])
        return detection_results



temp_video_list = [os.path.join("specific_long_videos_2", x) for x in os.listdir("../data/daycare_fall/Secaucus_data/specific_long_videos_2")]

results1_df = pd.read_csv("slowfast_predictions_long_videos.csv")
results1 = results1_df.to_dict("records")

# import time
# while True:
#     time.sleep(1)
#     print("running!")

# %%time

# results = []
# results1 = []
# results1_df = pd.read_csv("slowfast_predictions_long_videos.csv")
# results1 = results1_df.to_dict("records")

video_meta = []

for video_name in temp_video_list:#falling_test_videos:
    

    video_path = f"../data/daycare_fall/Secaucus_data/{video_name}"

    if not os.path.exists(video_path):
        continue

    print(f"starting video - {video_name}")
    
    # video_name = os.path.basename(video_path)
    output_dir = "../out_long_videos"
    output_video_filename = os.path.basename(video_name) + ".webm"
    
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, output_video_filename)
    print(f'Writing video to {output_video_path}')
    
    writer = None
    fourcc = cv2.VideoWriter_fourcc('V', 'P', '8', '0')
    
    video = cv2.VideoCapture(video_path)
    
    video_fps = video.get(cv2.CAP_PROP_FPS)
    print(f'Video FPS : {video_fps}')
    
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    display_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    display_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_meta.append({"video_name" : video_name, "video_fps" : video_fps, "total_frames" : total_frames, "total_duration" : total_frames // video_fps})
    # continue
    
    # seq_length = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE
    # no_frames_repeat = cfg.DEMO.SLOWMO
    
    current_fps = 0
    fps = FPS().start()
    frame_no = -1
    
    all_fpses = []
    
    frame_buffer = []
    clip_buffer = []

    num_continue = 0



    while True:
    
        _, frame = video.read()
        frame_no += 1
    
        if frame is None:
            break
            # video.release()
            # video = cv2.VideoCapture(video_path)
            # _, frame = video.read()
    
    
        # if frame.shape[0] > frame.shape[1]:
        #     frame = imutils.resize(frame, height=500)
        # else:
        #     frame = imutils.resize(frame, width=500)
    


        frame_buffer.append(frame)
        frame_buffer = frame_buffer[-cfg.DATA.NUM_FRAMES:]
    
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
        rgb = scale(cfg.DATA.TEST_CROP_SIZE, rgb)
        clip_buffer.append(rgb)
        clip_buffer = clip_buffer[-cfg.DATA.NUM_FRAMES:]
    
    
        if len(clip_buffer) < cfg.DATA.NUM_FRAMES:
            num_continue += 1 
            continue
    


        inputs = process_cv2_inputs(clip_buffer, cfg)


    
        # break
    
        frame = frame_buffer[(cfg.DATA.NUM_FRAMES // 2)-1]
        # tracking_results = yolov8_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, stream=True)
        # tracking_results = format_tracking_results(tracking_results, target_classes=[0])
        
        try:
            # detection_results = yolov8_model(frame, verbose=False, stream=False)
            # detection_results = format_detection_results(detection_results, target_classes=[0,1])
            tracking_results = yolov8_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, stream=True)
            tracking_results = format_tracking_results(tracking_results, target_classes=[0, 1])
        except Exception as e:
            print(f"e : {e}")
            # detection_results = []
            tracking_results = []
    
        if len(tracking_results) == 0:
            num_continue += 1
            continue

        boxes = [detection_result[:4] for detection_result in tracking_results]
        boxes = torch.from_numpy(np.array(boxes)).float()
    
        box_transformed = scale_boxes(
            cfg.DATA.TEST_CROP_SIZE,
            boxes,
            frame.shape[0],
            frame.shape[1],
        )
        
        # Pad frame index for each box.
        box_inputs = torch.cat(
            [
                torch.full((box_transformed.shape[0], 1), float(0)),
                box_transformed,
            ],
            axis=1,
        )
        box_inputs = torch.cat(
            [
                torch.full((boxes.shape[0], 1), float(0)),
                boxes,
            ],
            axis=1,
        )


        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
    
            box_inputs = box_inputs.cuda()
    
        preds = action_model(inputs, box_inputs)
    
        preds = preds.detach()
    
        if cfg.NUM_GPUS:
            preds = preds.cpu()
    
        preds = preds.numpy()


        # print(f"preds : {preds}")
       
        # break

        threshs = {'fall1': 0.082, 'fall2': 0.011, 'stand': 0.05, 'sit': 0.11}

        for box_idx, detection_result in enumerate(tracking_results):
            class_id = np.argmax(preds[box_idx])
            # pred_class = action_class_list[class_id]
            # action_list = np.array(action_class_list)[preds[box_idx] > 0.2].tolist()
            x1, y1, x2, y2, tid, class_id = detection_result[:6]
            

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            x1_norm, y1_norm, x2_norm, y2_norm = x1 / display_width, y1 / display_height, x2 / display_width, y2 / display_height

            
            fall_conf = preds[box_idx][ava_classes.index('fall down')]
            stand_conf = preds[box_idx][ava_classes.index('stand')]
            sit_conf = preds[box_idx][ava_classes.index('sit')]
            walk_conf = preds[box_idx][ava_classes.index("walk")]
            crawl_conf = preds[box_idx][ava_classes.index("crawl")]
            run_conf = preds[box_idx][ava_classes.index("run/jog")]
            getup_conf = preds[box_idx][ava_classes.index("get up")]
            jump_conf = preds[box_idx][ava_classes.index("jump/leap")]
            sleep_conf = preds[box_idx][ava_classes.index("lie/sleep")]
            talk_conf = preds[box_idx][ava_classes.index("talk to (e.g., self, a person, a group)")]
            hit_conf = preds[box_idx][ava_classes.index('hit (an object)')]
            fight_conf = preds[box_idx][ava_classes.index('fight/hit (a person)')]
            push_conf = preds[box_idx][ava_classes.index('push (another person)')]


            
            fall_detected = (fall_conf > threshs["fall1"]) or (fall_conf > threshs["fall2"] and stand_conf < threshs["stand"] and sit_conf < threshs["sit"])
            hit_detected = hit_conf > 0.1
            fight_detected = fight_conf > 0.1
            push_detected = push_conf > 0.1

            results1.append({"video_name" : video_name, "timestamp" : round(frame_no / video_fps, 1), "frame_no" : frame_no, "box_idx" : box_idx, "detection" : [x1, y1, x2, y2, class_id],
                             "obj_class" : class_id, "tid" : tid, "xmin" : x1, "ymin" : y1, "xmax" : x2, "ymax" : y2, "x1_norm" : x1_norm, "y1_norm" : y1_norm, "x2_norm" : x2_norm, "y2_norm" : y2_norm, 
                            "center_x" : center_x, "center_y" : center_y, "frame_width" : display_width, "frame_height" : display_height,
                            "action_probs" : preds[box_idx], "fall_conf" : fall_conf, "stand_conf" : stand_conf, "sit_conf" : sit_conf, "walk_conf" : walk_conf, 
                            "run_conf" : run_conf, "crawl_conf" : crawl_conf, "getup_conf" : getup_conf, "jump_conf" : jump_conf, 
                            "sleep_conf" : sleep_conf, "talk_conf" : talk_conf, "hit_conf" : hit_conf, "fight_conf" : fight_conf, "push_conf" : push_conf,
                            "fall_detected" : fall_detected}) 
    
            # results.append({"video_name" : video_name, "timestamp" : round(frame_no / int(video_fps), 1), "frame_no" : frame_no, "box_idx" : box_idx, "detection" : [x1, y1, x2, y2, class_id],
            #                 "action_probs" : preds[box_idx], "fall_conf" : fall_conf, "hit_conf" : hit_conf, "fight_conf" : fight_conf, "push_conf" : push_conf,
            #                "fall_detected" : fall_detected, "hit_detected" : hit_detected, "fight_detected" : fight_detected, "push_detected" : push_detected}) 
    
            bg_color = [50, 50, 255]#get_color_from_id(tid)
            text_color = isLightOrDark(bg_color)
    
            startX, startY, endX, endY = x1, y1, x2, y2

            cv2.rectangle(frame, (startX, startY), (endX, endY), [200, 200, 200], max(1, int(2 * np.min(frame.shape[:2]) / 500)))
            
            if fall_detected:
                cv2.rectangle(frame, (startX, startY), (endX, endY), bg_color, max(1, int(2 * np.min(frame.shape[:2]) / 500)))
                draw_bb_text(frame,f"Fall {str(round(fall_conf, 2))[:4]}", (startX, startY, endX, endY),cv2.FONT_HERSHEY_DUPLEX, 0.3, text_color, 1, bg_color)
                print(f"frame_no - {frame_no}, Fall Detected!")


            
            # elif hit_detected:
            #     cv2.rectangle(frame, (startX, startY), (endX, endY), bg_color, max(1, int(2 * np.min(frame.shape[:2]) / 500)))
            #     draw_bb_text(frame,f"Hit {str(round(hit_conf, 2))[:4]}", (startX, startY, endX, endY),cv2.FONT_HERSHEY_DUPLEX, 0.3, text_color, 1, bg_color)
            #     print(f"frame_no - {frame_no}, Hit Detected!")
        
            # elif fight_detected:
            #     cv2.rectangle(frame, (startX, startY), (endX, endY), bg_color, max(1, int(2 * np.min(frame.shape[:2]) / 500)))
            #     draw_bb_text(frame,f"Fight {str(round(fight_conf, 2))[:4]}", (startX, startY, endX, endY),cv2.FONT_HERSHEY_DUPLEX, 0.3, text_color, 1, bg_color)
            #     print(f"frame_no - {frame_no}, Fight Detected!")
        
            # elif push_detected:
            #     cv2.rectangle(frame, (startX, startY), (endX, endY), bg_color, max(1, int(2 * np.min(frame.shape[:2]) / 500)))
            #     draw_bb_text(frame,f"Push {str(round(push_conf, 2))[:4]}", (startX, startY, endX, endY),cv2.FONT_HERSHEY_DUPLEX, 0.3, text_color, 1, bg_color)
            #     print(f"frame_no - {frame_no}, Push Detected!")
        
        
        
        # if writer == None:
        #     writer = cv2.VideoWriter(output_video_path, fourcc, int(video_fps), (frame.shape[1], frame.shape[0]))
    
        # writer.write(frame)
    
    
        if frame_no != 0 and frame_no % 250 == 0:
            print(f"video_name : {video_name}, frame_no : {frame_no}/{total_frames} => Mean FPS : {round(np.mean(all_fpses), 2)}, STD FPS : {round(np.std(all_fpses), 2)}")
    

        # if frame_no == 9409:
        #     break
    
        fps.update()
        fps.stop()
        current_fps = round(fps.fps(), 2)
        all_fpses.append(current_fps)
        all_fpses = all_fpses[-100:]
        # time.sleep(0.02)
        
        # break
    
    
    # video.release()
    # writer.release()
    
    fps_mean = round(np.mean(all_fpses),4)
    fps_std = round(np.std(all_fpses), 4)
    
    print(f'Mean FPS : {fps_mean}, STD FPS : {fps_std}')

    # results1_df = pd.DataFrame(results1)
    # results1_df.to_csv("slowfast_predictions_long_videos_2.csv", index=False)

    # break