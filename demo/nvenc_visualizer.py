"""
NVENC Visualizer - Ultra-Fast GPU-Accelerated Video Encoding
Uses FFmpeg with NVIDIA NVENC (h264_nvenc) for real-time MP4 export at 80-150 FPS
"""

import cv2
import torch
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp
import subprocess
import logging

cv2.setNumThreads(0)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cv2_video_info(video_path):
    vid = cv2.VideoCapture(video_path)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = vid.get(cv2.CAP_PROP_FPS)
    frame_num = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    vid.release()
    return dict(
        width=int(width),
        height=int(height),
        fps=fps,
        frame_num=int(frame_num),
    )


class NVENCVisualizer(object):
    """Ultra-fast visualizer using FFmpeg with NVIDIA NVENC hardware encoding"""
    
    # category names are modified for better visualization
    CATEGORIES = [
        "bend/bow", "crawl", "crouch/kneel", "dance", "fall down", "get up",
        "jump/leap", "lie/sleep", "martial art", "run/jog", "sit", "stand",
        "swim", "walk", "answer phone", "brush teeth", "carry/hold sth.",
        "catch sth.", "chop", "climb", "clink glass", "close", "cook", "cut",
        "dig", "dress/put on clothing", "drink", "drive", "eat", "enter",
        "exit", "extract", "fishing", "hit sth.", "kick sth.", "lift/pick up",
        "listen to sth.", "open", "paint", "play board game",
        "play musical instrument", "play with pets", "point to sth.", "press",
        "pull sth.", "push sth.", "put down", "read", "ride", "row boat",
        "sail boat", "shoot", "shovel", "smoke", "stir", "take a photo",
        "look at a cellphone", "throw", "touch sth.", "turn", "watch screen",
        "work on a computer", "write", "fight/hit sb.", "give/serve sth. to sb.",
        "grab sb.", "hand clap", "hand shake", "hand wave", "hug sb.", "kick sb.",
        "kiss sb.", "lift sb.", "listen to sb.", "play with kids", "push sb.",
        "sing", "take sth. from sb.", "talk", "watch sb."
    ]
    
    COMMON_CATES = [
        'dance', 'run/jog', 'sit', 'stand', 'swim', 'walk', 'answer phone',
        'carry/hold sth.', 'drive', 'play musical instrument', 'ride',
        'fight/hit sb.', 'listen to sb.', 'talk', 'watch sb.'
    ]
    
    EXCLUSION = [
        "crawl", "brush teeth", "catch sth.", "chop", "clink glass", "cook",
        "dig", "exit", "extract", "fishing", "kick sth.", "paint",
        "play board game", "play with pets", "press", "row boat", "shovel",
        "stir", "kick sb.", "play with kids"
    ]
    
    def __init__(
            self,
            video_path,
            output_path,
            realtime,
            start,
            duration,
            show_time,
            confidence_threshold=0.5,
            exclude_class=None,
            common_cate=False,
            nvenc_preset='p1',  # p1 = fastest, p7 = slowest/best quality
            nvenc_bitrate='8M',
            nvenc_maxrate='12M',
    ):
        self.vid_info = cv2_video_info(video_path)
        fps = self.vid_info["fps"]
        if fps == 0 or fps > 100:
            logger.warning(f"The detected frame rate {fps} could be wrong.")

        self.realtime = realtime
        self.start = start
        self.duration = duration
        self.show_time = show_time
        self.confidence_threshold = confidence_threshold
        self.nvenc_preset = nvenc_preset
        self.nvenc_bitrate = nvenc_bitrate
        self.nvenc_maxrate = nvenc_maxrate
        
        if common_cate:
            self.cate_to_show = self.COMMON_CATES
            self.category_split = (6, 11)
        else:
            self.cate_to_show = self.CATEGORIES
            self.category_split = (14, 63)
            
        self.cls2label = {class_name: i for i, class_name in enumerate(self.cate_to_show)}
        
        if exclude_class is None:
            exclude_class = self.EXCLUSION
        self.exclude_id = [self.cls2label[cls_name] for cls_name in exclude_class if cls_name in self.cls2label]

        self.width = self.vid_info["width"]
        self.height = self.vid_info["height"]
        long_side = min(self.width, self.height)
        self.font_size = max(int(round((long_side / 40))), 1)
        self.box_width = max(int(round(long_side / 180)), 1)
        
        # OpenCV font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = self.font_size / 30.0
        self.font_thickness = max(int(self.font_scale * 2), 1)

        self.box_color = (191, 40, 41)  # BGR format for OpenCV
        self.category_colors = ((176, 85, 234), (87, 118, 198), (52, 189, 199))  # BGR
        self.category_alpha = 0.6

        self.action_dictionary = dict()

        if realtime:
            # For realtime, use OpenCV VideoWriter (NVENC not supported in realtime mode)
            width = self.vid_info["width"]
            height = self.vid_info["height"]
            fps = self.vid_info["fps"]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.out_vid = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        else:
            self.frame_queue = mp.JoinableQueue(512)
            self.result_queue = mp.JoinableQueue()
            self.track_queue = mp.JoinableQueue()
            self.done_queue = mp.Queue()
            self.frame_loader = mp.Process(target=self._load_frame, args=(video_path,))
            self.frame_loader.start()
            self.video_writer = mp.Process(target=self._write_frame_nvenc, args=(output_path,))
            self.video_writer.start()

    def realtime_write_frame(self, result, orig_img, boxes, scores, ids):
        if result is not None:
            result, timestamp, result_ids = result
            update_boxes = result.bbox
            update_scores = result.get_field("scores")
            update_ids = result_ids
            if update_boxes is not None:
                self.update_action_dictionary(update_scores, update_ids)

        if boxes is not None:
            orig_img = self.visual_frame_fast(orig_img, boxes, ids)

        cv2.imshow("my webcam", orig_img)
        self.out_vid.write(orig_img)

        if cv2.waitKey(1) == 27:
            return False
        return True

    def _load_frame(self, video_path):
        vid = cv2.VideoCapture(video_path)
        vid.set(cv2.CAP_PROP_POS_MSEC, self.start)
        vid_avail = True
        while True:
            vid_avail, frame = vid.read()
            if not vid_avail:
                break
            mills = vid.get(cv2.CAP_PROP_POS_MSEC)
            if self.duration != -1 and mills > self.start + self.duration:
                break
            self.frame_queue.put((frame, mills))

        vid.release()
        self.frame_queue.put("DONE")
        self.frame_queue.join()
        self.frame_queue.close()

    def _write_frame_nvenc(self, output_path):
        """Write frames using FFmpeg with NVENC hardware acceleration"""
        width = self.vid_info["width"]
        height = self.vid_info["height"]
        fps = self.vid_info["fps"]

        # FFmpeg command for NVENC encoding
        # Using h264_nvenc with ultra-low latency settings
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',  # OpenCV uses BGR24
            '-s', f'{width}x{height}',
            '-r', str(fps),
            '-i', '-',  # Read from stdin
            '-c:v', 'h264_nvenc',  # NVIDIA hardware encoder
            '-preset', self.nvenc_preset,  # p1 = fastest, p7 = best quality
            '-tune', 'll',  # Low latency
            '-rc', 'vbr',  # Variable bitrate
            '-b:v', self.nvenc_bitrate,  # Target bitrate
            '-maxrate', self.nvenc_maxrate,  # Max bitrate
            '-bufsize', '16M',  # Buffer size
            '-bf', '0',  # No B-frames for low latency
            '-g', str(int(fps * 2)),  # GOP size (2 seconds)
            '-movflags', '+faststart',  # Enable fast start for web playback
            output_path
        ]

        logger.info(f"Starting FFmpeg NVENC encoder: {' '.join(ffmpeg_cmd)}")
        
        try:
            # Start FFmpeg process
            ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8  # Large buffer for smooth streaming
            )
            
            has_frame = True
            frame_count = 0

            result = self.result_queue.get()
            timestamp = float('inf')
            result_ids = None
            if not isinstance(result, str):
                result, timestamp, result_ids = result
                
            while has_frame:
                track_result = self.track_queue.get()

                # read frame
                data = self.frame_queue.get()
                self.frame_queue.task_done()

                if isinstance(result, str) and data == "DONE":
                    self.track_queue.task_done()
                    self.result_queue.task_done()
                    break

                # note that the timestamp should be in milliseconds
                frame, mills = data

                if self.show_time:
                    frame = self.visual_timestamp_fast(frame, mills)
                    
                if mills - timestamp + 0.5 > 0:
                    boxes = result.bbox
                    scores = result.get_field("scores")
                    ids = result_ids

                    self.result_queue.task_done()
                    result = self.result_queue.get()
                    if not isinstance(result, str):
                        result, timestamp, result_ids = result
                    else:
                        timestamp = float('inf')
                else:
                    boxes, ids = track_result
                    scores = None

                if boxes is not None:
                    self.update_action_dictionary(scores, ids)
                    new_frame = self.visual_frame_fast(frame, boxes, ids)
                    # Write frame to FFmpeg stdin
                    try:
                        ffmpeg_process.stdin.write(new_frame.tobytes())
                    except BrokenPipeError:
                        logger.error("FFmpeg pipe broken!")
                        break
                else:
                    try:
                        ffmpeg_process.stdin.write(frame.tobytes())
                    except BrokenPipeError:
                        logger.error("FFmpeg pipe broken!")
                        break

                frame_count += 1
                self.track_queue.task_done()
                self.done_queue.put(True)

            # Close FFmpeg stdin and wait for process to finish
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()
            
            # Check for errors
            if ffmpeg_process.returncode != 0:
                stderr_output = ffmpeg_process.stderr.read().decode()
                logger.error(f"FFmpeg error: {stderr_output}")
            else:
                logger.info(f"Successfully encoded {frame_count} frames with NVENC")
                tqdm.write("The output video has been written to the disk.")
                
        except FileNotFoundError:
            logger.error("FFmpeg not found! Please install FFmpeg with NVENC support.")
            logger.error("Falling back to OpenCV VideoWriter...")
            # Fallback to OpenCV writer
            self._write_frame_opencv_fallback(output_path)
        except Exception as e:
            logger.error(f"Error during NVENC encoding: {e}")
            raise

    def _write_frame_opencv_fallback(self, output_path):
        """Fallback to OpenCV VideoWriter if FFmpeg fails"""
        width = self.vid_info["width"]
        height = self.vid_info["height"]
        fps = self.vid_info["fps"]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_vid = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out_vid.isOpened():
            logger.warning("Failed to open video writer with mp4v, trying XVID...")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            if output_path.endswith('.mp4'):
                output_path = output_path.replace('.mp4', '.avi')
            out_vid = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        has_frame = True

        result = self.result_queue.get()
        timestamp = float('inf')
        result_ids = None
        if not isinstance(result, str):
            result, timestamp, result_ids = result
            
        while has_frame:
            track_result = self.track_queue.get()

            data = self.frame_queue.get()
            self.frame_queue.task_done()

            if isinstance(result, str) and data == "DONE":
                self.track_queue.task_done()
                self.result_queue.task_done()
                break

            frame, mills = data

            if self.show_time:
                frame = self.visual_timestamp_fast(frame, mills)
                
            if mills - timestamp + 0.5 > 0:
                boxes = result.bbox
                scores = result.get_field("scores")
                ids = result_ids

                self.result_queue.task_done()
                result = self.result_queue.get()
                if not isinstance(result, str):
                    result, timestamp, result_ids = result
                else:
                    timestamp = float('inf')
            else:
                boxes, ids = track_result
                scores = None

            if boxes is not None:
                self.update_action_dictionary(scores, ids)
                new_frame = self.visual_frame_fast(frame, boxes, ids)
                out_vid.write(new_frame)
            else:
                out_vid.write(frame)

            self.track_queue.task_done()
            self.done_queue.put(True)

        out_vid.release()
        tqdm.write("The output video has been written to the disk (OpenCV fallback).")

    def hou_min_sec(self, total_millis):
        total_millis = int(total_millis)
        millis = total_millis % 1000
        total_millis //= 1000
        seconds = total_millis % 60
        total_millis //= 60
        minutes = total_millis % 60
        total_millis //= 60
        hours = total_millis
        return "%02d:%02d:%02d.%03d" % (hours, minutes, seconds, millis)

    def visual_timestamp_fast(self, frame, mills):
        """Fast timestamp overlay using OpenCV"""
        time_text = self.hou_min_sec(mills)
        
        (text_w, text_h), baseline = cv2.getTextSize(
            time_text, self.font, self.font_scale, self.font_thickness
        )
        
        padding = 10
        bg_x1, bg_y1 = 0, frame.shape[0] - text_h - padding * 2
        bg_x2, bg_y2 = text_w + padding * 2, frame.shape[0]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        text_x, text_y = padding, frame.shape[0] - padding - baseline
        cv2.putText(frame, time_text, (text_x, text_y),
                   self.font, self.font_scale, (255, 255, 255), self.font_thickness)
        
        return frame

    def update_action_dictionary(self, scores, ids):
        """Update action_dictionary"""
        if scores is not None:
            for score, id in zip(scores, ids):
                show_idx = torch.nonzero(score >= self.confidence_threshold, as_tuple=False).squeeze(1)
                captions = []
                bg_colors = []

                for category_id in show_idx:
                    if category_id in self.exclude_id:
                        continue
                    label = self.cate_to_show[category_id]
                    conf = " %.2f" % score[category_id]
                    caption = label + conf
                    captions.append(caption)
                    if category_id < self.category_split[0]:
                        bg_colors.append(0)
                    elif category_id < self.category_split[1]:
                        bg_colors.append(1)
                    else:
                        bg_colors.append(2)

                self.action_dictionary[int(id)] = {
                    "captions": captions,
                    "bg_colors": bg_colors,
                }

    def visual_frame_fast(self, frame, boxes, ids):
        """Fast visualization using OpenCV"""
        overlay = frame.copy()
        
        bboxes = boxes
        
        for box, id in zip(bboxes, ids):
            x1, y1, x2, y2 = map(int, box.tolist())
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.box_color, self.box_width)
            
            caption_and_color = self.action_dictionary.get(int(id), None)
            
            if caption_and_color is None:
                continue
                
            captions = caption_and_color['captions']
            bg_colors = caption_and_color['bg_colors']
            
            if len(captions) == 0:
                continue
            
            caption_sizes = []
            for caption in captions:
                (text_w, text_h), _ = cv2.getTextSize(
                    caption, self.font, self.font_scale, self.font_thickness
                )
                caption_sizes.append((text_w, text_h))
            
            padding = 4
            y_offset = y1
            
            for i, (caption, (text_w, text_h)) in enumerate(zip(captions, caption_sizes)):
                label_y1 = max(y_offset - text_h - padding * 2, padding)
                label_y2 = label_y1 + text_h + padding * 2
                label_x1 = x1
                label_x2 = x1 + text_w + padding * 2
                
                color = self.category_colors[bg_colors[i]]
                cv2.rectangle(overlay, (label_x1, label_y1), (label_x2, label_y2), color, -1)
                
                text_x = x1 + padding
                text_y = label_y1 + text_h + padding
                cv2.putText(frame, caption, (text_x, text_y),
                           self.font, self.font_scale, (255, 255, 255), self.font_thickness)
                
                y_offset = label_y1
        
        cv2.addWeighted(overlay, self.category_alpha, frame, 1 - self.category_alpha, 0, frame)
        
        return frame

    def send(self, result):
        self.result_queue.put(result)

    def send_track(self, result):
        self.track_queue.put(result)

    def close(self):
        if self.realtime:
            self.out_vid.release()
        else:
            self.result_queue.join()
            self.result_queue.close()

            self.track_queue.join()
            self.track_queue.close()

    def progress_bar(self, total):
        cnt = 0
        while not self.done_queue.empty():
            _ = self.done_queue.get()
            cnt += 1
        pbar = tqdm(total=total, initial=cnt, desc="NVENC Video Writer", unit=" frame")
        while cnt < total:
            _ = self.done_queue.get()
            cnt += 1
            pbar.update(1)
        pbar.close()

