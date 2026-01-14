import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import hailo
import time
import os
import glob
import sys
import threading

class UserData:
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.last_time = time.time()
        self.fps_interval_count = 0

    def increment(self):
        self.frame_count += 1
        self.fps_interval_count += 1
        
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            fps = self.fps_interval_count / (current_time - self.last_time)
            print(f"FPS: {fps:.2f}")
            self.last_time = current_time
            self.fps_interval_count = 0

    def get_count(self):
        return self.frame_count

def app_callback(pad, info, user_data):
    user_data.increment()
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    return Gst.PadProbeReturn.OK

def on_message(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("Fin du flux vidéo (EOS)")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Erreur: {err}, {debug}")
        loop.quit()
    return True

def process_video(video_path):
    print(f"Processing video: {video_path}")
    
    pipeline_str = (
        f"filesrc location={video_path} ! "
        "decodebin ! "
        "videoconvert ! "
        "videoscale ! "
        "video/x-raw, width=640, height=640, format=RGB ! "
        "queue max-size-buffers=3 ! "
        "hailonet hef-path=/home/yanis/edge_computing/yolov11s.hef batch-size=1 nms-score-threshold=0.3 nms-iou-threshold=0.45 output-format-type=HAILO_FORMAT_TYPE_FLOAT32 ! "
        "queue max-size-buffers=3 ! "
        "hailofilter name=filter so-path=/usr/local/hailo/resources/so/libyolo_hailortpp_postprocess.so function-name=filter qos=false ! "
        "queue max-size-buffers=3 ! "
        "hailooverlay ! "
        "videoconvert ! "
        "x264enc tune=zerolatency speed-preset=ultrafast bitrate=5000 ! "
        "h264parse ! "
        "mpegtsmux ! "
        "tcpserversink name=sink host=0.0.0.0 port=8000 sync=false"
    )

    try:
        pipeline = Gst.parse_launch(pipeline_str)
    except Exception as e:
        print(f"Erreur création pipeline pour {video_path}: {e}")
        return

    hailo_filter = pipeline.get_by_name("filter")
    if hailo_filter:
        src_pad = hailo_filter.get_static_pad("src")
        user_data = UserData()
        src_pad.add_probe(Gst.PadProbeType.BUFFER, app_callback, user_data)
    
    sink = pipeline.get_by_name("sink")
    if sink:
        client_event = threading.Event()
        
        def block_callback(pad, info, user_data=None):
            client_event.wait()
            return Gst.PadProbeReturn.REMOVE
            
        sink_pad = sink.get_static_pad("sink")
        sink_pad.add_probe(Gst.PadProbeType.BLOCK_DOWNSTREAM, block_callback)

        def on_client_added(element, socket, pipe):
            print("Client connected! Starting playback...")
            client_event.set()
            GLib.idle_add(pipe.set_state, Gst.State.PLAYING)
            
        sink.connect("client-added", on_client_added, pipeline)
    
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, loop)
    
    print("Waiting for client connection on port 8000...")
    pipeline.set_state(Gst.State.PAUSED)
    
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nInterruption utilisateur...")
        pipeline.set_state(Gst.State.NULL)
        sys.exit(0)
    finally:
        pipeline.set_state(Gst.State.NULL)
        time.sleep(2.0)

if __name__ == "__main__":
    Gst.init(None)
    
    video_dir = "/home/yanis/edge_computing/videos"
    video_extensions = ['*.mp4', '*.webm', '*.avi', '*.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, ext)))
        
    video_files.sort()
    
    if not video_files:
        print(f"Aucune vidéo trouvée dans {video_dir}")
        exit(1)
        
    print(f"Trouvé {len(video_files)} vidéos. Lancement du traitement...")
    
    import gc
    for video_file in video_files:
        process_video(video_file)
        gc.collect()
        print("-" * 50)
        time.sleep(1)
        
    print("Traitement terminé pour toutes les vidéos.")
