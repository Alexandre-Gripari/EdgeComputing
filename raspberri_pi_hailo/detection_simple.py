import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import hailo
import time

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

if __name__ == "__main__":
    print("Lancement du stream sur le port 8000...")
    
    Gst.init(None)
    
    pipeline_str = (
        "libcamerasrc ! "
        "video/x-raw, width=1296, height=972, format=NV12, framerate=30/1 ! "
        "videocrop left=162 right=162 ! " 
        "videoscale ! "
        "video/x-raw, width=640, height=640, format=RGB ! "
        "videoconvert ! "
        "queue max-size-buffers=3 leaky=downstream ! "
        "hailonet hef-path=/home/yanis/edge_computing/yolov11s.hef batch-size=1 "
        "nms-score-threshold=0.3 nms-iou-threshold=0.45 output-format-type=HAILO_FORMAT_TYPE_UINT8 ! "
        "queue max-size-buffers=3 leaky=downstream ! "
        "hailofilter name=filter so-path=/usr/local/hailo/resources/so/libyolo_hailortpp_postprocess.so "
        "function-name=filter qos=false ! "
        "queue max-size-buffers=3 leaky=downstream ! "
        "hailooverlay ! "
        "videoconvert ! "
        "jpegenc quality=85 ! "
        "multipartmux boundary=spboundary ! "
        "tcpserversink host=0.0.0.0 port=8000 sync=false"
    )
    
    try:
        pipeline = Gst.parse_launch(pipeline_str)
    except Exception as e:
        print(f"Erreur pipeline: {e}")
        exit(1)
    
    hailo_filter = pipeline.get_by_name("filter")
    if hailo_filter:
        src_pad = hailo_filter.get_static_pad("src")
        user_data = UserData()
        src_pad.add_probe(Gst.PadProbeType.BUFFER, app_callback, user_data)
    
    pipeline.set_state(Gst.State.PLAYING)
    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nArrêt...")
        pipeline.set_state(Gst.State.NULL)