
import subprocess
import numpy as np

class FFmpegGPUSaver:
    def __init__(self, output_path, width, height, fps=30, bitrate_kbps=2000):
        self.process = subprocess.Popen([
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', str(fps),
            '-i', '-',
            '-an',
            '-c:v', 'hevc_amf',
            '-quality', 'quality',
            '-b:v', f'{bitrate_kbps}k',
            '-maxrate', f'{int(bitrate_kbps * 1.3)}k',
            '-bufsize', f'{int(bitrate_kbps * 2)}k',
            '-g', '25',
            '-movflags', '+faststart',
            '-avoid_negative_ts', '1',
            '-fflags', '+genpts',
            output_path
        ], stdin=subprocess.PIPE)

    def write_frame(self, frame: np.ndarray):
        if self.process and self.process.stdin:
            self.process.stdin.write(frame.tobytes())

    def close(self):
        if self.process:
            self.process.stdin.close()
            self.process.wait()
