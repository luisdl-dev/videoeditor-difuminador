import subprocess
import numpy as np

class FFmpegCPUSaver:
    def __init__(self, output_path, width, height, fps=30, bitrate_kbps=2000):
        """
        Saver de video usando FFmpeg en CPU (libx264), compatible con tu flujo de ExportThread.
        """
        self.process = subprocess.Popen([
            'ffmpeg',
            '-y',                        # Sobrescribir archivo si existe
            '-f', 'rawvideo',            # Formato de entrada raw
            '-vcodec', 'rawvideo',       # Codec de entrada raw
            '-pix_fmt', 'bgr24',         # Formato de los frames
            '-s', f'{width}x{height}',   # Resolución
            '-r', str(fps),              # FPS
            '-i', '-',                   # Leer desde stdin
            '-an',                       # Sin audio
            '-c:v', 'libx264',           # Codec de video CPU
            '-preset', 'fast',           # Preset rápido
            '-b:v', f'{bitrate_kbps}k', # Bitrate
            '-maxrate', f'{int(bitrate_kbps * 1.3)}k',
            '-bufsize', f'{int(bitrate_kbps * 2)}k',
            '-g', '25',                  # GOP
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',       # Formato compatible MP4
            output_path
        ], stdin=subprocess.PIPE)

    def write_frame(self, frame: np.ndarray):
        """
        Escribe un frame en el proceso FFmpeg.
        """
        if self.process and self.process.stdin:
            self.process.stdin.write(frame.tobytes())

    def close(self):
        """
        Cierra el proceso FFmpeg y espera que termine.
        """
        if self.process:
            self.process.stdin.close()
            self.process.wait()

