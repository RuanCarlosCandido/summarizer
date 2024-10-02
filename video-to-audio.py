from moviepy.editor import VideoFileClip

def video_to_audio(video_path, audio_path):
    # Carrega o vídeo
    video = VideoFileClip(video_path)
    
    # Extrai o áudio
    audio = video.audio
    
    # Salva o áudio no caminho especificado
    audio.write_audiofile(audio_path)

# Exemplo de uso
video_path = 'v4.mp4'
audio_path = './v4.mp3'
video_to_audio(video_path, audio_path)

