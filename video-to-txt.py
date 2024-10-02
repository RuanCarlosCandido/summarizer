import asyncio
import concurrent.futures
import os
import speech_recognition as sr
from moviepy.editor import VideoFileClip
import subprocess
from tqdm.asyncio import tqdm
import logging

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_audio_segment(video_path, start_time, end_time, output_path):
    try:
        subprocess.run([
            "ffmpeg", "-i", video_path, "-ss", str(start_time), "-to", str(end_time), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_path
        ], check=True)
        logging.info(f"Áudio extraído: {output_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Erro ao extrair áudio: {str(e)}")
        return f"Erro ao extrair áudio: {str(e)}"

def transcribe_segment(audio_path, language='pt-BR'):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        transcription = recognizer.recognize_google(audio_data, language=language)
        logging.info(f"Transcrição concluída para: {audio_path}")
        return transcription
    except (sr.UnknownValueError, sr.RequestError) as e:
        error_msg = f"Erro ao transcrever: {str(e)}"
        logging.error(error_msg)
        return error_msg

async def extract_audio(video_path, segment_length, total_duration):
    extract_tasks = []
    audio_paths = []

    if not os.path.exists("temp_audio"):
        os.makedirs("temp_audio")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        loop = asyncio.get_running_loop()

        for start_time in range(0, total_duration, segment_length):
            end_time = min(start_time + segment_length, total_duration)
            audio_path = f"temp_audio/audio_{start_time}_{end_time}.wav"
            audio_paths.append(audio_path)

            task = loop.run_in_executor(executor, extract_audio_segment, video_path, start_time, end_time, audio_path)
            extract_tasks.append(task)

        for task in tqdm(asyncio.as_completed(extract_tasks), total=len(extract_tasks), desc="Extraindo áudio"):
            result = await task
            if isinstance(result, str) and result.startswith("Erro"):
                raise Exception(result)

    return audio_paths

async def transcribe_audio(audio_paths):
    transcribe_tasks = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        loop = asyncio.get_running_loop()

        for audio_path in audio_paths:
            task = loop.run_in_executor(executor, transcribe_segment, audio_path)
            transcribe_tasks.append(task)

        results = []
        for task in tqdm(asyncio.as_completed(transcribe_tasks), total=len(transcribe_tasks), desc="Transcrevendo"):
            result = await task
            results.append(result)

    return results

async def transcribe_video(video_path, segment_length=50):
    try:
        clip = VideoFileClip(video_path)
        total_duration = int(clip.duration)

        audio_paths = await extract_audio(video_path, segment_length, total_duration)
        transcriptions = await transcribe_audio(audio_paths)

        for audio_path in os.listdir("temp_audio"):
            os.remove(os.path.join("temp_audio", audio_path))

        return transcriptions
    except Exception as e:
        logging.error(f"Erro na transcrição do vídeo: {str(e)}")
        return [f"Erro na transcrição do vídeo: {str(e)}"]

def save_transcription(transcriptions, output_file='transcription.txt', error_file='transcription_errors.txt'):
    with open(output_file, 'w') as file, open(error_file, 'w') as err_file:
        for i, transcription in enumerate(transcriptions):
            if transcription.startswith("Erro"):
                error_code = f"ERRO+{i+1:03}"
                file.write(f"[{error_code}] ")
                err_file.write(f"{error_code}: {transcription}\n")
            else:
                file.write(transcription + " ")
        logging.info(f"Transcrição salva em {output_file}")
        logging.info(f"Erros salvos em {error_file}")

def main():
    video_path = 'v4.mp4'
    transcriptions = asyncio.run(transcribe_video(video_path))
    save_transcription(transcriptions)

if __name__ == "__main__":
    main()
