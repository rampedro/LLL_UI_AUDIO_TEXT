from pydub import AudioSegment
from pydub.effects import high_pass_filter
from pydub.silence import split_on_silence
from os import listdir, path, remove
import torchaudio
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
from noisereduce import reduce_noise
import numpy as np


class SpeechToText:
    def __init__(self, src_lang="eng", tgt_lang="eng"):
        self.processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
        self.model = SeamlessM4Tv2ForSpeechToText.from_pretrained("facebook/seamless-m4t-v2-large")
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def transcribe_audio(self, audio_file):
        audio, orig_freq = torchaudio.load(audio_file)
        audio_inputs = self.processor(audios=audio, src_lang=self.src_lang, return_tensors="pt", sampling_rate=orig_freq)

        # Read and transcribe each audio file
        output_tokens = self.model.generate(**audio_inputs, tgt_lang=self.tgt_lang)[0].cpu().numpy().squeeze()
        transcription = self.processor.decode(output_tokens.tolist(), skip_special_tokens=True, tgt_lang=self.tgt_lang)

        return transcription


class Preprocess:
    def __init__(self, export_path):
        self.audio_chunks = []
        self.export_path = export_path
        self.keep_silence = 250

    def process_audio(self, audio):
        chunks = self.split_audio(audio)
        self.export_audio_chunks(chunks)

    def split_audio(self, input_file):
        audio = AudioSegment.from_file(input_file)

        audio = self.denoiser(audio)

        # resample the audio to 16kHz
        audio = audio.set_frame_rate(16000)

        normalized_audio = audio.normalize()

        audio = high_pass_filter(normalized_audio, cutoff=1000)

        # Step 1: Split with silence longer than 1 second
        segments_1s = split_on_silence(
            audio,
            min_silence_len=1000,  # 1 second
            silence_thresh=audio.dBFS - 16,
            keep_silence=self.keep_silence
        )


        # Step 2: Split chunks longer than 10 seconds with 0.5 seconds silence
        segments_10s = []
        for segment_1s in segments_1s:
            if len(segment_1s) > 6000:  # 10 seconds
                segments_10s.extend(
                    split_on_silence(
                        segment_1s,
                        min_silence_len=750,  # 0.5 second
                        silence_thresh=audio.dBFS - 17,
                        keep_silence=self.keep_silence
                    )
                )
            else:
                segments_10s.append(segment_1s)

        # Step 3: Split remaining chunks longer than 10 seconds with 0.25 seconds silence
        final_segments = []
        for segment_10s in segments_10s:
            if len(segment_10s) > 6000:  # 10 seconds
                final_segments.extend(
                    split_on_silence(
                        segment_10s,
                        min_silence_len=300,  # 0.25 second
                        silence_thresh=audio.dBFS - 20,
                        keep_silence=self.keep_silence
                    )
                )
            else:
                final_segments.append(segment_10s)

        # if there are still audios longer than 9 seconds, split them into 5 seconds chunks by maintaining the order
        for i, segment in enumerate(final_segments):
            if len(segment) > 9000:
                final_segments[i:i + 1] = segment[:6000], segment[6000:]

        # if there is a chunck with duration less than 1 second, merge it with the previous chunk
        i = 0
        while i < len(final_segments):
            if len(final_segments[i]) < 1000:
                final_segments[i - 1] += final_segments[i]
                del final_segments[i]
            else:
                i += 1

        return final_segments

    @staticmethod
    def denoiser(audio):
        # Convert audio to numpy array
        samples = np.array(audio.get_array_of_samples())

        # Reduce noise
        reduced_noise = reduce_noise(samples, sr=audio.frame_rate)

        # Convert reduced noise signal back to audio
        reduced_audio = AudioSegment(
            reduced_noise.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=audio.channels
        )

        return reduced_audio

    def export_audio_chunks(self, audio_chunks):
        # Export each segment as a separate file
        for i, segment in enumerate(audio_chunks, start=1):
            output_file = f"{self.export_path}/chunk_{i}.mp3"
            segment.export(output_file, format="mp3")

            # Calculate the duration of each chunk
            chunk_duration = len(segment)  # in milliseconds

            # Print the duration of each chunk
            print(f"Chunk {i} duration: {chunk_duration / 1000}s")


class Pipeline:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocess = preprocessor
        self.audio_folder = preprocessor.export_path
        self.export_path = "outputs/prediction.txt"

    def transcribe(self, audio):
        self.preprocess.process_audio(audio)
        audio_files = self.list_audio_files()

        transcriptions = []
        for audio in audio_files:
            audio_path = path.join(self.audio_folder, audio)
            transcription = self.model.transcribe_audio(audio_path)
            transcriptions.append(transcription)

        concatenated_string = ' '.join(sublist for sublist in transcriptions)

        concatenated_string = self.postprocess(concatenated_string)

        self.export_text(concatenated_string)

    @staticmethod
    def postprocess(text):
        # remove shard signs
        text = text.replace("#", "")
        # remove words that are repeated several times sequentially in a sentence, e.g. "hello hello hello" and  only keep one
        text = " ".join([word for i, word in enumerate(text.split()) if i == 0 or word != text.split()[i-1]])
        return text

    def list_audio_files(self):
        # List all audio files in the folder
        audio_files = [f for f in listdir(self.audio_folder) if f.startswith('chunk_')]

        # Sort the list of file names based on the numeric part
        audio_files = sorted(audio_files, key=self.extract_number)

        return audio_files

    @staticmethod
    def extract_number(file_name):
        return int(file_name.split('_')[1].split('.')[0])

    def export_text(self, text):
        with open(self.export_path, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"Transcription saved to {self.export_path}")


if __name__ == "__main__":
    # clear the outputs folder
    for f in listdir("outputs"):
        if f.endswith(".txt") or f.endswith(".mp3"):
            remove(path.join("outputs", f))

    pipeline = Pipeline(model=SpeechToText(src_lang="eng", tgt_lang="eng"),
                        preprocessor=Preprocess(export_path="outputs"))
    pipeline.transcribe("audio_files/Sample Virtual Dr visit with dictated Medical Report Nephrology.mp3")