import os
import numpy as np
import extract_feats.opensmile as of
import extract_feats.librosa as lf
import models
import utils
from pydub import AudioSegment
import shutil
import sounddevice as sd
import scipy.io.wavfile as wav
import time
from attachments.inaSpeechSegmenter.inaSpeechSegmenter import Segmenter


def predict(config, audio_path: str, model) -> None:
    """
    Predict emotion from the audio file.

    Args:
        config: Configuration object with model settings.
        audio_path (str): Path to the audio file.
        model: Pre-trained model used for emotion prediction.

    Returns:
        tuple: Predicted emotion and probability distribution.
    """
    if config.feature_method == 'o':
        of.get_data(config, audio_path, train=False)
        test_feature = of.load_feature(config, train=False)
    elif config.feature_method == 'l':
        test_feature = lf.get_data(config, audio_path, train=False)

    result = model.predict(test_feature)
    result_prob = model.predict_proba(test_feature)
    
    print(f"Recognition: {config.class_labels[int(result)]}")
    print(f"Probability: {result_prob}")
    
    return config.class_labels[int(result)], result_prob


def record_and_save(dirpath, duration=5, sample_rate=48000):
    """
    Records audio, segments speech, and predicts emotion for each segment.

    Args:
        dirpath (str): Directory path to save audio files.
        duration (int): Duration of the recording in seconds.
        sample_rate (int): Sample rate for audio recording.
    """
    config = utils.parse_opt()
    model = models.load(config)
    recording = True
    i = 1

    while recording:
        # Record audio
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype=np.int16)
        sd.wait()

        # Save audio to file
        filepath = f"{i}.wav"
        save_path = os.path.join(dirpath, filepath)

        if len(audio_data.shape) > 1 and audio_data.shape[1] == 2:
            audio_data = np.mean(audio_data, axis=1, dtype=np.int16)
        
        wav.write(save_path, sample_rate, audio_data)
        i += 1

        # Segment the audio
        seg = Segmenter()
        segmentation = seg(save_path)
        print(segmentation)

        audio_path = os.path.join(dirpath, "temp")
        j = 0
        os.makedirs(audio_path, exist_ok=True)

        # Process speech segments
        for label, start, stop in segmentation:
            if label == 'speech':
                file_path = f"{j}.wav"
                final_path = os.path.join(audio_path, file_path)
                audio = AudioSegment.from_wav(save_path)
                audio = audio[int(start * 1000):int(stop * 1000)]
                audio.export(final_path, format="wav")
                j += 1

        # Predict emotion for each segment
        for filename in os.listdir(audio_path):
            if filename.endswith(".wav"):
                input_path = os.path.join(audio_path, filename)
                audio = AudioSegment.from_file(input_path, format="wav")
                total_duration = len(audio) / 1000

                predicted_class, scores = predict(config, input_path, model)

                # Log predictions
                log_file_path = os.path.join(dirpath, 'logs', 'log.txt')
                os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
                with open(log_file_path, 'a') as log_file:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    log_file.write(f"{timestamp} - Duration: {total_duration} - Prediction: {predicted_class} - Scores: {scores}\n")


if __name__ == '__main__':
    record_and_save("/home/kasra/speech_emotion_recognition/attachments/temp/")
