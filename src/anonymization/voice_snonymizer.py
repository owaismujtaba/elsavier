import os
import librosa
import soundfile as sf
import parselmouth
import numpy as np


class VoiceAnonymizerPipeline:
    """
    A pipeline-style class for voice anonymization using pitch and formant shifting.
    """

    def __init__(self, pitch_steps=4, formant_ratio=1.2, target_sr=16000):
        """
        Initializes the VoiceAnonymizerPipeline.

        Args:
            pitch_steps (int): Number of semitones for pitch shift.
            formant_ratio (float): Scaling factor for formants.
            target_sr (int): Target sample rate for processing.
        """
        self.pitch_steps = pitch_steps
        self.formant_ratio = formant_ratio
        self.target_sr = target_sr

    def load(self, file_path):
        """
        Loads audio and resamples to target sample rate.

        Args:
            file_path (str): Path to input audio file.

        Returns:
            tuple: (audio, sample_rate)
        """
        audio, sr = librosa.load(file_path, sr=self.target_sr)
        return audio, sr

    def pitch_shift(self, audio, sr):
        """
        Applies pitch shifting.

        Args:
            audio (np.ndarray): Input audio.
            sr (int): Sample rate.

        Returns:
            np.ndarray: Pitch-shifted audio.
        """
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=self.pitch_steps)

    def formant_shift(self, audio, sr):
        """
        Applies formant shifting using Parselmouth (Praat).

        Args:
            audio (np.ndarray): Input audio.
            sr (int): Sample rate.

        Returns:
            np.ndarray: Formant-shifted audio.
        """
        snd = parselmouth.Sound(audio, sampling_frequency=sr)
        manipulation = parselmouth.praat.call(
            snd, "To Manipulation", 0.01, 75, 600
        )
        parselmouth.praat.call(
            manipulation, "Shift formants", 1, self.formant_ratio
        )
        resynth = parselmouth.praat.call(
            manipulation, "Get resynthesis (overlap-add)"
        )
        return resynth.values[0]

    def transform(self, audio, sr):
        """
        Applies pitch and formant shifting.

        Args:
            audio (np.ndarray): Input audio signal.
            sr (int): Sample rate.

        Returns:
            np.ndarray: Transformed anonymized audio.
        """
        audio_shifted = self.pitch_shift(audio, sr)
        audio_anonymized = self.formant_shift(audio_shifted, sr)
        return audio_anonymized

    def fit(self, X=None, y=None):
        """
        Placeholder for compatibility with scikit-learn pipelines.
        """
        return self

    def fit_transform(self, file_path):
        """
        Loads and transforms audio from a file path.

        Args:
            file_path (str): Path to the input audio file.

        Returns:
            np.ndarray: Anonymized audio signal.
        """
        audio, sr = self.load(file_path)
        return self.transform(audio, sr)

    def save(self, audio, sr, output_path):
        """
        Saves audio to a file.

        Args:
            audio (np.ndarray): Audio to save.
            sr (int): Sample rate.
            output_path (str): Destination file path.
        """
        sf.write(output_path, audio, sr)
        print(f"Anonymized audio saved to '{output_path}'")
