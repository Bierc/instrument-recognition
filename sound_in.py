import librosa
import sounddevice as sd
import wave
import numpy as np


def sound_in():
    # Configurações de áudio
    fs = 44100  # Frequência de amostragem
    duration = 5  # Duração da gravação em segundos

    output_file = 'teste.wav'

    # Captura de áudio em tempo real
    audio_data = sd.rec(int(fs * duration), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("gravando...")
    audio_data2 = audio_data.astype(np.int16)

    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio_data2.tobytes())