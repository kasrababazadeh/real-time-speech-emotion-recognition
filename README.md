# Real-Time Speech Emotion Recognition

This repository implements a **Real-Time Speech Emotion Recognition** system tailored for Persian language audio. The system records audio, removes non-speech segments such as noise and music, and predicts the emotional state of the speaker using a pre-trained LSTM model. The emotions are categorized into **positive**, **negative**, and **neutral** classes.

## Features
- **Real-time audio recording**: Records live audio from the microphone.
- **Speech segmentation**: Filters out noise and non-speech segments (like music) using a speech segmentation model.
- **Feature extraction**: Uses **Librosa** to preprocess the audio and extract relevant features.
- **LSTM-based emotion prediction**: The model is trained using an LSTM network to predict emotions from speech.
- **High accuracy**: Achieves over **90% accuracy** on a Persian speech dataset.
- **Emotion categories**:
  - **Positive**: Happy, Surprise
  - **Negative**: Sad, Angry, Fear
  - **Neutral**: Neutral

## Installation

### Prerequisites
1. Install the required Python packages:
The necessary packages include:

- `numpy`
- `pydub`
- `librosa`
- `sounddevice`
- `scipy`
- `inaSpeechSegmenter`
- Any additional dependencies required for feature extraction and model training.

### Install FFmpeg for audio processing:

- **On Linux**:
  ```bash
  sudo apt install ffmpeg
  ```
  - On Windows: Download and install FFmpeg from https://ffmpeg.org/download.html.

  ## Running the Program

### Clone the repository:

```bash
git clone https://github.com/kasrababazadeh/real-time-speech-emotion-recognition.git
cd speech-emotion-recognition
```
Set up the configuration for the feature extraction and model in utils.py.

### Start real-time recording, segmentation, and emotion prediction:
```bash
python main.py
```

## Model Details

- **Dataset**: This system is trained on a Persian speech dataset that includes real-life recordings of various emotions.
- **Preprocessing**: Uses **Librosa** to extract audio features such as Mel-frequency cepstral coefficients (MFCCs) for emotion classification.
- **Model Architecture**: The system uses a **Long Short-Term Memory (LSTM)** network to model the temporal dynamics of speech and predict emotions.
- **Segmentation**: The system employs **inaSpeechSegmenter** to segment the audio and extract only the speech parts, filtering out non-speech elements like background noise or music.

## File Structure

- `main.py`: Main script for real-time recording and emotion prediction.
- `models.py`: Contains functions to load the pre-trained LSTM model.
- `utils.py`: Utility functions for handling configuration, logging, etc.
- `attachments/`: Contains model files, segmenters, and other required assets.
  - `inaSpeechSegmenter/`: Includes speech segmentation tools to isolate speech from noise.
- `logs/`: Logs the emotion predictions with timestamps.

## Emotion Categories

The system classifies emotions into three categories:

- **Negative**:
  - Sad
  - Angry
  - Fear
- **Neutral**:
  - Neutral
- **Positive**:
  - Happy
  - Surprise

## Usage

### Record and Predict Emotion
The system records audio for a specified duration (default: 5 seconds), segments it into speech parts, and predicts emotions using the LSTM model. The results are logged with timestamps, emotion labels, and confidence scores.

### Log Example

Each prediction is logged with the following format:

```less
2024-10-08 14:35:21 - Duration: 3.5 - Prediction: Happy - Scores: [0.1, 0.7, 0.1, 0.1]
```

Where:

- **Duration**: The duration of the speech segment in seconds.
- **Prediction**: The predicted emotion.
- **Scores**: The confidence scores for each emotion class.

## Performance

The system achieves over **90% accuracy** on the Persian speech emotion dataset, distinguishing between positive, negative, and neutral emotions.

## Contributing

Contributions are welcome! If you find any issues or want to add new features, feel free to submit a Pull Request or open an Issue.

## License

Copyright 2024 Kasra Babazadeh-Mahalleh
For more details, see the [LICENSE](LICENSE) file.
