# Sign Language Assistant

A real-time sign language interpreter and context-aware assistant that helps bridge communication gaps between sign language users and others. Built during the Entrepreneur First x UCL x Imperial Hackathon 2024.

## Winning submission to UCL x ICL Hack by Entrepreneur First

## Demo

_Note that we had to record to the screen using a phone as we couldn't screen record while using OBS to Open_CV_
[Demo](https://youtu.be/U1oAggfj_y8)

## Features

- **Real-time Sign Language Recognition**: Converts sign language gestures into text using computer vision
- **Intelligent Text Correction**: Uses GPT-4 to correct and improve the text output from sign language recognition
- **Text-to-Speech**: Converts corrected text into natural-sounding speech
- **Context-Aware Vision**: Provides environmental descriptions when requested
- **Multi-Modal Input**: Supports both sign language and voice commands

## Technologies Used

- **Computer Vision**: OpenCV, MediaPipe for hand gesture recognition
- **Machine Learning**: Custom trained Random Forest classifier for sign language detection
- **AI/Language Models**: OpenAI GPT-4 for text correction and context understanding
- **Speech Processing**: Speech Recognition, OpenAI TTS for voice output
- **Audio Processing**: PyGame for audio playback
- **Data Processing**: NumPy for numerical operations

## Setup

1. Make a Facebook account and stream video from glasses.

2. Use OBS to create a virtual monitor that contains the Facebook stream.
   - This will be piped to OpenCV.

4. Clone the repository:
```bash
   git clone https://github.com/yourusername/sign-language-assistant.git
   cd sign-language-assistant
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Set up environment variables:
   Create a `.env` file with your OpenAI API key:
   ```plaintext
   OPENAI_API_KEY=your_api_key_here
   ```

6. Run the application:
   ```bash
   python src/context_audio.py
   ```

## Usage

1. **Sign Language Recognition**:
   - Stand in front of the camera
   - Perform sign language gestures
   - The system will recognize, correct, and speak the interpreted text

2. **Voice Commands**:
   - Say "describe surroundings" to get a description of the environment
   - Say "sign language" to activate sign language recognition mode

3. **Environment Description**:
   - The system will analyze the surroundings and provide audio descriptions
   - Useful for understanding the immediate environment

## Project Structure

```
hack_ucl_imperial/
├── src/
│   ├── context_audio.py      # Main application file
│   ├── voice_processor.py    # Voice processing and text correction
│   ├── classifier/           # Sign language classification models
│   └── data_gen/            # Data generation and processing
├── requirements.txt
└── README.md
```

## Made By

This project was developed by Darius Chitoroaga, Adrien Ropartz and Anton Zhulkovskiy during the Entrepreneur First x UCL x Imperial Hackathon 2024.
