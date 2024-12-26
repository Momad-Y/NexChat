# AI Chatbot Project

This project implements an AI-powered chatbot that utilizes large language models (LLMs) for question answering and summarization, along with computer vision capabilities for image captioning and visual question answering. The chatbot also supports audio input and output for a more interactive user experience.

## Project Structure

```
ai-chatbot-project
├── src
│   ├── chatbot.py          # Main chatbot class for user interactions and conversation management
│   ├── memory.py           # Memory management system for storing and retrieving past interactions
│   ├── nlp
│   │   ├── question_answering.py  # Functionality for answering questions using LLMs
│   │   ├── summarization.py       # Functionality for summarizing documents using LLMs
│   │   └── __init__.py            # Initializes the NLP module
│   ├── cv
│   │   ├── image_captioning.py     # Image captioning functionality using transformer models
│   │   ├── visual_question_answering.py  # Visual question answering functionality using transformer models
│   │   └── __init__.py             # Initializes the computer vision module
│   ├── audio
│   │   ├── audio_input.py           # Handles audio input for voice prompts
│   │   ├── audio_output.py          # Handles audio output for reading responses aloud
│   │   └── __init__.py              # Initializes the audio module
│   └── utils.py                     # Utility functions for file handling and data processing
├── requirements.txt                 # Lists project dependencies
├── README.md                        # Project documentation
└── .gitignore                       # Specifies files to ignore in version control
```

## Features

- **Question Answering**: Utilize open-source LLMs to answer questions based on various document types (PDF, CSV, Arxiv papers).
- **Summarization**: Summarize the contents of documents using LLMs.
- **Memory Management**: Maintain context during conversations by storing and retrieving past interactions.
- **Image Captioning**: Generate captions for images using transformer-based models.
- **Visual Question Answering**: Answer questions about images using transformer-based models.
- **Audio Input/Output**: Accept voice prompts and read responses aloud for an interactive experience.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd ai-chatbot-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the chatbot:
   ```
   python src/chatbot.py
   ```

## Usage Examples

- Ask the chatbot questions about uploaded documents.
- Summarize lengthy documents for quick insights.
- Interact with the chatbot using voice commands.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.