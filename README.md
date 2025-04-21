# task3_AI_Chatbot

Company Name : CodTech IT Solutions Pvt. Ltd.  
Name : Vinay Mahendra Wankhade  
Intern ID : CT12UCB  
Domain : Python Programming  
Duration : 8 Weeks  
Mentor : Neela Santosh

# Description

# NLP-Powered Chatbot

## Introduction

This project implements a Natural Language Processing (NLP) chatbot using Python. The chatbot, named Milo, can understand user queries, identify intents, and provide relevant responses using a combination of predefined answers and a knowledge base with similarity matching. This implementation demonstrates fundamental NLP concepts including text preprocessing, intent recognition, and response generation using libraries such as NLTK and scikit-learn.

## Features

- **Natural Language Understanding**: Processes and understands user input using tokenization and lemmatization
- **Intent Recognition**: Identifies user intentions (greetings, questions, goodbyes, etc.)
- **Knowledge Base**: Contains information that can be retrieved based on query relevance
- **Learning Capability**: Can be taught new information to expand its knowledge base
- **TF-IDF Vectorization**: Uses Term Frequency-Inverse Document Frequency for text similarity computation
- **Simple Conversation Management**: Maintains a basic dialog flow with appropriate responses

## Prerequisites

Before running the chatbot, ensure you have the following installed:
- Python 3.6 or higher
- pip (Python package installer)

## Required Libraries

The chatbot relies on the following Python libraries:
- NLTK (Natural Language Toolkit)
- scikit-learn
- NumPy
- re (Regular Expressions library - included in standard Python)

## Installation

1. Clone the repository or download the source code:
```bash
git clone https://github.com/yourusername/nlp-chatbot.git
cd nlp-chatbot
```

2. Set up a virtual environment (recommended):
```bash
python -m venv .venv
```

3. Activate the virtual environment:
   - Windows:
   ```bash
   .venv\Scripts\activate
   ```
   - Linux/Mac:
   ```bash
   source .venv/bin/activate
   ```

4. Install the required dependencies:
```bash
pip install nltk scikit-learn numpy
```

5. Download necessary NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## Project Structure

```
nlp-chatbot/
├── chatbot.py          # Main chatbot implementation
├── requirements.txt    # List of required packages
└── README.md           # This documentation file
```

## How It Works

### The SimpleChatbot Class

The core of the project is the `SimpleChatbot` class, which handles all the NLP processing and response generation:

#### Initialization
During initialization, the chatbot:
- Sets up a lemmatizer for word normalization
- Defines response templates for various intents
- Creates a knowledge base with information about chatbots, NLP, and related topics
- Initializes the TF-IDF vectorizer for text comparison

#### Text Preprocessing
The `preprocess` method:
- Converts text to lowercase
- Removes punctuation
- Splits text into tokens
- Lemmatizes tokens to their base form

#### Intent Recognition
The `identify_intent` method:
- Analyzes preprocessed tokens from user input
- Matches patterns to determine user intent (greeting, farewell, question, etc.)
- Returns the identified intent category

#### Response Generation
The chatbot generates responses through:
- The `get_response` method - selects appropriate response type based on intent
- The `get_response_from_knowledge_base` method - finds relevant information using cosine similarity with TF-IDF vectors

#### Knowledge Base Expansion
The `expand_knowledge_base` method allows the chatbot to learn new information provided by users, enhancing its capabilities over time.

### The Main Function

The `main` function creates the conversation loop:
1. Initializes the chatbot
2. Greets the user
3. Enters a loop that:
   - Takes user input
   - Generates and displays responses
   - Offers to learn new information when it can't provide a good answer
4. Exits when the user types 'quit', 'exit', or 'bye'

## Technical Details

### NLP Techniques Used

1. **Tokenization**: Breaking text into individual words
2. **Lemmatization**: Reducing words to their base form
3. **TF-IDF Vectorization**: Converting text to numerical vectors
4. **Cosine Similarity**: Measuring text similarity

### Limitations

- The chatbot has a limited knowledge base
- Pattern matching for intent recognition is basic
- No contextual awareness across multiple exchanges
- Simple similarity matching may not capture complex questions
- No support for spelling correction or advanced NLP features

## Future Enhancements

The current implementation can be extended in several ways:
1. **Improve Intent Recognition**: Implement machine learning-based intent classification
2. **Add Named Entity Recognition**: Identify specific entities in user queries
3. **Implement Context Management**: Remember previous exchanges for more coherent conversations
4. **Integrate External APIs**: Connect to weather, news, or other services
5. **Add Sentiment Analysis**: Detect user emotions and adapt responses
6. **Implement More Advanced NLP Models**: Use transformer-based models like BERT or GPT
7. **Create a Web or GUI Interface**: Make the chatbot more accessible and user-friendly
8. **Add Database Integration**: Store conversations and expand knowledge permanently

## Usage

To run the chatbot:

```bash
python chatbot.py
```

Sample interaction:

```
Initializing chatbot...
Chatbot: Hello! I'm Milo, your NLP-powered assistant. Type 'quit' to exit.
You: hi
Chatbot: Hi there! What can I do for you?
You: what is nlp?
Chatbot: Natural Language Processing (NLP) is a field of AI that helps computers understand human language.
You: thanks
Chatbot: Happy to help!
You: bye
Chatbot: See you later!
```

## Conclusion

This NLP chatbot project demonstrates the fundamental concepts and techniques used in building conversational agents. While relatively simple, it provides a solid foundation for understanding how chatbots process and respond to natural language. The modular design allows for easy extension and enhancement, making it an excellent starting point for more advanced NLP applications.

By combining basic NLP techniques with a flexible architecture, this chatbot can serve as a learning tool for those interested in natural language processing and conversational AI development.
