import nltk
import random
import string
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download necessary NLTK data (with the correct resources)
# uncomment for use one time only
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')


class SimpleChatbot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        # Pre-defined responses for specific topics
        self.responses = {
            'greeting': [
                "Hello! How can I help you today?",
                "Hi there! What can I do for you?",
                "Hey! What would you like to know?"
            ],
            'goodbye': [
                "Goodbye! Have a great day!",
                "See you later!",
                "Bye! Feel free to chat again if you need help."
            ],
            'thanks': [
                "You're welcome!",
                "Happy to help!",
                "No problem at all!"
            ],
            'name': [
                "I'm Milo, your friendly NLP assistant.",
                "You can call me Milo!",
                "I'm an NLP-powered chatbot here to help you."
            ],
            'help': [
                "I can answer questions, provide information, or just chat. What do you need?",
                "I'm here to assist you with information and conversation. What can I help with?",
                "Ask me anything, and I'll do my best to help!"
            ],
            'fallback': [
                "I'm not sure I understand. Could you rephrase that?",
                "I don't have the answer to that yet.",
                "I'm still learning and don't have information about that.",
                "That's an interesting question, but I don't have a good answer right now."
            ]
        }

        # Knowledge base for the chatbot
        self.knowledge_base = [
            "A chatbot is a computer program designed to simulate conversation with human users.",
            "Natural Language Processing (NLP) is a field of AI that helps computers understand human language.",
            "NLTK is a popular Python library for working with human language data.",
            "Python is a programming language often used for AI and machine learning.",
            "Machine learning is a method of teaching computers to learn from data.",
            "AI stands for Artificial Intelligence.",
            "Chatbots use various techniques to understand and respond to user queries.",
            "The Turing test is a test of a machine's ability to exhibit intelligent behavior.",
            "Named Entity Recognition is a process where an algorithm identifies named entities in text.",
            "Word embeddings are a type of word representation in NLP."
        ]

        # Process and vectorize the knowledge base
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.knowledge_base)

    def preprocess(self, text):
        """Preprocess text by tokenizing, removing punctuation, and lemmatizing."""
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)

        # Use simple tokenization instead of word_tokenize to avoid punkt_tab issue
        tokens = text.split()
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        return lemmatized_tokens

    def identify_intent(self, user_input):
        """Identify the user's intent based on their input."""
        tokens = set(self.preprocess(user_input))

        # Check for matches with predefined intents
        if any(word in tokens for word in ['hi', 'hello', 'hey']):
            return 'greeting'
        elif any(word in tokens for word in ['bye', 'goodbye', 'exit', 'quit']):
            return 'goodbye'
        elif any(word in tokens for word in ['thanks', 'thank']):
            return 'thanks'
        elif any(word in tokens for word in ['name', 'who']):
            return 'name'
        elif any(word in tokens for word in ['help', 'assist']):
            return 'help'
        else:
            return 'query'  # Default to treating as a knowledge query

    def get_response_from_knowledge_base(self, user_input):
        """Get a response from the knowledge base using TF-IDF and cosine similarity."""
        user_vector = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vector, self.tfidf_matrix)[0]

        # If the best match is too low, use fallback
        if np.max(similarities) < 0.2:
            return random.choice(self.responses['fallback'])

        # Get the most similar document
        best_match_index = np.argmax(similarities)
        return self.knowledge_base[best_match_index]

    def get_response(self, user_input):
        """Generate a response based on the user's input."""
        intent = self.identify_intent(user_input)  # Fixed function name here

        # Handle predefined intents
        if intent in self.responses and intent != 'fallback':
            return random.choice(self.responses[intent])

        # For queries, try to find a relevant response from the knowledge base
        return self.get_response_from_knowledge_base(user_input)

    def expand_knowledge_base(self, new_information):
        """Add new information to the knowledge base."""
        self.knowledge_base.append(new_information)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.knowledge_base)


def main():
    print("Initializing chatbot...")
    chatbot = SimpleChatbot()
    print("Chatbot: Hello! I'm Milo, your NLP-powered assistant. Type 'quit' to exit.")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Milo:", random.choice(chatbot.responses['goodbye']))
            break

        response = chatbot.get_response(user_input)
        print("Milo:", response)

        # Option to teach the chatbot (simplified for this example)
        if response in chatbot.responses['fallback']:
            teach = input("Would you like to teach me a better response? (yes/no): ")
            if teach.lower() == 'yes':
                new_info = input("Please provide the information: ")
                chatbot.expand_knowledge_base(new_info)
                print("Milo: Thank you! I've learned something new.")


if __name__ == "__main__":
    main()