#transformer v0.264
import numpy as np
import pickle
import re

# Constants
KB_MEMORY_UNCOMPRESSED = 7000
hidden_dim = 7000
learning_rate = 0.01
epochs = 5
n = 4
generate_length = 40  # Number of n-grams to generate sequentially
temperature = 0.7  # Temperature for softmax

def dict_to_vector(vector_dict, vocab):
    """Convert a dictionary of n-grams into a vector based on the vocabulary order."""
    vector = np.zeros(len(vocab))
    for i, ngram in enumerate(vocab):
        vector[i] = vector_dict.get(ngram, 0)
    return vector

def softmax(x, temperature=1.0):
    """Softmax function with temperature."""
    x = np.array(x) / temperature
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def forward_pass(X, W1, b1, W2, b2, W3, b3):
    Z1 = np.tensordot(X, W1, axes=([0], [0])) + b1
    A1 = np.tanh(Z1)
    
    Z2 = np.tensordot(A1, W2, axes=([0], [0])) + b2
    A2 = np.tanh(Z2)
    
    Z3 = np.tensordot(A2, W3, axes=([0], [0])) + b3
    A3 = softmax(Z3, temperature)
    
    return A3, A2, A1
    
def adjust_temperature(length, initial_temp=0.7, growth_factor=0.01):
    """Adjust temperature dynamically based on the length of generated text."""
    return initial_temp + growth_factor * length

def dynamic_sizemic_weighting(ngram, vocab, ngram_frequencies, length):
    """Apply dynamic sizemic weighting to n-grams based on frequency and length."""
    base_weight = 1 / (ngram_frequencies.get(ngram, 1) + 1e-9)
    length_factor = np.log(length + 1)  # Growth factor based on the length of the generated text
    return base_weight * length_factor

def vectorize_ngram_with_frequencies(text, vocab, n, ngram_frequencies, pad_token="<pad>"):
    """Vectorizes text into n-grams with sizemic weighting based on frequency and length."""
    words = text.split()
    ngram_counts = {ngram: 0 for ngram in vocab}
    
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i+n])
        if ngram in ngram_counts:
            ngram_counts[ngram] += 1
    
    return ngram_counts
    
def base_vectorize_ngram_with_frequencies(text, vocab, n, ngram_frequencies, pad_token="<pad>"):
    """Vectorizes text into n-grams with sizemic weighting based on frequency and length."""
    words = vocab
    ngram_counts = {ngram: 0 for ngram in vocab}
    
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i+n])
        if ngram in ngram_counts:
            ngram_counts[ngram] += 1
    
    return ngram_counts
    
def compute_ngram_frequencies(text, n):
    """Compute the frequency of each n-gram in the given text."""
    words = text.split()
    ngram_counts = {}
    
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i+n])
        if ngram in ngram_counts:
            ngram_counts[ngram] += 1
        else:
            ngram_counts[ngram] = 1
    
    return ngram_counts
    
def chat_with_neural_network(model_params, vocab, user_input, generate_length, n=3):
    W1, b1, W2, b2, W3, b3, ngram_frequencies = model_params
    vocab_size = len(vocab)
    output = []
    current_input = user_input
    
    for length in range(generate_length):


        input_dict = vectorize_ngram_with_frequencies(current_input, vocab, n, ngram_frequencies)
        input_vector = dict_to_vector(input_dict, vocab)  # Use vector instead of scalar
        
        # Forward pass with 3D tensors
        A3, A2, A1 = forward_pass(input_vector, W1, b1, W2, b2, W3, b3)
        
        # Adjust temperature dynamically
        temp = adjust_temperature(length)
        
        probabilities = softmax(A3, temperature=temp)
        
        # Sample from the distribution
        predicted_idx = np.random.choice(range(len(probabilities)), p=probabilities)
        
        ngram_word = vocab[predicted_idx] if predicted_idx < len(vocab) else tuple([''])
        
        output.append(' '.join(ngram_word))
        
        current_input = ' '.join(output)
    
    return ' '.join(output)
    
def custom_normalization(x):
    """Normalize the input vector to probabilities."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)
    
def train_model(vocab, text_data, n, learning_rate, epochs):
    # Compute n-gram frequencies from the training data
    ngram_frequencies = compute_ngram_frequencies(text_data, n)
    
    input_dict = base_vectorize_ngram_with_frequencies(text_data, vocab, n, ngram_frequencies)
    input_vector = dict_to_vector(input_dict, vocab)  # Use vector instead of scalar

    target_dict = vectorize_ngram_with_frequencies(text_data, vocab, n, input_dict)
    target_vector = dict_to_vector(target_dict, vocab)  # Use vector instead of scalar

    input_dim = len(vocab)  # Vector context
    output_dim = len(vocab)

    # Initialize weights for 3 layers
    W1 = np.random.randn(input_dim, hidden_dim) * 0.01
    b1 = input_vector # hack requires equal hidden_dim and KB_UNCOMPRESSED
    W2 = np.random.randn(hidden_dim, hidden_dim) * 0.01  # Added second layer weights
    b2 = target_vector # hack requires equal hidden_dim and KB_UNCOMPRESSED
    W3 = np.random.randn(hidden_dim, output_dim) * 0.01  # Output layer weights
    b3 = target_vector
    for epoch in range(epochs):
        # Forward pass with 3 layers
        A3, A2, A1 = forward_pass(input_vector, W1, b1, W2, b2, W3, b3)
        
        # Backpropagation (3 layers)
        dA3 = A3 - input_vector
        dZ3 = dA3  # Gradient for softmax
        dW3 = np.outer(A2, dZ3)
        db3 = dZ3

        dA2 = np.dot(W3, dZ3) * (1 - A2 ** 2)  # Gradient w.r.t A2
        dW2 = np.outer(A1, dA2)
        db2 = dA2

        dA1 = np.dot(W2, dA2) * (1 - A1 ** 2)  # Gradient w.r.t A1
        dW1 = np.outer(target_vector, dA1)
        db1 = dA1
        b3 = db1 # hack requires equal hidden_dim and KB_UNCOMPRESSED

        # Update parameters
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3

       
        print(f"Epoch {epoch}")

    return W1, b1, W2, b2, W3, b3, ngram_frequencies

def save_model(model_params, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model_params, f)

def load_model(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def build_vocabulary(text_data, n):
    """Build a vocabulary of n-grams from text data, excluding symbols and numbers."""
    
    # Remove symbols and numbers using regex
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text_data)
    
    # Split text into words
    words = cleaned_text.split()
    
    # Filter out one-character words
    words = [word for word in words if len(word) > 1 or word == "a" or word == "i"]
    
    # Generate n-grams
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    
    # Create a list of unique n-grams
    vocab = list(set(ngrams))
    
    return vocab

def main():
    with open("test.txt", encoding="UTF-8") as f:
        text_data = f.read()

    vocab = build_vocabulary(text_data, n)[:KB_MEMORY_UNCOMPRESSED]

    choice = input("Save new model/Load old model? [s/l]: ")
    
    if choice == 's':
        model_params = train_model(vocab, text_data, n, learning_rate, epochs)
        save_model(model_params, 'model.pkl')
        print("Model saved.")
    elif choice == 'l':
        model_params = load_model('model.pkl')
        print("Model loaded.")
    
    while True:
        user_input = input("Enter text: ")
        
        # Generate n-grams sequentially
        ngram_predictions = chat_with_neural_network(model_params, vocab, user_input, generate_length, n=n).lower()

        # Print the top 10 longest predictions
        print("Generated n-grams:", ngram_predictions)

if __name__ == '__main__':
    main()