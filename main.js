const tf = require('@tensorflow/tfjs');
const natural = require('natural');
const readlineSync = require('readline-sync');
const fs = require('fs');
const path = require('path');

tf.setBackend('cpu');
tf.enableProdMode();

// Constants
const SEQUENCE_LENGTH = 10;
const EMBEDDING_DIM = 64;
const VOCAB_SIZE = 5000;
const MODEL_PATH = path.join(__dirname, 'model');
const MEMORY_PATH = path.join(__dirname, 'memory.json');

// Text preprocessing utilities
const tokenizer = new natural.WordTokenizer();
const stemmer = natural.PorterStemmer;

class NeuralTextModel {
  constructor() {
    this.model = null;
    this.wordIndex = {};
    this.indexWord = {};
    this.memory = [];
    this.loadMemory();
    
    // Initialize vocabulary immediately from memory
    if (this.memory.length > 0) {
      this.initializeVocabulary();
    }
  }

  // Load previous conversations if they exist
  loadMemory() {
    try {
      if (fs.existsSync(MEMORY_PATH)) {
        this.memory = JSON.parse(fs.readFileSync(MEMORY_PATH, 'utf-8'));
        console.log('Loaded previous learning data.');
      }
    } catch (error) {
      console.log('No previous learning data found, starting fresh.');
    }
  }

  // Initialize vocabulary from all memory entries
  initializeVocabulary() {
    console.log('Building vocabulary from memory...');
    this.wordIndex = {};
    this.indexWord = {};
    
    // First pass: collect all unique words
    const allWords = new Set();
    this.memory.forEach(item => {
      const text = item.input + ' ' + item.output;
      const tokens = tokenizer.tokenize(text.toLowerCase());
      const stems = tokens.map(token => stemmer.stem(token));
      stems.forEach(word => allWords.add(word));
    });
    
    // Second pass: assign indices
    let index = 1; // Start from 1, reserve 0 for unknown
    allWords.forEach(word => {
      this.wordIndex[word] = index;
      this.indexWord[index] = word;
      index++;
    });
    
    console.log(`Vocabulary built with ${allWords.size} unique words`);
  }

  // Save conversations to memory
  saveMemory() {
    fs.writeFileSync(MEMORY_PATH, JSON.stringify(this.memory));
  }

  // Preprocess text into sequences
  preprocessText(text) {
    const tokens = tokenizer.tokenize(text.toLowerCase());
    const stems = tokens.map(token => stemmer.stem(token));
    
    // Don't update vocabulary during preprocessing, just use the existing one
    return stems.map(word => this.wordIndex[word] || 0);
  }

  // Create sequences for training
  createSequences() {
    // Make sure vocabulary is initialized
    if (Object.keys(this.wordIndex).length === 0) {
      this.initializeVocabulary();
    }
    
    const sequences = [];
    const nextWords = [];
    const vocabSize = Object.keys(this.wordIndex).length;
    
    if (vocabSize === 0) {
      throw new Error('Vocabulary is empty. Cannot create training sequences.');
    }
    
    console.log(`Working with vocabulary size: ${vocabSize}`);
    
    this.memory.forEach(item => {
      const indices = this.preprocessText(item.input + " " + item.output);
      
      // Only create sequences if there are enough tokens
      if (indices.length > SEQUENCE_LENGTH) {
        for (let i = 0; i < indices.length - SEQUENCE_LENGTH; i++) {
          const sequence = indices.slice(i, i + SEQUENCE_LENGTH);
          const nextWord = indices[i + SEQUENCE_LENGTH];
          
          // Make sure sequences and next words contain valid indices
          if (!sequence.includes(undefined) && nextWord !== undefined && nextWord > 0) {
            sequences.push(sequence);
            nextWords.push(nextWord);
          }
        }
      }
    });
    
    // Check if we have valid training data
    if (sequences.length === 0) {
      throw new Error('Not enough valid sequences to train on. Add more varied text examples.');
    }
    
    console.log(`Created ${sequences.length} valid training sequences`);
    
    return {
      sequences: tf.tensor2d(sequences),
      // Change to float32 to fix the type error
      nextWords: tf.tensor1d(nextWords, 'float32')
    };
  }

  // Build the model
  buildModel() {
    // Ensure vocabulary is initialized
    if (Object.keys(this.wordIndex).length === 0) {
      this.initializeVocabulary();
    }
    
    const vocabSize = Object.keys(this.wordIndex).length + 1; // +1 for unknown token (0)
    console.log(`Building model with vocabulary size: ${vocabSize}`);
    
    this.model = tf.sequential();
    
    // Input layer with embedding
    this.model.add(tf.layers.embedding({
      inputDim: vocabSize,
      outputDim: EMBEDDING_DIM,
      inputLength: SEQUENCE_LENGTH
    }));
    
    // LSTM layer
    this.model.add(tf.layers.lstm({
      units: 128,
      returnSequences: true
    }));
    
    // Another LSTM layer
    this.model.add(tf.layers.lstm({
      units: 128
    }));
    
    // Dense layer with dropout
    this.model.add(tf.layers.dropout(0.2));
    this.model.add(tf.layers.dense({
      units: 64,
      activation: 'relu'
    }));
    
    // Output layer
    this.model.add(tf.layers.dense({
      units: vocabSize,
      activation: 'softmax'
    }));
    
    // Compile the model - change to sparseCategoricalCrossentropy
    this.model.compile({
      optimizer: 'adam',
      loss: 'sparseCategoricalCrossentropy',
      metrics: ['accuracy']
    });
    
    return this.model;
  }

  // Train the model
  async train() {
    if (this.memory.length < 2) {
      console.log('Not enough data to train the model yet.');
      return;
    }
    
    try {
      console.log('Preparing for training...');
      
      // Reset and rebuild vocabulary to ensure consistency
      this.initializeVocabulary();
      
      // Clear existing model to build fresh
      this.model = null;
      this.buildModel();
      
      const { sequences, nextWords } = this.createSequences();
      
      console.log('Training model...');
      console.log(`Training on ${sequences.shape[0]} sequences`);

      // Safeguard against empty tensors
      if (sequences.shape[0] === 0) {
        console.log('No valid sequences to train on, add more diverse text examples');
        return;
      }
      
      // Use a smaller batch size for better stability
      await this.model.fit(sequences, nextWords, {
        epochs: 50,
        batchSize: 16, // Reduced from 32 to improve stability
        shuffle: true,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            if (epoch % 10 === 0) {
              console.log(`Epoch ${epoch}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`);
            }
          }
        }
      });
      
      console.log('Training complete!');
      
      try {
        await this.saveModel();
        console.log('Model saved successfully.');
      } catch (error) {
        console.error('Failed to save model:', error);
      }
    } catch (error) {
      console.error('Training error:', error.message);
      console.error('Try running with a smaller batch size or adding more data.');
      // Print more details if available
      if (error.stack) {
        console.error('Stack trace:', error.stack.split('\n').slice(0, 3).join('\n'));
      }
    }
  }

  // Apply reinforcement learning
  reinforcementUpdate(input, output, feedback) {
    // Store interaction with feedback score
    this.memory.push({
      input,
      output,
      feedback: parseInt(feedback)
    });
    
    // Update vocabulary with new words from this interaction
    const text = input + " " + output;
    const tokens = tokenizer.tokenize(text.toLowerCase());
    const stems = tokens.map(token => stemmer.stem(token));
    
    let vocabUpdated = false;
    stems.forEach(word => {
      if (!this.wordIndex[word]) {
        const newIndex = Object.keys(this.wordIndex).length + 1;
        this.wordIndex[word] = newIndex;
        this.indexWord[newIndex] = word;
        vocabUpdated = true;
      }
    });
    
    if (vocabUpdated) {
      console.log(`Vocabulary updated. Now contains ${Object.keys(this.wordIndex).length} words.`);
    }
    
    // Save updated memory
    this.saveMemory();
  }

  // Generate text based on input
  async generateText(inputText) {
    if (!this.model || Object.keys(this.wordIndex).length < 5) {
      // Not enough data or model not trained
      if (this.memory.length >= 5) {
        return "I have enough data to learn, but I need to be trained first. Type 'train' to start the learning process.";
      } else {
        return "I'm still learning. Please teach me more!";
      }
    }
    
    // Preprocess input
    const inputSequence = this.preprocessText(inputText);
    
    // If sequence is too short, provide a simple response
    if (inputSequence.length < SEQUENCE_LENGTH) {
      return "I need more context to generate a meaningful response.";
    }
    
    // Get the last SEQUENCE_LENGTH tokens
    const sequence = inputSequence.slice(-SEQUENCE_LENGTH);
    
    // Generate words one by one
    let outputText = [];
    let currentSequence = [...sequence];
    
    // Generate up to 20 words
    for (let i = 0; i < 20; i++) {
      // Predict next word
      const prediction = this.model.predict(tf.tensor2d([currentSequence]));
      const predictedIndex = tf.argMax(prediction, 1).dataSync()[0];
      
      // Stop if we hit end of sequence or unknown token
      if (predictedIndex === 0 || !this.indexWord[predictedIndex]) break;
      
      // Add predicted word to output
      outputText.push(this.indexWord[predictedIndex]);
      
      // Update sequence for next prediction (remove first word, add predicted word)
      currentSequence.shift();
      currentSequence.push(predictedIndex);
    }
    
    return outputText.join(' ');
  }

  // Save model to disk
  async saveModel() {
    if (!this.model) return;
    
    // For the pure JS version, we need to use the IOHandler format
    await this.model.save('file://' + MODEL_PATH);
    
    // Save vocabulary
    fs.writeFileSync(
      path.join(__dirname, 'vocab.json'), 
      JSON.stringify({ wordIndex: this.wordIndex, indexWord: this.indexWord })
    );
    
    // Also save a readable version of the vocabulary for debugging
    const readableVocab = {};
    Object.keys(this.wordIndex).forEach(word => {
      readableVocab[word] = this.wordIndex[word];
    });
    
    fs.writeFileSync(
      path.join(__dirname, 'vocab_readable.json'), 
      JSON.stringify(readableVocab, null, 2)
    );
  }

  // Load model from disk
  async loadModel() {
    try {
      if (fs.existsSync(path.join(MODEL_PATH, 'model.json'))) {
        // Load with the correct format for pure JS version
        this.model = await tf.loadLayersModel('file://' + path.join(MODEL_PATH, 'model.json'));
        console.log('Model loaded successfully.');
        
        // Load vocabulary
        const vocabData = JSON.parse(fs.readFileSync(path.join(__dirname, 'vocab.json'), 'utf-8'));
        this.wordIndex = vocabData.wordIndex;
        this.indexWord = vocabData.indexWord;
        return true;
      }
    } catch (error) {
      console.error('Failed to load model:', error);
    }
    return false;
  }
}

// Main function
async function main() {
  console.log('Welcome to Neru - Neural Text Learning System');
  console.log('============================================');
  
  const neru = new NeuralTextModel();
  
  // Try to load existing model
  const modelLoaded = await neru.loadModel();
  if (modelLoaded) {
    console.log('Existing model loaded and ready to use.');
  } else {
    console.log("No trained model found. I'll learn from our conversation.");
  }
  
  let continueChat = true;
  
  while (continueChat) {
    console.log('\n(Type "exit" to quit, "train" to train the model)\n');
    const userInput = readlineSync.question('You: ');
    
    // Handle commands more explicitly
    if (userInput.toLowerCase() === 'exit') {
      continueChat = false;
      continue;
    }
    
    if (userInput.toLowerCase() === 'train') {
      await neru.train();
      continue;
    }
    
    // Generate response
    const response = await neru.generateText(userInput);
    console.log('Neru: ' + response);
    
    // Get feedback for reinforcement learning (1-5 scale)
    const feedback = readlineSync.question('Rate response (1-5, 5 being best): ');
    if (/^[1-5]$/.test(feedback)) {
      neru.reinforcementUpdate(userInput, response, feedback);
      if (feedback >= 4) {
        console.log("Thanks! I'll remember what worked well.");
      } else if (feedback <= 2) {
        console.log("I'll try to improve my responses.");
      }
    } else {
      console.log("Please provide a rating between 1 and 5 to help me learn.");
    }
  }
  
  console.log('Goodbye! Saving learning progress...');
  neru.saveMemory();
}

// Start the application
main().catch(error => console.error('Error:', error));
