const AdvancedNetwork = require('./advancedNetwork');
const WordEmbeddings = require('./wordEmbeddings');

/**
 * Utility to enhance existing neural networks with advanced capabilities
 */
class NetworkEnhancer {
    /**
     * Create an enhancer
     * @param {Object} options - Configuration options
     */
    constructor(options = {}) {
        this.options = {
            embeddingDim: 50,
            useLSTM: true,
            hiddenSizes: [64, 32],
            dropout: 0.2,
            learningRate: 0.001,
            ...options
        };
    }
    
    /**
     * Convert a basic network to an advanced network
     * @param {NeuralNetwork} basicNetwork - Original basic network
     * @param {TextProcessor} textProcessor - Text processor
     * @param {WordEmbeddings} wordEmbeddingDb - Optional word embedding database
     * @returns {AdvancedNetwork} - New advanced network
     */
    enhanceNetwork(basicNetwork, textProcessor, wordEmbeddingDb = null) {
        const vocabSize = textProcessor.getVocabularySize();
        
        // Create advanced network configuration
        const config = {
            vocabSize,
            embeddingDim: this.options.embeddingDim,
            hiddenSizes: this.options.hiddenSizes,
            dropout: this.options.dropout,
            learningRate: this.options.learningRate,
            useLSTM: this.options.useLSTM,
            wordEmbeddingDb
        };
        
        // Create new advanced network
        const advancedNetwork = new AdvancedNetwork(config);
        
        // Load pre-trained embeddings if available
        if (wordEmbeddingDb) {
            advancedNetwork.loadEmbeddings(textProcessor.vocabulary);
        }
        
        return advancedNetwork;
    }
    
    /**
     * Create training data from examples for the advanced network
     * @param {Array<Object>} examples - Original training examples
     * @returns {Array<Object>} - Enhanced training data
     */
    prepareTrainingData(examples) {
        // Convert existing format to what the advanced network expects
        return examples.map(example => ({
            input: example.input,
            target: example.expectedOutput,
            inputWord: example.inputWord,
            outputWord: example.outputWord
        }));
    }
    
    /**
     * Train an enhanced network on examples
     * @param {AdvancedNetwork} network - Enhanced network
     * @param {Array<Object>} examples - Training examples
     * @param {Object} options - Training options
     * @returns {number} - Final loss
     */
    async trainEnhancedNetwork(network, examples, options = {}) {
        const enhancedExamples = this.prepareTrainingData(examples);
        
        // Split data into training and validation if needed
        let trainingSet = enhancedExamples;
        let validationSet = null;
        
        if (options.validationSplit) {
            const splitIndex = Math.floor(enhancedExamples.length * (1 - options.validationSplit));
            trainingSet = enhancedExamples.slice(0, splitIndex);
            validationSet = enhancedExamples.slice(splitIndex);
        }
        
        // Default training options
        const trainingOptions = {
            epochs: options.epochs || 50,
            batchSize: options.batchSize || 32,
            validationData: validationSet,
            earlyStoppingPatience: options.patience || 5,
            verbose: options.verbose !== false
        };
        
        // Train the network
        return network.trainBatch(trainingSet, trainingOptions);
    }
    
    /**
     * Generate a response using the enhanced network
     * @param {AdvancedNetwork} network - Enhanced network
     * @param {TextProcessor} textProcessor - Text processor
     * @param {string} inputText - Input text
     * @param {Object} options - Response options
     * @returns {string} - Generated response
     */
    generateResponse(network, textProcessor, inputText, options = {}) {
        const words = textProcessor.cleanText(inputText);
        if (!words || words.length === 0) {
            return "Please provide some text to generate a response.";
        }
        
        const responseLength = options.responseLength || 10;
        const temperature = options.temperature || 0.7;
        
        // Take the last few words as context
        const contextSize = Math.min(words.length, 3);
        const contextWords = words.slice(-contextSize);
        
        if (!textProcessor.vocabulary.has(contextWords[0])) {
            return `I don't recognize "${contextWords[0]}" yet. Try another word or teach me about it.`;
        }
        
        // Create input vector
        const vocabSize = textProcessor.getVocabularySize();
        const inputVector = Array(vocabSize).fill(0);
        
        // Set primary word with full weight
        inputVector[textProcessor.vocabulary.get(contextWords[0])] = 1;
        
        // Add context words with decreasing weight
        for (let i = 1; i < contextWords.length; i++) {
            if (textProcessor.vocabulary.has(contextWords[i])) {
                inputVector[textProcessor.vocabulary.get(contextWords[i])] = 1 / (i + 1);
            }
        }
        
        // Generate response
        let response = contextWords.join(' ');
        let currentInput = inputVector;
        const usedWords = new Set(contextWords);
        
        for (let i = 0; i < responseLength; i++) {
            // Get prediction
            const output = network.forward(currentInput, false, temperature);
            
            // Select next word (with improved sampling)
            let nextWord;
            if (options.topk) {
                // Get top-k words
                const topIndices = this.getTopKIndices(output, options.topk);
                
                // Sample from top-k with diversity-aware sampling
                const sampledIndex = this.diversitySampling(
                    topIndices, 
                    topIndices.map(idx => output[idx]),
                    usedWords,  // Pass used words to avoid repetition
                    textProcessor
                );
                
                nextWord = this.getWordFromIndex(textProcessor.vocabulary, sampledIndex);
            } else {
                // Use regular word selection
                nextWord = textProcessor.outputToText(output);
            }
            
            // Skip if we got an unknown word
            if (nextWord === "(unknown)") continue;
            
            // Add to response
            response += ' ' + nextWord;
            
            // Update used words
            usedWords.add(nextWord);
            
            // Update input for next word
            currentInput = Array(vocabSize).fill(0);
            currentInput[textProcessor.vocabulary.get(nextWord)] = 1;
            
            // Include context from the last word with lower weight
            for (let j = 1; j < contextWords.length; j++) {
                if (textProcessor.vocabulary.has(contextWords[j])) {
                    currentInput[textProcessor.vocabulary.get(contextWords[j])] = 0.3 / j;
                }
            }
        }
        
        return response;
    }
    
    /**
     * Diversity-aware weighted sampling to reduce repetition
     * @param {Array<number>} indices - Indices to sample from
     * @param {Array<number>} weights - Weights for sampling
     * @param {Set<string>} usedWords - Already used words
     * @param {TextProcessor} textProcessor - For word lookup
     * @returns {number} - Sampled index
     */
    diversitySampling(indices, weights, usedWords, textProcessor) {
        // Create modified weights with penalties for recent words
        const modifiedWeights = weights.map((w, i) => {
            const word = this.getWordFromIndex(textProcessor.vocabulary, indices[i]);
            return usedWords.has(word) ? w * 0.3 : w; // Apply penalty if word was used
        });
        
        // Normalize weights
        const sum = modifiedWeights.reduce((a, b) => a + b, 0);
        const normalizedWeights = sum > 0 ? modifiedWeights.map(w => w / sum) : weights.map(w => 1/weights.length);
        
        // Sample with modified weights
        const random = Math.random();
        let cumSum = 0;
        
        for (let i = 0; i < indices.length; i++) {
            cumSum += normalizedWeights[i];
            if (random <= cumSum) {
                return indices[i];
            }
        }
        
        // Default to first index
        return indices[0];
    }
    
    /**
     * Get top K indices from an array
     * @param {Array<number>} arr - Input array
     * @param {number} k - Number of top indices to return
     * @returns {Array<number>} - Top K indices
     */
    getTopKIndices(arr, k) {
        return arr
            .map((val, idx) => ({ val, idx }))
            .sort((a, b) => b.val - a.val)
            .slice(0, k)
            .map(item => item.idx);
    }
    
    /**
     * Sample from a distribution
     * @param {Array<number>} indices - Indices to sample from
     * @param {Array<number>} weights - Weights for sampling
     * @returns {number} - Sampled index
     */
    weightedSample(indices, weights) {
        // Normalize weights
        const sum = weights.reduce((a, b) => a + b, 0);
        const normalizedWeights = weights.map(w => w / sum);
        
        // Generate random value
        const random = Math.random();
        
        // Sample
        let cumSum = 0;
        for (let i = 0; i < indices.length; i++) {
            cumSum += normalizedWeights[i];
            if (random <= cumSum) {
                return indices[i];
            }
        }
        
        // Default to first index
        return indices[0];
    }
    
    /**
     * Get word from index using vocabulary
     * @param {Map<string, number>} vocabulary - Vocabulary map
     * @param {number} index - Word index
     * @returns {string} - Word
     */
    getWordFromIndex(vocabulary, index) {
        for (const [word, idx] of vocabulary.entries()) {
            if (idx === index) {
                return word;
            }
        }
        return "(unknown)";
    }
    
    /**
     * Create a new WordEmbeddings instance
     * @param {Object} options - Configuration options
     * @returns {WordEmbeddings} - New word embeddings instance
     */
    createWordEmbeddings(options = {}) {
        return new WordEmbeddings({
            dimensions: options.dimensions || this.options.embeddingDim,
            storageDir: options.storageDir || './embeddings'
        });
    }
}

module.exports = NetworkEnhancer;
