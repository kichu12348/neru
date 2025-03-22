const readline = require('readline');
const fs = require('fs').promises;
const NeuralNetwork = require('./neuralNetwork');
const TextProcessor = require('./textProcessor');
const trainingData = require('./trainingData');
const WordEmbeddings = require('./wordEmbeddings');
const ModelStorage = require('./modelStorage');
const AssociationTester = require('./testAssociations');
const AssociationAnalyzer = require('./AssociationAnalyzer');
const AdvancedNetwork = require('./advancedNetwork');
const NetworkEnhancer = require('./networkEnhancer');

// Create interface for terminal input/output
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

// Neural network configuration
const HIDDEN_LAYERS = [96, 64, 48]; // Larger layers for more capacity
const N_GRAM_SIZE = 3; // Context window
let RESPONSE_TEMPERATURE = 0.7; // Response randomness
const RESPONSE_LENGTH = 10; // Response length
const MIN_ERROR_THRESHOLD = 0.0001; // Stricter error threshold
const BATCH_SIZE = 15; // Larger batch size
const INITIAL_TRAINING_EPOCHS = 2000; // More epochs for initial training

// Initialize text processor with n-gram support
const textProcessor = new TextProcessor();
textProcessor.setNGramSize(N_GRAM_SIZE);

// Initialize WordEmbeddings and ModelStorage instances
const wordEmbeddings = new WordEmbeddings();
const modelStorage = new ModelStorage();

// Process explicit training data
console.log("Processing training data...");

// First add all words to vocabulary
trainingData.forEach(pair => {
    textProcessor.updateVocabulary(`${pair[0]} ${pair[1]}`);
});

const initialVocabSize = textProcessor.getVocabularySize();
console.log(`Initial vocabulary size: ${initialVocabSize} words`);

// Initialize neural network with reinforcement learning capability
const network = new NeuralNetwork(
    initialVocabSize,
    HIDDEN_LAYERS,
    initialVocabSize
);

// Initialize enhanced components
const networkEnhancer = new NetworkEnhancer({
    embeddingDim: 50,
    useLSTM: true,  // Set to false for simpler model
    hiddenSizes: [64, 48, 32],
    dropout: 0.2,
    learningRate: 0.001
});

let useEnhancedNetwork = true; // Set this to use the enhanced network architecture

// Initialize the enhanced neural network
let enhancedNetwork = null;
if (useEnhancedNetwork) {
    enhancedNetwork = networkEnhancer.enhanceNetwork(
        network, 
        textProcessor, 
        wordEmbeddings
    );
}

// Convert explicit pairs to training examples
console.log("Converting training data to examples...");
const trainingExamples = [];
trainingData.forEach(pair => {
    const example = textProcessor.createExampleFromPair(pair[0], pair[1]);
    if (example) trainingExamples.push(example);
});

console.log(`Created ${trainingExamples.length} training examples`);

// Train on explicit pairs with more intensive verification
console.log("Training on examples...");
let failedExamples = [];

for (let epoch = 0; epoch < INITIAL_TRAINING_EPOCHS; epoch++) {
    // Shuffle examples each epoch
    const shuffled = [...trainingExamples].sort(() => Math.random() - 0.5);
    
    let totalError = 0;
    let examples = shuffled.length;
    
    // Process in batches
    for (let i = 0; i < shuffled.length; i += BATCH_SIZE) {
        const batch = shuffled.slice(i, i + BATCH_SIZE);
        
        batch.forEach(example => {
            const error = network.train(
                example.input, 
                example.expectedOutput,
                example.inputWord,
                example.outputWord,
                1.5 // Higher base reinforcement for initial training
            );
            
            if (!isNaN(error) && isFinite(error)) {
                totalError += error;
            }
        });
    }
    
    const avgError = examples > 0 ? totalError / examples : 0;
    if (epoch % 100 === 0) {
        console.log(`Epoch ${epoch}, Average Error: ${avgError.toFixed(8)}`);
        
        // Verify training progress and focus on failing examples
        if (epoch > 0 && epoch % 300 === 0) {
            verifyAndRepairTraining();
        }
    }
    
    if (avgError < MIN_ERROR_THRESHOLD) {
        console.log(`Reached error threshold at epoch ${epoch}. Stopping training.`);
        break;
    }
}

// Verify training and fix failures
function verifyAndRepairTraining() {
    console.log("\nVerifying training examples...");
    failedExamples = [];
    
    let success = 0;
    let total = 0;
    const sampleSize = Math.min(20, trainingExamples.length);
    const sampledIndices = new Set();
    
    // Test a random sample first
    while (sampledIndices.size < sampleSize) {
        sampledIndices.add(Math.floor(Math.random() * trainingExamples.length));
    }
    
    sampledIndices.forEach(index => {
        const example = trainingExamples[index];
        const output = network.feedForward(example.input);
        const predicted = textProcessor.outputToText(output);
        total++;
        
        if (predicted === example.outputWord) {
            success++;
        } else {
            failedExamples.push(example);
            console.log(`Failed: "${example.inputWord}" → "${predicted}" (Expected: "${example.outputWord}")`);
        }
    });
    
    const successRate = (success / total * 100).toFixed(1);
    console.log(`Success rate: ${successRate}% (${success}/${total})`);
    
    // If we have failures, try to fix some of them
    if (failedExamples.length > 0) {
        console.log(`Attempting to fix ${Math.min(5, failedExamples.length)} failing examples...`);
        
        // Fix up to 5 failing examples with intensive training
        for (let i = 0; i < Math.min(5, failedExamples.length); i++) {
            const example = failedExamples[i];
            console.log(`Fixing: "${example.inputWord}" → "${example.outputWord}"`);
            
            const result = network.trainUntilLearned(
                example.input,
                example.expectedOutput,
                example.inputWord,
                example.outputWord,
                50, // More attempts
                0.00001 // Stricter threshold
            );
            
            // Test the result
            const output = network.feedForward(example.input);
            const predicted = textProcessor.outputToText(output);
            
            console.log(`Result: "${example.inputWord}" → "${predicted}" (${result.success ? "Fixed!" : "Still failing"})`);
        }
    }
}

// Final verification of all training examples
console.log("\nFinal verification of all training examples...");
const tester = new AssociationTester(network, textProcessor);
const testResults = tester.testMultiple(trainingData);
console.log(`Overall success rate: ${testResults.successRate.toFixed(1)}% (${testResults.successCount}/${testResults.total})`);

// If success rate is below 80%, fix failing associations
if (testResults.successRate < 80) {
    console.log("Success rate is below 80%, fixing failing associations...");
    tester.fixFailedAssociations(100);
}

console.log("Initial training complete!");

// Handle special commands
async function handleCommand(cmd) {
    const commandParts = cmd.substring(1).trim().split(' ');
    const command = commandParts[0].toLowerCase();
    const args = commandParts.slice(1);
    
    switch(command) {
        case 'help':
            console.log("\nImproved Neural Network Text Learning Commands:");
            console.log("  /help               - Show this help message");
            console.log("  /test word          - Test what the network outputs for a specific word");
            console.log("  /save filename      - Save the current neural network state");
            console.log("  /load filename      - Load a saved neural network state");
            console.log("  /clear              - Clear the console");
            console.log("  /vocab              - Show the current vocabulary");
            console.log("  /temp value         - Set temperature (0.1-2.0) for response generation");
            console.log("  /context word       - Show context words learned for a specific word");
            console.log("  /train text         - Intensively train on specific text");
            console.log("  /respond text       - Generate a response without training");
            console.log("  /trainuntil input output [maxAttempts] - Train until the network learns the association");
            console.log("  /forgetall          - Reset the neural network and start over");
            console.log("  /verify             - Verify all training examples and show success rate");
            console.log("  /forcelearn input output [factor] - Force learning a specific association");
            console.log("  /conflicts          - Find and display conflicting associations");
            console.log("  /models             - List available models");
            console.log("  /similar word       - Find words with similar vectors");
            console.log("  /embed word         - Get word embedding vector");
            console.log("  /useadvanced [true/false] - Switch between basic and enhanced network architecture");
            console.log("\nTraining Formats:");
            console.log("  [input, output]     - Explicitly teach the network that input → output");
            console.log("  text with words     - Train on word sequences (each word → next word)");
            console.log("  single word         - Just adds to vocabulary (no training)");
            return true;
            
        case 'test':
            if (args.length === 0) {
                console.log("Please provide a word to test, e.g., /test hello");
                return true;
            }
            
            const testWord = args[0].toLowerCase();
            if (!textProcessor.vocabulary.has(testWord)) {
                console.log(`The word "${testWord}" is not in the vocabulary yet.`);
                return true;
            }
            
            // Test the network with the word
            const vocabSize = textProcessor.getVocabularySize();
            const inputVector = Array(vocabSize).fill(0);
            inputVector[textProcessor.vocabulary.get(testWord)] = 1;
            
            const output = network.feedForward(inputVector);
            const nextWord = textProcessor.outputToText(output);
            console.log(`Test: When given "${testWord}", the network outputs: "${nextWord}"`);
            return true;
            
        case 'vocab':
            const words = Array.from(textProcessor.vocabulary.keys()).sort();
            console.log(`\nVocabulary (${words.length} words):`);
            console.log(words.join(', '));
            return true;
            
        case 'save':
            const saveFilename = args.length > 0 ? args[0] : 'network.json';
            saveModel(saveFilename)
                .then(() => console.log(`Model saved to ${saveFilename}`))
                .catch(err => console.error('Error saving model:', err));
            return true;
            
        case 'load':
            const loadFilename = args.length > 0 ? args[0] : 'network.json';
            loadModel(loadFilename)
                .then(() => console.log(`Model loaded from ${loadFilename}`))
                .catch(err => console.error('Error loading model:', err));
            return true;
            
        case 'clear':
            console.clear();
            return true;
            
        case 'temp':
            if (args.length === 0) {
                console.log(`Current temperature: ${RESPONSE_TEMPERATURE}`);
                return true;
            }
            
            const newTemp = parseFloat(args[0]);
            if (isNaN(newTemp) || newTemp <= 0) {
                console.log("Temperature must be a positive number. Recommended: 0.1-2.0");
                return true;
            }
            
            RESPONSE_TEMPERATURE = newTemp;
            console.log(`Temperature set to: ${RESPONSE_TEMPERATURE}`);
            return true;
            
        case 'context':
            if (args.length === 0) {
                console.log("Please provide a word to check its context, e.g., /context hello");
                return true;
            }
            
            const contextWord = args[0].toLowerCase();
            if (!textProcessor.vocabulary.has(contextWord)) {
                console.log(`The word "${contextWord}" is not in the vocabulary yet.`);
                return true;
            }
            
            const contextWords = textProcessor.getContextWords(contextWord);
            if (contextWords.length > 0) {
                console.log(`Words that commonly follow "${contextWord}": ${contextWords.join(', ')}`);
            } else {
                console.log(`No context data available for "${contextWord}"`);
            }
            return true;
            
        case 'train':
            // Train the network intensively
            if (args.length === 0) {
                console.log("Please provide text to train on, e.g., /train hello world");
                return true;
            }
            
            const trainText = args.join(' ');
            console.log(`Intensively training on: "${trainText}"`);
            
            // Update vocabulary
            textProcessor.updateVocabulary(trainText);
            
            // Get training examples
            const trainExamples = textProcessor.textToInputs(trainText);
            if (!trainExamples || trainExamples.length === 0) {
                console.log("Could not create training examples from this text.");
                return true;
            }
            
            if (useEnhancedNetwork) {
                // Train enhanced network
                console.log("Training enhanced network intensively...");
                
                // Resize if necessary
                if (enhancedNetwork.vocabSize < textProcessor.getVocabularySize()) {
                    enhancedNetwork.resize(textProcessor.getVocabularySize());
                }
                
                // Train with enhanced options
                const trainingOptions = {
                    epochs: 50,
                    batchSize: 8,
                    validationSplit: 0.2,
                    patience: 5,
                    verbose: true
                };
                
                const enhancedExamples = networkEnhancer.prepareTrainingData(trainExamples);
                enhancedNetwork.trainBatch(enhancedExamples, trainingOptions);
                
                // Test enhanced network
                const testExample = enhancedExamples[0];
                const output = enhancedNetwork.forward(testExample.input);
                const predictedWord = textProcessor.outputToText(output);
                
                console.log(`Test: "${testExample.inputWord}" → "${predictedWord}" (expected: "${testExample.outputWord}")`);
            } else {
                // Use original training code
                let finalError = 0;
            
                // Iterate through examples, training each intensively
                trainExamples.forEach((example, i) => {
                    const error = network.trainIntensively(example.input, example.expectedOutput);
                    finalError += error;
                    
                    // Show progress for larger training sets
                    if (i % 5 === 0 && trainExamples.length > 5) {
                        process.stdout.write('.');
                    }
                });
                
                if (trainExamples.length > 5) process.stdout.write('\n');
                console.log(`Intensive training complete. Final avg error: ${(finalError / trainExamples.length).toExponential(4)}`);
                
                // Test the results immediately for user feedback
                const testExample = trainExamples[0];
                const trainingTestOutput = network.feedForward(testExample.input);
                const predictedWord = textProcessor.outputToText(trainingTestOutput);
                
                console.log(`Test: "${testExample.inputWord}" → "${predictedWord}" (expected: "${testExample.outputWord}")`);
            }
            return true;
            
        case 'respond':
            // Generate response without training
            if (args.length === 0) {
                console.log("Please provide starting text for response, e.g., /respond hello");
                return true;
            }
            
            const startText = args.join(' ');
            
            if (useEnhancedNetwork) {
                const response = networkEnhancer.generateResponse(
                    enhancedNetwork, 
                    textProcessor, 
                    startText, 
                    {
                        responseLength: RESPONSE_LENGTH,
                        temperature: RESPONSE_TEMPERATURE,
                        topk: 5 // Use top-5 sampling for better diversity
                    }
                );
                console.log("Enhanced network response:", response);
            } else {
                generateResponse(startText, true);
            }
            return true;

        case 'trainuntil':
            // New command for training until successful
            if (args.length < 2) {
                console.log("Usage: /trainuntil input output [maxAttempts]");
                return true;
            }
            
            const inputWord = args[0].toLowerCase();
            let outputWord;
            
            // Check if it's multiple outputs in JSON array syntax
            if (args.length >= 3 && args[1] === '[') {
                // Find the closing bracket
                let closingIndex = -1;
                for (let i = 2; i < args.length; i++) {
                    if (args[i].endsWith(']')) {
                        closingIndex = i;
                        break;
                    }
                }
                
                if (closingIndex > 0) {
                    // Extract the outputs as an array
                    const outputs = args.slice(2, closingIndex + 1).join(' ').replace(/\]$/, '').split(',');
                    outputWord = outputs.map(o => o.trim().toLowerCase());
                    args.splice(1, closingIndex); // Remove the array arguments
                } else {
                    outputWord = args[1].toLowerCase();
                }
            } else {
                outputWord = args[1].toLowerCase();
            }
            
            const maxAttempts = args.length > 2 ? parseInt(args[2]) : 50;
            
            // Create training example with possibly multiple outputs
            const example = textProcessor.createExampleFromPair(inputWord, outputWord);
            if (!example) {
                console.log("Could not create training example from these words.");
                return true;
            }
            
            console.log(`Training "${inputWord}" → "${Array.isArray(outputWord) ? outputWord.join('", "') : outputWord}" until learned (max ${maxAttempts} attempts)...`);
            
            // Train until learned
            const result = network.trainUntilLearned(
                example.input,
                example.expectedOutput,
                inputWord,
                example.outputWord,
                maxAttempts,
                0.0001
            );
            
            // ...existing rest of the command...
            
            return true;
            
        case 'forgetall':
            // Command to reset the network and start over
            console.log("Reinitializing neural network (forgetting everything)...");
            // Create a new network with the same architecture
            network = new NeuralNetwork(initialVocabSize, HIDDEN_LAYERS, initialVocabSize);
            console.log("Neural network reinitialized. All learned patterns have been reset.");
            return true;

        case 'verify':
            // Verify all training examples and show success rate
            console.log("Verifying all training examples...");
            const tester = new AssociationTester(network, textProcessor);
            const testResults = tester.testMultiple(trainingData);
            
            console.log(`Overall success rate: ${testResults.successRate.toFixed(1)}% (${testResults.successCount}/${testResults.total})`);
            
            if (args.length > 0 && args[0].toLowerCase() === 'fix') {
                console.log("Fixing failing associations...");
                tester.fixFailedAssociations(100);
            }
            return true;
            
        case 'forcelearn':
            // Force learning a specific association
            if (args.length < 2) {
                console.log("Usage: /forcelearn input output [factor]");
                return true;
            }
            
            const forceInputWord = args[0].toLowerCase();
            const forceOutputWord = args[1].toLowerCase();
            const factor = args.length > 2 ? parseFloat(args[2]) : 10.0;
            
            console.log(`Forcing association: "${forceInputWord}" → "${forceOutputWord}" (factor: ${factor})`);
            
            // Create training example
            const forceExample = textProcessor.createExampleFromPair(forceInputWord, forceOutputWord);
            if (!forceExample) {
                console.log("Could not create example from these words.");
                return true;
            }
            
            // Before state
            const beforeOutput = network.feedForward(forceExample.input);
            const beforePrediction = textProcessor.outputToText(beforeOutput);
            console.log(`Before: "${forceInputWord}" → "${beforePrediction}"`);
            
            // Apply direct weight modification
            network.forceAssociation(forceExample.input, forceExample.expectedOutput, factor);
            
            // After state
            const afterOutput = network.feedForward(forceExample.input);
            const afterPrediction = textProcessor.outputToText(afterOutput);
            console.log(`After: "${forceInputWord}" → "${afterPrediction}"`);
            
            if (afterPrediction === forceOutputWord) {
                console.log("Successfully forced the association!");
            } else {
                console.log("Association was not completely successful, try increasing the factor.");
            }
            return true;
            
        case 'conflicts':
            // Find and display conflicting associations
            console.log("Analyzing association conflicts...");
            const analyzer = new AssociationAnalyzer(network, textProcessor, trainingData);
            const conflicts = analyzer.findConflicts();
            
            if (conflicts.length === 0) {
                console.log("No conflicts found.");
            } else {
                console.log(`Found ${conflicts.length} conflicts:`);
                conflicts.slice(0, 10).forEach((conflict, i) => {
                    console.log(`${i+1}. "${conflict.word}" has conflicting targets: ${conflict.targets.join(', ')}`);
                });
                
                if (conflicts.length > 10) {
                    console.log(`...and ${conflicts.length - 10} more conflicts.`);
                }
            }
            return true;

        case 'models':
            // List available models
            const models = await modelStorage.listModels();
            console.log("\nAvailable models:");
            models.forEach((model, i) => {
                const date = new Date(model.timestamp).toLocaleString();
                console.log(`${i+1}. ${model.name} (${date})`);
            });
            return true;

        case 'similar':
            // Find words with similar vectors
            if (args.length === 0) {
                console.log("Please provide a word to find similar words for");
                return true;
            }
            
            const similarWords = await wordEmbeddings.findSimilarWords(args[0]);
            console.log(`\nWords similar to "${args[0]}":`);
            similarWords.forEach((item, i) => {
                console.log(`${i+1}. ${item.word} (${item.similarity.toFixed(4)})`);
            });
            return true;

        case 'embed':
            // Get word embedding vector
            if (args.length === 0) {
                console.log("Please provide a word to show its embedding");
                return true;
            }
            
            const embedding = await wordEmbeddings.getWordVector(args[0]);
            console.log(`\nEmbedding for "${args[0]}":`);
            console.log(embedding.slice(0, 10).map(v => v.toFixed(4)).join(', ') + '...');
            return true;

        case 'useadvanced':
            const useAdvanced = args.length > 0 ? args[0].toLowerCase() === 'true' : true;
            useEnhancedNetwork = useAdvanced;
            
            if (useEnhancedNetwork && !enhancedNetwork) {
                enhancedNetwork = networkEnhancer.enhanceNetwork(
                    network, 
                    textProcessor, 
                    wordEmbeddings
                );
                console.log("Enhanced network initialized. Use /train to train it.");
            }
            
            console.log(`Using ${useEnhancedNetwork ? 'enhanced' : 'basic'} network architecture.`);
            return true;

        default:
            console.log(`Unknown command: ${command}. Type /help for available commands.`);
            return true;
    }
}

// Save model to file
async function saveModel(filename = 'network.json') {
    try {
        // Save with model storage
        const modelPath = await modelStorage.saveModel(network, filename, {
            vocabSize: textProcessor.getVocabularySize(),
            trainingData: `${trainingData.length} examples`,
            timestamp: Date.now()
        });
        
        // Also update word embeddings
        await wordEmbeddings.updateFromVocabulary(textProcessor);
        await wordEmbeddings.save();
        
        return modelPath;
    } catch (err) {
        console.error("Error saving model:", err);
        throw err;
    }
}

// Load model from file
async function loadModel(filename = 'network.json') {
    try {
        // Load with model storage
        const result = await modelStorage.loadModel(filename, network);
        
        // Update text processor vocabulary if available
        // ...existing vocabulary restoration code...
        
        return result;
    } catch (err) {
        console.error("Error loading model:", err);
        throw err;
    }
}

// Initialize storage systems
async function initializeStorage() {
    console.log("Initializing storage systems...");
    try {
        await Promise.all([
            wordEmbeddings.initialize?.() || Promise.resolve(), // Use optional chaining in case initialize doesn't exist
            modelStorage.initialize?.() || Promise.resolve()
        ]);
        console.log("Storage systems ready.");
    } catch (error) {
        console.error("Error initializing storage:", error);
        console.log("Continuing without storage functionality.");
    }
}

// Generate a response with improved word selection
function generateResponse(inputText, verbose = false) {
    const words = textProcessor.cleanText(inputText);
    
    if (!words || words.length === 0) {
        console.log("Please provide some text to generate a response.");
        return;
    }
    
    // Take the last 1-3 words as context for response
    const contextWords = words.slice(Math.max(0, words.length - N_GRAM_SIZE));
    
    if (!textProcessor.vocabulary.has(contextWords[0])) {
        console.log(`I don't recognize "${contextWords[0]}" yet. Try another word or teach me about it.`);
        return;
    }
    
    const vocabSize = textProcessor.getVocabularySize();
    
    // Create input vector with primary word and context
    const inputVector = Array(vocabSize).fill(0);
    inputVector[textProcessor.vocabulary.get(contextWords[0])] = 1;
    
    // Add context weights for other words if available
    for (let i = 1; i < contextWords.length; i++) {
        if (textProcessor.vocabulary.has(contextWords[i])) {
            inputVector[textProcessor.vocabulary.get(contextWords[i])] = 1 / (i + 1);
        }
    }
    
    // Generate response
    let response = contextWords.join(' ');
    let currentVector = inputVector;
    let usedWords = new Set(contextWords);
    let lastWord = contextWords[contextWords.length - 1];
    
    // Maintain a history of recent responses to detect loops
    const recentWordHistory = [...contextWords];
    
    // Generate response words with improved variety
    for (let i = 0; i < RESPONSE_LENGTH; i++) {
        // Get output with temperature control
        const output = network.feedForward(currentVector, RESPONSE_TEMPERATURE);
        
        // Get multiple options
        const topN = Math.min(10, vocabSize);
        const topWords = textProcessor.getTopNWords(output, topN);
        
        // Get context words
        const contextSuggestions = textProcessor.getContextWords(lastWord, 3);
        
        // Score words based on multiple criteria
        const scoredWords = [];
        
        // Process all candidate words
        const candidateWords = [...new Set([...topWords, ...contextSuggestions])];
        for (const word of candidateWords) {
            // Skip empty words and single characters (except "i" and "a")
            if (!word || word === '' || (word.length === 1 && !['i', 'a'].includes(word))) continue;
            
            let score = 0;
            
            // Prefer words from network output (higher in list = better)
            const outputRank = topWords.indexOf(word);
            if (outputRank >= 0) {
                score += (topN - outputRank) / topN; // 1.0 to 0.1 based on rank
            }
            
            // Prefer contextually relevant words
            const contextRank = contextSuggestions.indexOf(word);
            if (contextRank >= 0) {
                score += 0.5 * (3 - contextRank) / 3; // 0.5 to ~0.17 based on rank
            }
            
            // Penalize words that were already used recently
            if (usedWords.has(word)) {
                score -= 0.7;
            }
            
            // Avoid repeating the most recent word
            if (word === lastWord) {
                score -= 0.9;
            }
            
            // Severely penalize words that would create a pattern repeat
            // Check for repeating patterns in recent words
            let patternRepeatPenalty = 0;
            
            // Check for immediate repeats (stricter penalty)
            if (word === lastWord) {
                patternRepeatPenalty = 1.5;
            }
            
            // Check for recent usage (graduated penalty)
            else if (recentWordHistory.includes(word)) {
                // Calculate how recently the word was used
                const lastIndex = recentWordHistory.lastIndexOf(word);
                const recency = recentWordHistory.length - lastIndex;
                
                // More recent = higher penalty (up to -1.0 for very recent)
                patternRepeatPenalty = Math.max(0, 1.0 - (recency * 0.1));
            }
            
            // Check for repeating bigrams or trigrams
            if (recentWordHistory.length >= 4) {
                // Check if this would form a repeating bigram
                const potentialBigram = [lastWord, word];
                
                // Look through history for this bigram
                for (let j = 1; j < recentWordHistory.length; j++) {
                    if (recentWordHistory[j-1] === potentialBigram[0] && 
                        recentWordHistory[j] === potentialBigram[1]) {
                        // Found a repeat - apply strong penalty
                        patternRepeatPenalty = Math.max(patternRepeatPenalty, 2.0);
                        break;
                    }
                }
            }
            
            score -= patternRepeatPenalty;
            
            // Add a slight preference for longer words (avoid very short words)
            if (word.length > 3) {
                score += 0.1;
            }
            
            // Add to scored words list
            scoredWords.push({ word, score });
        }
        
        // Sort by score
        scoredWords.sort((a, b) => b.score - a.score);
        
        // Get the best word
        let nextWord = scoredWords.length > 0 ? scoredWords[0].word : "(unknown)";
        
        // Debug info
        if (verbose && i === 0) {
            console.log("Word selection options:");
            scoredWords.slice(0, 5).forEach((item, idx) => {
                console.log(`  ${idx + 1}. ${item.word} (score: ${item.score.toFixed(2)})`);
            });
        }
        
        response += ' ' + nextWord;
        usedWords.add(nextWord);
        recentWordHistory.push(nextWord);
        lastWord = nextWord;
        
        // Update input for next word with better context handling
        currentVector = Array(vocabSize).fill(0);
        
        // Primary word (most recent)
        if (textProcessor.vocabulary.has(nextWord)) {
            currentVector[textProcessor.vocabulary.get(nextWord)] = 1;
        }
        
        // Add context from recent words with decreasing weights
        const contextWindowSize = Math.min(N_GRAM_SIZE - 1, recentWordHistory.length - 1);
        for (let j = 1; j <= contextWindowSize; j++) {
            const contextWord = recentWordHistory[recentWordHistory.length - 1 - j];
            if (textProcessor.vocabulary.has(contextWord)) {
                currentVector[textProcessor.vocabulary.get(contextWord)] = 0.5 / j;
            }
        }
    }
    
    console.log("Network response:", response);
    return response;
}

// Function to process user input with reinforcement learning
function processUserInput() {
    rl.question("Enter text or [input, expectedOutput] pair (or '/help' for commands): ", (input) => {
        if (input.toLowerCase() === 'quit') {
            rl.close();
            return;
        }
        
        // Handle empty input
        if (!input || input.trim() === '') {
            console.log("Please enter some text or a command.");
            processUserInput();
            return;
        }
        
        // Check for commands
        if (input.startsWith('/')) {
            if (handleCommand(input)) {
                processUserInput();
                return;
            }
        }
        
        // Check if input is in explicit pair format
        const isExplicitPair = input.trim().startsWith('[') && input.trim().includes(',');
        
        if (isExplicitPair) {
            // Parse explicit pair with improved parsing
            try {
                // Extract the input and output words with better parsing
                const cleanText = input.replace(/\[|\]/g, '').trim();
                
                // We need a more complex parsing to handle array responses
                let inputWord = "";
                let outputWords = [];
                
                // Check if we have an array format [input, [output1, output2]]
                const arrayMatch = cleanText.match(/([^,]+),\s*\[(.*)\]/);
                if (arrayMatch && arrayMatch.length >= 3) {
                    inputWord = arrayMatch[1].trim().toLowerCase();
                    outputWords = arrayMatch[2].split(',').map(o => o.trim().toLowerCase());
                } else {
                    // Regular format [input, output]
                    const parts = cleanText.split(',');
                    if (parts.length >= 2) {
                        inputWord = parts[0].trim().toLowerCase();
                        outputWords = [parts[1].trim().toLowerCase()];
                    }
                }
                
                if (!inputWord || outputWords.length === 0) {
                    console.log("Invalid format. Use [input, output] or [input, [output1, output2]]");
                    processUserInput();
                    return;
                }
                
                // Create training example with multiple possible outputs
                const example = textProcessor.createExampleFromPair(inputWord, outputWords);
                if (!example) {
                    console.log("Could not create training example from these words.");
                    processUserInput();
                    return;
                }
                
                console.log(`Training: "${inputWord}" → "${outputWords.length > 1 ? outputWords.join('", "') : outputWords[0]}"`);
                
                // Check current prediction before training
                const initialOutput = network.feedForward(example.input);
                const initialPrediction = textProcessor.outputToText(initialOutput);
                console.log(`Before training: "${inputWord}" → "${initialPrediction}"`);
                
                if (useEnhancedNetwork) {
                    // Train enhanced network on explicit pair
                    console.log(`Training enhanced network: "${inputWord}" → "${outputWords.length > 1 ? outputWords.join('", "') : outputWords[0]}"`);
                    
                    // Resize if needed
                    if (enhancedNetwork.vocabSize < textProcessor.getVocabularySize()) {
                        enhancedNetwork.resize(textProcessor.getVocabularySize());
                    }
                    
                    // Test before training
                    const beforeOutput = enhancedNetwork.forward(example.input);
                    const beforePrediction = textProcessor.outputToText(beforeOutput);
                    console.log(`Before training: "${inputWord}" → "${beforePrediction}"`);
                    
                    // Train intensively
                    const enhancedExample = {
                        input: example.input,
                        target: example.expectedOutput,
                        inputWord: example.inputWord,
                        outputWord: example.outputWord
                    };
                    
                    // Train multiple times
                    for (let i = 0; i < 50; i++) {
                        enhancedNetwork.train(enhancedExample.input, enhancedExample.target);
                    }
                    
                    // Test after training
                    const afterOutput = enhancedNetwork.forward(example.input);
                    const afterPrediction = textProcessor.outputToText(afterOutput);
                    console.log(`After training: "${inputWord}" → "${afterPrediction}" (Expected: "${outputWords.length > 1 ? outputWords.join('", "') : outputWords[0]}")`);
                    
                    if (outputWords.includes(afterPrediction)) {
                        console.log("Successfully learned!");
                    } else {
                        console.log("Training was not completely successful. Try repeating the training.");
                    }
                } else {
                    // Train until learned using reinforcement learning
                    const result = network.trainUntilLearned(
                        example.input,
                        example.expectedOutput,
                        inputWord,
                        outputWords,
                        30,
                        0.0001
                    );
                    
                    // Check result after training
                    const finalOutput = network.feedForward(example.input);
                    const finalPrediction = textProcessor.outputToText(finalOutput);
                    
                    if (result.success) {
                        console.log(`Successfully learned in ${result.attempts} attempts!`);
                    } else {
                        console.log(`Training completed with ${result.attempts} attempts.`);
                    }
                    
                    console.log(`Test: "${inputWord}" → "${finalPrediction}" (Expected: "${outputWords.length > 1 ? outputWords.join('", "') : outputWords[0]}")`);
                    
                    if (!outputWords.includes(finalPrediction)) {
                        console.log("Training was not completely successful. Try repeating the training.");
                    }
                }
            } catch (error) {
                console.log("Error processing explicit pair:", error.message);
            }
        } else {
            // Regular text input
            const words = textProcessor.cleanText(input);
            
            if (words && words.length > 0) {
                // Add new words to vocabulary
                const beforeSize = textProcessor.getVocabularySize();
                textProcessor.updateVocabulary(input);
                const afterSize = textProcessor.getVocabularySize();
                
                if (afterSize > beforeSize) {
                    console.log(`Added ${afterSize - beforeSize} new words to vocabulary`);
                    network.resize(afterSize, afterSize);
                }
                
                // Extract training examples if multiple words
                if (words.length > 1) {
                    const examples = textProcessor.textToInputs(input);
                    
                    if (examples && examples.length > 0) {
                        console.log("Training on examples:");
                        examples.forEach(example => {
                            console.log(`  ${example.inputWord} → ${example.outputWord}`);
                        });
                        
                        // Train on examples
                        let totalError = 0;
                        examples.forEach(example => {
                            const error = network.trainIntensively(
                                example.input, 
                                example.expectedOutput,
                                example.inputWord,
                                example.outputWord,
                                100
                            );
                            totalError += error;
                        });
                        
                        console.log(`Training complete. Average error: ${(totalError / examples.length).toExponential(4)}`);
                    }
                } else {
                    console.log("Added to vocabulary, but need more words to train.");
                }
                
                // Generate a response with the appropriate network
                if (useEnhancedNetwork) {
                    const response = networkEnhancer.generateResponse(
                        enhancedNetwork, 
                        textProcessor, 
                        input, 
                        {
                            responseLength: RESPONSE_LENGTH,
                            temperature: RESPONSE_TEMPERATURE,
                            topk: 5 // Use top-5 sampling for better diversity
                        }
                    );
                    console.log("Enhanced network response:", response);
                } else {
                    generateResponse(input);
                }
            }
        }
        
        processUserInput();
    });
}

// Initialize storage and enhanced network components
async function initializeComponents() {
    try {
        console.log("Initializing storage systems...");
        await Promise.all([
            wordEmbeddings.initialize?.() || Promise.resolve(),
            modelStorage.initialize?.() || Promise.resolve()
        ]);
        console.log("Storage systems ready.");
        
        if (useEnhancedNetwork) {
            console.log("Initializing enhanced network...");
            enhancedNetwork = networkEnhancer.enhanceNetwork(
                network, 
                textProcessor, 
                wordEmbeddings
            );
            await enhancedNetwork.loadEmbeddings(textProcessor.vocabulary);
            console.log("Enhanced network ready.");
        }
    } catch (error) {
        console.error("Error initializing components:", error);
        console.log("Continuing with limited functionality.");
    }
}

// Start the interactive session
initializeComponents().then(() => {
    console.log("Neural network is ready!");
    console.log(`Using ${useEnhancedNetwork ? 'enhanced' : 'basic'} architecture.`);
    console.log("Type /help for available commands or [input, output] to teach specific associations.");
    processUserInput();
}).catch(error => {
    console.error("Error during initialization:", error);
    console.log("Starting with limited functionality.");
    processUserInput();
});

