const EmbeddingLayer = require('./embeddingLayer');
const LSTMLayer = require('./lstmLayer');

/**
 * Advanced neural network with embeddings, LSTM layers and Adam optimizer
 */
class AdvancedNetwork {
    /**
     * Create a new advanced neural network
     * @param {Object} config - Configuration
     */
    constructor(config) {
        this.vocabSize = config.vocabSize;
        this.embeddingDim = config.embeddingDim || 50;
        this.hiddenSizes = config.hiddenSizes || [64, 32];
        this.dropout = config.dropout || 0.2;
        this.learningRate = config.learningRate || 0.001;
        this.beta1 = config.beta1 || 0.9; // Adam optimizer param
        this.beta2 = config.beta2 || 0.999; // Adam optimizer param
        this.epsilon = config.epsilon || 1e-8; // Adam optimizer param
        this.useEmbeddings = config.useEmbeddings !== false;
        this.useLSTM = config.useLSTM !== false;
        
        this.layers = [];
        this.embeddings = null;
        this.wordEmbeddingDb = config.wordEmbeddingDb;
        
        this.adamOptParams = {
            t: 0,           // Time step
            m: [],          // First moment vectors
            v: []           // Second moment vectors
        };
        
        this.buildNetwork();
    }
    
    /**
     * Build network architecture
     */
    buildNetwork() {
        // Add embedding layer if enabled
        if (this.useEmbeddings) {
            this.embeddings = new EmbeddingLayer(
                this.vocabSize, 
                this.embeddingDim, 
                this.wordEmbeddingDb
            );
            
            // First layer size depends on embeddings
            let inputSize = this.embeddingDim;
            
            // Add LSTM or dense layers
            if (this.useLSTM) {
                // Add LSTM layers
                for (let i = 0; i < this.hiddenSizes.length; i++) {
                    const lstmLayer = new LSTMLayer(
                        inputSize, 
                        this.hiddenSizes[i], 
                        this.dropout
                    );
                    this.layers.push(lstmLayer);
                    inputSize = this.hiddenSizes[i];
                }
            } else {
                // Add dense layers
                for (let i = 0; i < this.hiddenSizes.length; i++) {
                    const denseLayer = {
                        type: 'dense',
                        inputSize: inputSize,
                        outputSize: this.hiddenSizes[i],
                        weights: Array(this.hiddenSizes[i]).fill().map(() => 
                            Array(inputSize).fill().map(() => 
                                (Math.random() * 2 - 1) * Math.sqrt(6 / (inputSize + this.hiddenSizes[i]))
                            )
                        ),
                        biases: Array(this.hiddenSizes[i]).fill(0),
                        lastInput: null,
                        lastOutput: null,
                        dropoutRate: this.dropout
                    };
                    this.layers.push(denseLayer);
                    inputSize = this.hiddenSizes[i];
                }
            }
            
            // Output layer
            const outputLayer = {
                type: 'dense',
                inputSize: inputSize,
                outputSize: this.vocabSize,
                weights: Array(this.vocabSize).fill().map(() => 
                    Array(inputSize).fill().map(() => 
                        (Math.random() * 2 - 1) * Math.sqrt(6 / (inputSize + this.vocabSize))
                    )
                ),
                biases: Array(this.vocabSize).fill(0),
                lastInput: null,
                lastOutput: null,
                dropoutRate: 0 // No dropout in output layer
            };
            this.layers.push(outputLayer);
            
            // Initialize Adam optimizer params
            this.initAdamParams();
        }
    }
    
    /**
     * Initialize Adam optimizer parameters
     */
    initAdamParams() {
        this.adamOptParams.m = this.layers.map(layer => {
            if (layer.type === 'dense') {
                return {
                    weights: Array(layer.outputSize).fill().map(() => 
                        Array(layer.inputSize).fill(0)
                    ),
                    biases: Array(layer.outputSize).fill(0)
                };
            }
            return null; // For LSTM layers, handled separately
        });
        
        this.adamOptParams.v = this.layers.map(layer => {
            if (layer.type === 'dense') {
                return {
                    weights: Array(layer.outputSize).fill().map(() => 
                        Array(layer.inputSize).fill(0)
                    ),
                    biases: Array(layer.outputSize).fill(0)
                };
            }
            return null; // For LSTM layers, handled separately
        });
    }
    
    /**
     * ReLU activation function
     * @param {number} x - Input value
     * @returns {number} - ReLU output
     */
    relu(x) {
        return Math.max(0, x);
    }
    
    /**
     * Sigmoid activation function
     * @param {number} x - Input value
     * @returns {number} - Sigmoid output
     */
    sigmoid(x) {
        // Avoid numerical instability
        if (x < -16) return 0;
        if (x > 16) return 1;
        return 1 / (1 + Math.exp(-x));
    }
    
    /**
     * Softmax activation function
     * @param {Array<number>} x - Input array
     * @returns {Array<number>} - Softmax output
     */
    softmax(x) {
        // For numerical stability, subtract max value
        const max = Math.max(...x);
        const expValues = x.map(val => Math.exp(val - max));
        const sumExp = expValues.reduce((acc, val) => acc + val, 0);
        return expValues.map(val => val / sumExp);
    }
    
    /**
     * Apply dropout to a vector
     * @param {Array<number>} x - Input vector
     * @param {number} rate - Dropout rate (0-1)
     * @param {boolean} training - Whether in training mode
     * @returns {Array<number>} - Output with dropout applied
     */
    applyDropout(x, rate, training) {
        if (!training || rate === 0) return x;
        
        // Create dropout mask
        const mask = Array(x.length).fill().map(() => 
            Math.random() > rate ? 1 / (1 - rate) : 0
        );
        
        // Apply mask
        return x.map((val, i) => val * mask[i]);
    }
    
    /**
     * Forward pass through dense layer
     * @param {Object} layer - Layer object
     * @param {Array<number>} input - Input vector
     * @param {boolean} training - Whether in training mode
     * @returns {Array<number>} - Output vector
     */
    forwardDenseLayer(layer, input, training) {
        layer.lastInput = input;
        
        // Compute weighted sum and add bias
        const preActivation = Array(layer.outputSize).fill(0);
        for (let i = 0; i < layer.outputSize; i++) {
            for (let j = 0; j < layer.inputSize; j++) {
                preActivation[i] += layer.weights[i][j] * input[j];
            }
            preActivation[i] += layer.biases[i];
        }
        
        // Apply activation function (ReLU for hidden, softmax for output)
        let output;
        if (layer === this.layers[this.layers.length - 1]) {
            // Output layer - softmax
            output = this.softmax(preActivation);
        } else {
            // Hidden layer - ReLU
            output = preActivation.map(x => this.relu(x));
            
            // Apply dropout
            output = this.applyDropout(output, layer.dropoutRate, training);
        }
        
        layer.lastOutput = output;
        return output;
    }
    
    /**
     * Forward pass through the network
     * @param {Array<number>} input - One-hot encoded input
     * @param {boolean} training - Whether in training mode
     * @param {number} temperature - Temperature for sampling (1.0 is normal)
     * @returns {Array<number>} - Output probabilities
     */
    forward(input, training = false, temperature = 1.0) {
        // Initialize state for LSTM layers if needed
        if (this.useLSTM && training) {
            this.layers.forEach(layer => {
                if (layer instanceof LSTMLayer) {
                    layer.resetState();
                }
            });
        }
        
        // Process through embedding layer if enabled
        let currentOutput;
        if (this.useEmbeddings) {
            currentOutput = this.embeddings.forward(input);
        } else {
            currentOutput = input;
        }
        
        // Process through each layer
        for (const layer of this.layers) {
            if (layer instanceof LSTMLayer) {
                currentOutput = layer.forward(currentOutput, training);
            } else {
                currentOutput = this.forwardDenseLayer(layer, currentOutput, training);
            }
        }
        
        // Apply temperature scaling if needed
        if (temperature !== 1.0 && currentOutput.length > 0) {
            // Apply temperature to logits
            const logits = currentOutput.map(p => {
                const clipped = Math.max(0.0001, Math.min(0.9999, p));
                return Math.log(clipped / (1 - clipped)) / temperature;
            });
            
            // Apply softmax to adjusted logits
            currentOutput = this.softmax(logits);
        }
        
        return currentOutput;
    }
    
    /**
     * Calculate categorical cross-entropy loss
     * @param {Array<number>} predicted - Predicted probabilities
     * @param {Array<number>} target - Target one-hot encoded vector
     * @returns {number} - Cross-entropy loss
     */
    crossEntropyLoss(predicted, target) {
        let loss = 0;
        
        // Find the target index (should be one-hot)
        const targetIndex = target.findIndex(v => v > 0);
        
        if (targetIndex !== -1) {
            // Safe log to avoid log(0)
            const safeLog = val => Math.log(Math.max(val, 1e-15));
            
            // Single target loss
            loss = -safeLog(predicted[targetIndex]);
        } else {
            // Full cross-entropy loss
            for (let i = 0; i < target.length; i++) {
                if (target[i] > 0) {
                    const safeLog = Math.log(Math.max(predicted[i], 1e-15));
                    loss -= target[i] * safeLog;
                }
            }
        }
        
        return loss;
    }
    
    /**
     * Calculate gradients for output layer with cross-entropy loss
     * @param {Array<number>} predicted - Predicted probabilities
     * @param {Array<number>} target - Target one-hot encoded vector
     * @returns {Array<number>} - Gradients
     */
    outputGradients(predicted, target) {
        // For cross-entropy loss with softmax, gradient is (predicted - target)
        return predicted.map((p, i) => p - target[i]);
    }
    
    /**
     * Backward pass through dense layer with Adam optimizer
     * @param {Object} layer - Layer object
     * @param {Array<number>} gradOutput - Gradient from next layer
     * @param {number} layerIndex - Index of layer in network
     * @returns {Array<number>} - Gradient to be propagated to previous layer
     */
    backwardDenseLayer(layer, gradOutput, layerIndex) {
        const input = layer.lastInput;
        const output = layer.lastOutput;
        
        // Calculate gradients for weights and biases
        const gradWeights = Array(layer.outputSize).fill().map(() => 
            Array(layer.inputSize).fill(0)
        );
        
        const gradBiases = Array(layer.outputSize).fill(0);
        
        // Calculate gradients for inputs to pass to previous layer
        const gradInput = Array(layer.inputSize).fill(0);
        
        // If last layer, apply ReLU gradient
        const gradActivation = layer === this.layers[this.layers.length - 1] ? 
            gradOutput : 
            gradOutput.map((g, i) => output[i] > 0 ? g : 0);
        
        // Compute gradients
        for (let i = 0; i < layer.outputSize; i++) {
            gradBiases[i] = gradActivation[i];
            
            for (let j = 0; j < layer.inputSize; j++) {
                gradWeights[i][j] = gradActivation[i] * input[j];
                gradInput[j] += gradActivation[i] * layer.weights[i][j];
            }
        }
        
        // Update with Adam optimizer
        this.adamUpdate(layer, gradWeights, gradBiases, layerIndex);
        
        return gradInput;
    }
    
    /**
     * Adam optimizer update
     * @param {Object} layer - Layer object
     * @param {Array<Array<number>>} gradWeights - Weight gradients
     * @param {Array<number>} gradBiases - Bias gradients
     * @param {number} layerIndex - Index of layer in network
     */
    adamUpdate(layer, gradWeights, gradBiases, layerIndex) {
        if (layer.type !== 'dense') return;
        
        this.adamOptParams.t += 1;
        const t = this.adamOptParams.t;
        const lr = this.learningRate;
        const beta1 = this.beta1;
        const beta2 = this.beta2;
        const epsilon = this.epsilon;
        
        const m = this.adamOptParams.m[layerIndex];
        const v = this.adamOptParams.v[layerIndex];
        
        // Update biases
        for (let i = 0; i < layer.outputSize; i++) {
            // Update first moment
            m.biases[i] = beta1 * m.biases[i] + (1 - beta1) * gradBiases[i];
            
            // Update second moment
            v.biases[i] = beta2 * v.biases[i] + (1 - beta2) * gradBiases[i] * gradBiases[i];
            
            // Bias correction
            const mCorrected = m.biases[i] / (1 - Math.pow(beta1, t));
            const vCorrected = v.biases[i] / (1 - Math.pow(beta2, t));
            
            // Update parameters
            layer.biases[i] -= lr * mCorrected / (Math.sqrt(vCorrected) + epsilon);
        }
        
        // Update weights
        for (let i = 0; i < layer.outputSize; i++) {
            for (let j = 0; j < layer.inputSize; j++) {
                // Update first moment
                m.weights[i][j] = beta1 * m.weights[i][j] + (1 - beta1) * gradWeights[i][j];
                
                // Update second moment
                v.weights[i][j] = beta2 * v.weights[i][j] + (1 - beta2) * gradWeights[i][j] * gradWeights[i][j];
                
                // Bias correction
                const mCorrected = m.weights[i][j] / (1 - Math.pow(beta1, t));
                const vCorrected = v.weights[i][j] / (1 - Math.pow(beta2, t));
                
                // Update parameters
                layer.weights[i][j] -= lr * mCorrected / (Math.sqrt(vCorrected) + epsilon);
            }
        }
    }
    
    /**
     * Train the network on a single example
     * @param {Array<number>} input - One-hot encoded input
     * @param {Array<number>} target - Target one-hot encoded vector
     * @returns {number} - Loss value
     */
    train(input, target) {
        // Forward pass
        const predicted = this.forward(input, true);
        
        // Calculate loss
        const loss = this.crossEntropyLoss(predicted, target);
        
        // Calculate output gradients
        let gradients = this.outputGradients(predicted, target);
        
        // Backward pass through layers in reverse order
        for (let i = this.layers.length - 1; i >= 0; i--) {
            const layer = this.layers[i];
            if (layer instanceof LSTMLayer) {
                gradients = layer.backward(gradients, this.learningRate);
            } else {
                gradients = this.backwardDenseLayer(layer, gradients, i);
            }
        }
        
        // Update embeddings if enabled
        if (this.useEmbeddings) {
            this.embeddings.backward(input, gradients, this.learningRate * 0.1);
        }
        
        return loss;
    }
    
    /**
     * Train the network on multiple examples
     * @param {Array<{input: Array<number>, target: Array<number>}>} examples - Training examples
     * @param {Object} options - Training options
     * @returns {number} - Average loss
     */
    trainBatch(examples, options = {}) {
        const epochs = options.epochs || 1;
        const miniBatchSize = options.batchSize || 32;
        const validationData = options.validationData;
        const earlyStoppingPatience = options.earlyStoppingPatience || 5;
        
        let bestLoss = Infinity;
        let patienceCounter = 0;
        
        // Train for multiple epochs
        for (let epoch = 0; epoch < epochs; epoch++) {
            // Shuffle examples
            const shuffled = [...examples].sort(() => Math.random() - 0.5);
            
            // Process in mini-batches
            let totalLoss = 0;
            
            for (let i = 0; i < shuffled.length; i += miniBatchSize) {
                const batch = shuffled.slice(i, i + miniBatchSize);
                
                // Train on mini-batch
                for (const example of batch) {
                    const loss = this.train(example.input, example.target);
                    totalLoss += loss;
                }
            }
            
            // Calculate average loss
            const avgLoss = totalLoss / shuffled.length;
            
            // Validation
            if (validationData) {
                const validLoss = this.validate(validationData);
                
                // Early stopping
                if (validLoss < bestLoss) {
                    bestLoss = validLoss;
                    patienceCounter = 0;
                } else {
                    patienceCounter++;
                    if (patienceCounter >= earlyStoppingPatience) {
                        console.log(`Early stopping at epoch ${epoch + 1}`);
                        break;
                    }
                }
                
                // Log metrics
                if (options.verbose && epoch % 5 === 0) {
                    console.log(`Epoch ${epoch + 1}/${epochs}, Loss: ${avgLoss.toFixed(6)}, Val Loss: ${validLoss.toFixed(6)}`);
                }
            } else if (options.verbose && epoch % 5 === 0) {
                console.log(`Epoch ${epoch + 1}/${epochs}, Loss: ${avgLoss.toFixed(6)}`);
            }
        }
        
        // Return final loss
        return this.validate(examples);
    }
    
    /**
     * Validate the network on examples
     * @param {Array<{input: Array<number>, target: Array<number>}>} examples - Validation examples
     * @returns {number} - Average loss
     */
    validate(examples) {
        let totalLoss = 0;
        
        // Validate on each example
        for (const example of examples) {
            const predicted = this.forward(example.input, false);
            const loss = this.crossEntropyLoss(predicted, example.target);
            totalLoss += loss;
        }
        
        // Return average loss
        return totalLoss / examples.length;
    }
    
    /**
     * Resize the network for larger vocabulary
     * @param {number} newVocabSize - New vocabulary size
     */
    resize(newVocabSize) {
        if (newVocabSize <= this.vocabSize) return;
        
        // Resize embedding layer
        if (this.useEmbeddings) {
            this.embeddings.resize(newVocabSize);
        }
        
        // Resize output layer
        const outputLayer = this.layers[this.layers.length - 1];
        const additionalWeights = Array(newVocabSize - this.vocabSize).fill().map(() => 
            Array(outputLayer.inputSize).fill().map(() => 
                (Math.random() * 0.2 - 0.1)
            )
        );
        
        outputLayer.weights = [...outputLayer.weights, ...additionalWeights];
        outputLayer.biases = [...outputLayer.biases, ...Array(newVocabSize - this.vocabSize).fill(0)];
        outputLayer.outputSize = newVocabSize;
        
        // Update vocabulary size
        this.vocabSize = newVocabSize;
        
        // Reinitialize Adam parameters
        this.initAdamParams();
    }
    
    /**
     * Load pre-trained word embeddings
     * @param {Map<string, number>} vocabulary - Map of word to index
     */
    async loadEmbeddings(vocabulary) {
        if (this.useEmbeddings && this.embeddings) {
            return this.embeddings.loadPretrainedEmbeddings(vocabulary);
        }
        return false;
    }
}

module.exports = AdvancedNetwork;
