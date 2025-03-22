const Neron = require('./neron');

class NeuralNetwork {
    constructor(inputSize, hiddenSizes, outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        
        // Support for multiple hidden layers
        this.layers = [];
        
        // Reinforcement learning components
        this.associationStrengths = new Map(); // Track word pair learning strength
        this.trainingAttempts = new Map(); // Track training attempts for each association
        this.reinforcementFactor = 1.0; // Base reinforcement factor
        
        // Xavier/Glorot initialization for weights
        function initializeWeight(fanIn, fanOut) {
            // Xavier/Glorot initialization: sqrt(6/(fanIn + fanOut))
            const scale = Math.sqrt(6 / (fanIn + fanOut));
            return (Math.random() * 2 - 1) * scale;
        }
        
        // Add hidden layers
        let prevLayerSize = inputSize;
        for (let layerSize of hiddenSizes) {
            const layer = [];
            for (let i = 0; i < layerSize; i++) {
                const weights = Array(prevLayerSize).fill()
                    .map(() => initializeWeight(prevLayerSize, layerSize));
                layer.push(new Neron(weights, initializeWeight(prevLayerSize, layerSize)));
            }
            this.layers.push(layer);
            prevLayerSize = layerSize;
        }
        
        // Add output layer
        this.outputLayer = [];
        for (let i = 0; i < outputSize; i++) {
            const weights = Array(prevLayerSize).fill()
                .map(() => initializeWeight(prevLayerSize, outputSize));
            this.outputLayer.push(new Neron(weights, initializeWeight(prevLayerSize, outputSize)));
        }
    }
    
    // Record and track association strength
    trackAssociation(inputWord, outputWord, success) {
        const key = `${inputWord}:${outputWord}`;
        
        // Initialize tracking if needed
        if (!this.associationStrengths.has(key)) {
            this.associationStrengths.set(key, 0.5); // Start at medium strength
            this.trainingAttempts.set(key, 0);
        }
        
        // Update tracking
        const attempts = this.trainingAttempts.get(key) + 1;
        this.trainingAttempts.set(key, attempts);
        
        // Update strength based on success/failure
        let currentStrength = this.associationStrengths.get(key);
        if (success) {
            currentStrength = currentStrength + (1 - currentStrength) * 0.2; // Increase
        } else {
            currentStrength = Math.max(0.1, currentStrength * 0.8); // Decrease, but keep some minimum
        }
        
        this.associationStrengths.set(key, currentStrength);
        return { attempts, strength: currentStrength };
    }
    
    // Calculate reinforcement factor for an association
    calculateReinforcement(inputWord, outputWord) {
        const key = `${inputWord}:${outputWord}`;
        if (!this.associationStrengths.has(key)) return 1.0;
        
        // Base reinforcement on attempts and current strength
        const attempts = this.trainingAttempts.get(key);
        const strength = this.associationStrengths.get(key);
        
        // Higher reinforcement for low strength + high attempts
        if (attempts > 5 && strength < 0.5) {
            return 5.0; // Strong reinforcement for persistent failure
        } else if (attempts > 3) {
            return 2.0; // Medium reinforcement
        }
        
        return 1.0; // Default reinforcement
    }
    
    // Resize the network when vocabulary size changes
    resize(newInputSize, newOutputSize) {
        // Resize first hidden layer input weights
        if (newInputSize > this.inputSize) {
            this.layers[0].forEach(neuron => {
                const additionalWeights = Array(newInputSize - this.inputSize)
                    .fill()
                    .map(() => Math.random() * 0.2 - 0.1); // Small random values
                neuron.weights = [...neuron.weights, ...additionalWeights];
            });
        }
        
        // Resize output layer
        if (newOutputSize > this.outputSize) {
            // Add neurons to output layer
            for (let i = this.outputSize; i < newOutputSize; i++) {
                const weights = Array(this.layers[this.layers.length - 1].length).fill().map(() => Math.random() * 2 - 1);
                this.outputLayer.push(new Neron(weights, Math.random() * 2 - 1));
            }
            
            this.outputSize = newOutputSize;
        }
        
        this.inputSize = newInputSize;
    }
    
    feedForward(inputs, temperature = 1.0) {
        // Ensure inputs have correct size
        if (inputs.length < this.inputSize) {
            const paddedInputs = [...inputs, ...Array(this.inputSize - inputs.length).fill(0)];
            inputs = paddedInputs;
        } else if (inputs.length > this.inputSize) {
            inputs = inputs.slice(0, this.inputSize);
        }
        
        // Feed through all hidden layers with optimized calculation
        let layerInput = inputs;
        for (const layer of this.layers) {
            layerInput = layer.map(neuron => neuron.output(layerInput));
        }
        
        // Get outputs from output layer
        const rawOutputs = this.outputLayer.map(neuron => neuron.output(layerInput));
        
        // Apply temperature (higher = more random, lower = more conservative)
        if (temperature !== 1.0) {
            // Scale logits by temperature with safety checks
            const logits = rawOutputs.map(p => {
                const clipped = Math.max(0.0001, Math.min(0.9999, p));
                return Math.log(clipped / (1 - clipped)) / temperature;
            });
            // Convert back to probabilities with safety checks
            return logits.map(l => {
                const expL = Math.exp(l);
                return expL / (1 + expL);
            });
        }
        
        return rawOutputs;
    }
    
    // Train with reinforcement learning
    train(inputs, expectedOutputs, inputWord = '', outputWord = '', reinforcementFactor = 1.0) {
        try {
            // Ensure inputs and outputs have correct dimensions
            if (inputs.length !== this.inputSize || expectedOutputs.length !== this.outputSize) {
                this.resize(Math.max(inputs.length, this.inputSize), 
                           Math.max(expectedOutputs.length, this.outputSize));
            }
            
            // Forward pass
            let currentInput = inputs;
            
            // Process through hidden layers
            for (const layer of this.layers) {
                currentInput = layer.map(neuron => neuron.output(currentInput));
            }
            
            // Process through output layer
            const finalOutputs = this.outputLayer.map(neuron => neuron.output(currentInput));
            
            // Calculate error
            let totalError = 0;
            for (let i = 0; i < expectedOutputs.length; i++) {
                const err = Math.pow(expectedOutputs[i] - finalOutputs[i], 2);
                if (!isNaN(err) && isFinite(err)) {
                    totalError += err;
                }
            }
            totalError /= expectedOutputs.length;
            
            // Calculate reinforcement based on association tracking
            if (inputWord && outputWord) {
                const specificReinforcement = this.calculateReinforcement(inputWord, outputWord);
                reinforcementFactor = Math.max(reinforcementFactor, specificReinforcement);
            }
            
            // Output layer error and training with reinforcement
            const outputDeltas = [];
            for (let i = 0; i < finalOutputs.length; i++) {
                const error = expectedOutputs[i] - finalOutputs[i];
                outputDeltas.push(this.outputLayer[i].train(error, reinforcementFactor));
            }
            
            // Backpropagation through hidden layers with reinforcement
            let nextLayerDeltas = outputDeltas;
            let nextLayerWeights = this.outputLayer.map(n => n.weights);
            
            // Go through layers in reverse order
            for (let l = this.layers.length - 1; l >= 0; l--) {
                const currentLayer = this.layers[l];
                const layerDeltas = [];
                
                for (let i = 0; i < currentLayer.length; i++) {
                    let error = 0;
                    // Sum errors from each neuron in the next layer
                    for (let j = 0; j < nextLayerDeltas.length; j++) {
                        if (i < nextLayerWeights[j].length) {
                            error += nextLayerDeltas[j] * nextLayerWeights[j][i];
                        }
                    }
                    
                    // Train with reinforcement
                    layerDeltas.push(currentLayer[i].train(error, reinforcementFactor));
                }
                
                // Prepare for next iteration
                nextLayerDeltas = layerDeltas;
                nextLayerWeights = currentLayer.map(n => n.weights);
            }
            
            return totalError;
            
        } catch (error) {
            console.error("Training error:", error);
            return 0;
        }
    }
    
    // Reinforcement learning with verification
    trainUntilLearned(inputs, expectedOutputs, inputWord, outputWord, maxAttempts = 30, targetError = 0.0001) {
        // First check if the current output matches expectation
        const currentOutput = this.feedForward(inputs);
        const currentPrediction = this.findHighestIndex(currentOutput);
        const targetIndex = this.findHighestIndex(expectedOutputs);
        
        if (currentPrediction === targetIndex) {
            // Already correct, just strengthen the association
            this.trackAssociation(inputWord, outputWord, true);
            return { error: 0, attempts: 0, success: true };
        }
        
        // Need to train
        let attempts = 0;
        let lastError = 1.0;
        let success = false;
        
        // Store an indicator if this is a difficult association
        const isDifficult = this.isConflictingAssociation(inputWord, outputWord);
        
        // Increase learning rate for this specific training
        this.adjustNeuronLearningRates(isDifficult ? 4.0 : 2.0);
        
        // Train multiple times with increasing reinforcement
        while (attempts < maxAttempts) {
            attempts++;
            
            // Calculate dynamic reinforcement based on attempts and difficulty
            const baseMultiplier = isDifficult ? 2.0 : 1.0;
            const dynamicReinforcement = baseMultiplier * (1.0 + attempts * 0.5);
            
            // Train with reinforcement
            lastError = this.train(inputs, expectedOutputs, inputWord, outputWord, dynamicReinforcement);
            
            // Check if we've successfully learned
            const output = this.feedForward(inputs);
            const prediction = this.findHighestIndex(output);
            
            if (prediction === targetIndex || lastError < targetError) {
                success = true;
                break;
            }
            
            // Every 3 attempts, try a more aggressive approach for difficult associations
            if (attempts % 3 === 0) {
                const forceFactor = isDifficult ? 20.0 : 10.0;
                this.forceAssociation(inputs, expectedOutputs, forceFactor);
                
                // Check if force helped
                const checkOutput = this.feedForward(inputs);
                const checkPrediction = this.findHighestIndex(checkOutput);
                
                if (checkPrediction === targetIndex) {
                    success = true;
                    break;
                }
            }
        }
        
        // Reset learning rates after intensive training
        this.resetNeuronLearningRates();
        
        // Track this association attempt
        this.trackAssociation(inputWord, outputWord, success);
        
        // If still unsuccessful, apply a direct connection technique as last resort
        if (!success && attempts >= maxAttempts) {
            console.log(`  Using direct connection technique for "${inputWord}" → "${outputWord}"`);
            this.createDirectConnection(inputs, expectedOutputs, 50.0);
            
            // Check if it worked
            const finalOutput = this.feedForward(inputs);
            const finalPrediction = this.findHighestIndex(finalOutput);
            success = finalPrediction === targetIndex;
        }
        
        return { error: lastError, attempts, success };
    }
    
    // Force an association by directly modifying weights
    forceAssociation(inputs, expectedOutputs, factor = 5.0) {
        // Identify the input and output indices with highest values
        const inputIndex = this.findHighestIndex(inputs);
        const outputIndex = this.findHighestIndex(expectedOutputs);
        
        // Only proceed if we found valid indices
        if (inputIndex >= 0 && outputIndex >= 0) {
            // Modify weights more directly to strengthen the connection
            // This is a more aggressive approach when normal training isn't working
            
            // For first hidden layer, strengthen connection from input to all neurons
            for (const neuron of this.layers[0]) {
                neuron.weights[inputIndex] += 0.5 * factor;
            }
            
            // For output layer, strengthen connection to target output
            const lastHiddenLayer = this.layers[this.layers.length - 1];
            for (let i = 0; i < lastHiddenLayer.length; i++) {
                // Increase weights from all hidden neurons to target output
                this.outputLayer[outputIndex].weights[i] += 0.3 * factor;
                
                // Reduce weights to competing outputs
                for (let j = 0; j < this.outputLayer.length; j++) {
                    if (j !== outputIndex) {
                        this.outputLayer[j].weights[i] -= 0.1 * factor;
                    }
                }
            }
        }
    }
    
    // Find highest value index in an array
    findHighestIndex(array) {
        if (!array || array.length === 0) return -1;
        
        let maxIndex = 0;
        let maxValue = array[0];
        
        for (let i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        
        return maxIndex;
    }
    
    // Adjust learning rates of all neurons
    adjustNeuronLearningRates(factor = 1.0) {
        // Adjust hidden layers
        for (const layer of this.layers) {
            for (const neuron of layer) {
                neuron.boostLearningRate(factor);
            }
        }
        
        // Adjust output layer
        for (const neuron of this.outputLayer) {
            neuron.boostLearningRate(factor);
        }
    }
    
    // Reset learning rates of all neurons
    resetNeuronLearningRates() {
        // Reset hidden layers
        for (const layer of this.layers) {
            for (const neuron of layer) {
                neuron.resetLearningRate();
            }
        }
        
        // Reset output layer
        for (const neuron of this.outputLayer) {
            neuron.resetLearningRate();
        }
    }
    
    // Train intensively with early stopping and reinforcement
    trainIntensively(inputs, expectedOutputs, inputWord = '', outputWord = '', maxEpochs = 500, targetError = 0.0001) {
        if (inputWord && outputWord) {
            return this.trainUntilLearned(inputs, expectedOutputs, inputWord, outputWord, maxEpochs, targetError).error;
        }
        
        // For non-word-specific training, use standard intensive training
        let bestError = Number.MAX_VALUE;
        let noImprovementCount = 0;
        
        for (let i = 0; i < maxEpochs; i++) {
            const error = this.train(inputs, expectedOutputs);
            
            // Check for improvement
            if (error < bestError) {
                bestError = error;
                noImprovementCount = 0;
            } else {
                noImprovementCount++;
            }
            
            // Early stopping conditions
            if (bestError < targetError || noImprovementCount > 20) {
                return bestError;
            }
        }
        
        return bestError;
    }
    
    // Check if this is a conflicting association (same input → different outputs in training)
    isConflictingAssociation(inputWord, targetWord) {
        // Check if this input word has been associated with multiple different outputs
        let hasConflict = false;
        let otherTarget = null;
        
        // Count number of different target words for this input
        const targetWords = new Set();
        
        for (const [key, strength] of this.associationStrengths.entries()) {
            if (key.startsWith(`${inputWord}:`)) {
                const parts = key.split(':');
                if (parts.length === 2) {
                    const target = parts[1];
                    if (target !== targetWord) {
                        targetWords.add(target);
                        otherTarget = target; // Store one example of conflicting target
                    }
                }
            }
        }
        
        // If we found more than 0 different targets, we have a conflict
        hasConflict = targetWords.size > 0;
        
        if (hasConflict) {
            console.log(`  Conflict detected: "${inputWord}" has previous target "${otherTarget}" vs new "${targetWord}"`);
        }
        
        return hasConflict;
    }
    
    // Create direct connection by heavily modifying weights
    createDirectConnection(inputs, expectedOutputs, factor = 50.0) {
        // Identify the input and output indices with highest values
        const inputIndex = this.findHighestIndex(inputs);
        const outputIndex = this.findHighestIndex(expectedOutputs);
        
        // Only proceed if we found valid indices
        if (inputIndex >= 0 && outputIndex >= 0) {
            // Directly modify weights in the first layer
            for (const neuron of this.layers[0]) {
                // Increase weight from input to first hidden layer neurons
                neuron.weights[inputIndex] += factor * 0.5;
            }
            
            // Create a strong path through the hidden layers
            let prevLayerSize = this.layers[0].length;
            let prevLayerNeurons = this.layers[0];
            
            // For middle layers, create a path
            for (let i = 1; i < this.layers.length; i++) {
                const currentLayer = this.layers[i];
                
                // Choose a neuron in this layer to be part of the direct path
                const pathNeuronIndex = i % currentLayer.length;
                
                // Strengthen connections from all previous layer neurons to this one
                for (let j = 0; j < prevLayerSize; j++) {
                    currentLayer[pathNeuronIndex].weights[j] += factor * 0.3;
                }
                
                prevLayerSize = currentLayer.length;
                prevLayerNeurons = currentLayer;
            }
            
            // For output layer, strengthen connection to target output
            for (let i = 0; i < prevLayerSize; i++) {
                // Increase weights to target output
                this.outputLayer[outputIndex].weights[i] += factor;
                
                // Decrease weights to competing outputs
                for (let j = 0; j < this.outputLayer.length; j++) {
                    if (j !== outputIndex) {
                        this.outputLayer[j].weights[i] -= factor * 0.5;
                    }
                }
            }
        }
    }
}

module.exports = NeuralNetwork;
