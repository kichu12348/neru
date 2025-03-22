class Neron {
    constructor(
        weights,
        bias
    ) {
        this.weights = weights;
        this.bias = bias;
        this.lastInputs = null;
        this.lastOutput = null;
        this.baseLearningRate = 0.1;
        this.learningRate = this.baseLearningRate;
        this.momentum = 0.9;
        this.prevWeightDeltas = Array(weights.length).fill(0);
        this.prevBiasDelta = 0;
        this.attempts = 0;  // Track training attempts for adaptive rates
    }

    // Reset learning rate to base value
    resetLearningRate() {
        this.learningRate = this.baseLearningRate;
        this.attempts = 0;
    }

    // Boost learning rate for reinforcement learning
    boostLearningRate(factor = 2.0) {
        this.learningRate = Math.min(0.5, this.baseLearningRate * factor);
        return this.learningRate;
    }

    // Adjust learning rate based on training attempts
    adjustLearningRate() {
        this.attempts++;
        // Increase learning rate if we've made multiple attempts to learn
        if (this.attempts > 3) {
            this.learningRate = Math.min(0.5, this.baseLearningRate * (1 + this.attempts * 0.1));
        }
        return this.learningRate;
    }

    activation(x) {
        // Faster sigmoid approximation with bound checking to avoid overflows
        if (x < -16) return 0;
        if (x > 16) return 1;
        return 1 / (1 + Math.exp(-x));
    }

    // Derivative of sigmoid function for backpropagation
    activationDerivative(x) {
        const activationValue = this.activation(x);
        return activationValue * (1 - activationValue);
    }

    output(inputs) {
        // Optimize the dot product calculation
        let sum = this.bias;
        const len = Math.min(inputs.length, this.weights.length);
        
        for(let i = 0; i < len; i++) {
            sum += inputs[i] * this.weights[i];
        }
        
        // Store inputs and output for training
        this.lastInputs = inputs;
        this.lastOutput = this.activation(sum);
        
        return this.lastOutput;
    }
    
    // Train neuron with error using momentum
    train(error, reinforcement = 1.0) {
        // Apply reinforcement to error for more aggressive learning
        const scaledError = error * reinforcement;
        const outputDerivative = scaledError * this.lastOutput * (1 - this.lastOutput);
        
        // Update weights with momentum
        for (let i = 0; i < this.weights.length; i++) {
            const weightDelta = this.learningRate * outputDerivative * this.lastInputs[i];
            this.weights[i] += weightDelta + (this.momentum * this.prevWeightDeltas[i]);
            this.prevWeightDeltas[i] = weightDelta;
        }
        
        // Update bias with momentum
        const biasDelta = this.learningRate * outputDerivative;
        this.bias += biasDelta + (this.momentum * this.prevBiasDelta);
        this.prevBiasDelta = biasDelta;
        
        return outputDerivative;
    }
}

module.exports = Neron;





