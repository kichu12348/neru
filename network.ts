import Neuron from "./neuron.ts";

interface Network{
    forward(inputs: number[]): number[];
    train(inputs: number[], targets: number[], learningRate: number): number;
}

// Neural network class
class Network {
    private inputSize: number;
    private hiddenSize: number;
    private outputSize: number;
    private hiddenLayer: Neuron[];
    private outputLayer: Neuron[];
    
    /**
     * Create a neural network with one hidden layer
     * @param inputSize Number of input features
     * @param hiddenSize Number of neurons in hidden layer
     * @param outputSize Number of output neurons
     */
    constructor(inputSize: number, hiddenSize: number, outputSize: number) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        
        // Create hidden layer neurons
        this.hiddenLayer = [];
        for (let i = 0; i < hiddenSize; i++) {
            this.hiddenLayer.push(new Neuron(inputSize));
        }
        
        // Create output layer neurons
        this.outputLayer = [];
        for (let i = 0; i < outputSize; i++) {
            this.outputLayer.push(new Neuron(hiddenSize));
        }
    }
    
    /**
     * Make a prediction with the network
     * @param inputs Array of input values
     * @returns Array of output values
     */
    forward(inputs: number[]): number[] {
        // Step 1: Pass inputs through hidden layer
        const hiddenOutputs: number[] = [];
        for (let i = 0; i < this.hiddenLayer.length; i++) {
            hiddenOutputs.push(this.hiddenLayer[i].forward(inputs));
        }
        
        // Step 2: Pass hidden layer outputs to output layer
        const outputs: number[] = [];
        for (let i = 0; i < this.outputLayer.length; i++) {
            outputs.push(this.outputLayer[i].forward(hiddenOutputs));
        }
        
        return outputs;
    }
    
    /**
     * Train the network with a single example
     * @param inputs Input values
     * @param targets Expected output values
     * @param learningRate How quickly the network learns
     * @returns The error (difference between prediction and target)
     */
    train(inputs: number[], targets: number[], learningRate: number): number {
        // Step 1: Forward pass to get all outputs
        const hiddenOutputs: number[] = [];
        for (let i = 0; i < this.hiddenLayer.length; i++) {
            hiddenOutputs.push(this.hiddenLayer[i].forward(inputs));
        }
        
        const outputs: number[] = [];
        for (let i = 0; i < this.outputLayer.length; i++) {
            outputs.push(this.outputLayer[i].forward(hiddenOutputs));
        }
        
        // Step 2: Update output layer weights based on error
        for (let i = 0; i < this.outputLayer.length; i++) {
            const error = targets[i] - outputs[i]; // How far off was our guess?
            this.outputLayer[i].update(hiddenOutputs, error, learningRate);
        }
        
        // Step 3: Update hidden layer (simplified backpropagation)
        for (let i = 0; i < this.hiddenLayer.length; i++) {
            // Simple error propagation for demonstration
            let error = 0;
            for (let j = 0; j < targets.length; j++) {
                error += targets[j] - outputs[j];
            }
            error /= targets.length; // Average error
            
            this.hiddenLayer[i].update(inputs, error * 0.5, learningRate);
        }
        
        // Calculate and return total error
        let totalError = 0;
        for (let i = 0; i < outputs.length; i++) {
            totalError += Math.pow(targets[i] - outputs[i], 2); // Squared error
        }
        return totalError / outputs.length; // Mean squared error
    }
}

export default Network;
export type { Network }; // Export the Network interface