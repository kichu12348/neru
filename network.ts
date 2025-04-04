import Neuron from "./neuron";

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
        
        // Step 2: Calculate output layer gradients and update
        const outputGradients: number[] = [];
        for (let i = 0; i < this.outputLayer.length; i++) {
            const error = targets[i] - outputs[i];
            const derivative = outputs[i] * (1 - outputs[i]); 
            const gradient = error * derivative;
            outputGradients.push(gradient);
            this.outputLayer[i].update(hiddenOutputs, gradient, learningRate);
        }
        
        // Step 3: Calculate hidden layer gradients and update
        for (let i = 0; i < this.hiddenLayer.length; i++) {
            let hiddenError = 0;
            for (let j = 0; j < this.outputLayer.length; j++) {
                hiddenError += outputGradients[j] * this.outputLayer[j].getWeight(i);
            }
            const derivative = hiddenOutputs[i] * (1 - hiddenOutputs[i]);
            const gradient = hiddenError * derivative;
            this.hiddenLayer[i].update(inputs, gradient, learningRate);
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