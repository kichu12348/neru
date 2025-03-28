interface Neuron {
  forward(inputs: number[]): number;
  update(inputs: number[], gradient: number, learningRate: number): void;
  getWeight(index: number): number;
}

class Neuron {
  private weights: number[];
  private bias: number;

  constructor(inputSize: number) {
    // Initialize random weights and bias
    this.weights = Array(inputSize)
      .fill(0)
      .map(() => Math.random() * 2 - 1); // Random between -1 and 1
    this.bias = Math.random() * 2 - 1; // value between -1 and 1
  }
  #activate(x: number) { // # means private
    return 1 / (1 + Math.exp(-x)); // Sigmoid function 1/(1+e^-x) gives value between 0 and 1
  }
  // Forward propagation - calculate output
  forward(inputs:number[]) {
    if (inputs.length !== this.weights.length)
      throw new Error("Puck you >:( \nInput size does not match weight size");

    // Calculate weighted sum
    let sum = this.bias;
    for (let i = 0; i < inputs.length; i++) {
      sum += inputs[i] * this.weights[i];
    }

    return this.#activate(sum);
  }

  // Get weight at specific index (needed for backpropagation)
  getWeight(index: number): number {
    if (index < 0 || index >= this.weights.length) {
      throw new Error("Weight index out of bounds");
    }
    return this.weights[index];
  }

  // Update weights and bias
  update(inputs: number[], gradient: number, learningRate: number) {
    this.bias += learningRate * gradient;
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] += learningRate * gradient * inputs[i];
    }
  }
}

// Export the Neuron class
export default Neuron;
export type { Neuron }; // Export the Neuron interface

