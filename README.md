# Neru - A Simple Neural Network Implementation

Neru is a lightweight neural network library implemented in TypeScript. It provides a simple, educational implementation of feed-forward neural networks with backpropagation learning.

## Features

- Simple, understandable implementation of neural networks
- Feed-forward propagation
- Simplified backpropagation learning
- Configurable network architecture
- Sigmoid activation function

## Components

### Neuron

The basic building block of the neural network. Each neuron:
- Contains weights for each input connection
- Has a bias value
- Uses a sigmoid activation function
- Performs forward propagation to produce an output
- Can update its weights based on error during training

### Network

Orchestrates multiple neurons into a complete neural network. The network:
- Consists of input, hidden, and output layers
- Performs forward propagation through all layers
- Implements simplified backpropagation for training
- Calculates error metrics

## Usage Example

Here's how to use Neru to solve the NAND logic problem:

```typescript
import Network from './network.ts';

// Create a network with 2 inputs, 8 hidden neurons, and 1 output
const network = new Network(2, 8, 1);

// Training data for NAND gate
const trainingData = [
    { inputs: [0, 0], targets: [1] },
    { inputs: [0, 1], targets: [1] },
    { inputs: [1, 0], targets: [1] },
    { inputs: [1, 1], targets: [0] }
];

// Train the network
const learningRate = 0.1;
const epochs = 10000;

for (let i = 0; i < epochs; i++) {
    // Training code...
}

// Make predictions
const prediction = network.forward([1, 0]); // Should be close to 1
```

## How It Works

1. **Initialization**: Each neuron is created with random initial weights and biases
2. **Forward Propagation**: Input signals flow through the network, with each neuron computing its output
3. **Training**: 
   - The network compares its prediction with the expected output
   - It calculates the error
   - It updates weights and biases to reduce the error in future predictions
4. **Prediction**: Once trained, the network can make predictions on new inputs

## Getting Started

1. Clone the repository
2. Ensure TypeScript is installed
3. Run the example with:
   ```
   ts-node main.ts
   ```

## Example Output

When running the NAND gate example, you should see output similar to:

```
Training the network...
Epoch 0, Error: 0.247241, Learning rate: 0.1
Epoch 1000, Error: 0.001237, Learning rate: 0.01
...

Testing the network:
Input: [0,0], Expected: 1, Predicted: 0.997
Input: [0,1], Expected: 1, Predicted: 0.992
Input: [1,0], Expected: 1, Predicted: 0.991
Input: [1,1], Expected: 0, Predicted: 0.003

Final predictions:
0 NAND 0 = 0.997 (Rounded: 1)
0 NAND 1 = 0.992 (Rounded: 1)
1 NAND 0 = 0.991 (Rounded: 1)
1 NAND 1 = 0.003 (Rounded: 0)
```

## Project Structure

- `neuron.ts`: Implementation of a single neuron
- `network.ts`: Implementation of the neural network
- `main.ts`: Example usage solving the NAND problem

## Limitations

This is an educational implementation and has some limitations:
- Uses simplified backpropagation
- Limited to one hidden layer
- Only implements sigmoid activation function
- Not optimized for performance

