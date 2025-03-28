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

Here's how to use Neru to solve the 3-input NAND logic problem:

```typescript
import Network from './network.ts';

// Create a network with 3 inputs, 8 hidden neurons, and 1 output
const network = new Network(3, 8, 1);

// Training data for 3-input NAND gate
const trainingData = [
    { inputs: [0, 0, 0], targets: [1] },
    { inputs: [0, 0, 1], targets: [1] },
    { inputs: [0, 1, 0], targets: [1] },
    { inputs: [0, 1, 1], targets: [0] },
    { inputs: [1, 0, 0], targets: [1] },
    { inputs: [1, 0, 1], targets: [1] },
    { inputs: [1, 1, 0], targets: [1] },
    { inputs: [1, 1, 1], targets: [0] }
];

// Train the network
let learningRate = 0.3;
const epochs = 15000;
const initialLearningRate = 0.3;

for (let i = 0; i < epochs; i++) {
    // Training code...
}

// Make predictions
const prediction = network.forward([1, 0, 0]); // Should be close to 1
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

When running the 3-input NAND gate example, you should see output similar to:

```
Training the network...
Epoch 0, Error: 1.572464, Learning rate: 0.300000
Epoch 1000, Error: 0.009319, Learning rate: 0.280000
Epoch 2000, Error: 0.003812, Learning rate: 0.260000
Epoch 3000, Error: 0.002380, Learning rate: 0.240000
Epoch 4000, Error: 0.001746, Learning rate: 0.220000
Epoch 5000, Error: 0.001395, Learning rate: 0.200000
Epoch 6000, Error: 0.001177, Learning rate: 0.180000
Epoch 7000, Error: 0.001030, Learning rate: 0.160000
Epoch 8000, Error: 0.000926, Learning rate: 0.140000
Epoch 9000, Error: 0.000851, Learning rate: 0.120000
Epoch 10000, Error: 0.000797, Learning rate: 0.100000
Epoch 11000, Error: 0.000756, Learning rate: 0.080000
Epoch 12000, Error: 0.000728, Learning rate: 0.060000
Epoch 13000, Error: 0.000708, Learning rate: 0.040000
Epoch 14000, Error: 0.000697, Learning rate: 0.020000

Testing the network:
Input: [0,0,0], Expected: 1, Predicted: 1.000
Input: [0,0,1], Expected: 1, Predicted: 0.990
Input: [0,1,0], Expected: 1, Predicted: 0.990
Input: [0,1,1], Expected: 0, Predicted: 0.013
Input: [1,0,0], Expected: 1, Predicted: 1.000
Input: [1,0,1], Expected: 1, Predicted: 0.992
Input: [1,1,0], Expected: 1, Predicted: 0.993
Input: [1,1,1], Expected: 0, Predicted: 0.014

Final predictions:
Prediction for [0, 0,0]: 1
Prediction for [0, 1,0]: 1
Prediction for [1, 0,0]: 1
Prediction for [1, 1,1]: 0
Prediction for [0, 0,1]: 1
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

