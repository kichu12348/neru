# Neru - A Simple Neural Network Implementation

Neru is a lightweight JavaScript implementation of a feedforward neural network. This project provides a simple and educational neural network that can be trained to solve basic problems like logical operations.

## Features

- Simple feedforward neural network with configurable layers
- Backpropagation algorithm for training
- Sigmoid activation function
- Customizable learning rate
- Utility method for quick network creation and training

## Getting Started

### Prerequisites

- Node.js installed on your system

### Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/yourusername/neru.git
cd neru
```

## Usage

```javascript
// Import the NeuralNetwork class
const { NeuralNetwork } = require('./main.js');

// Define your training data
const trainingData = [
  { inputs: [0, 0], outputs: [0] }, // Example: XOR operation
  { inputs: [0, 1], outputs: [1] },
  { inputs: [1, 0], outputs: [1] },
  { inputs: [1, 1], outputs: [0] }
];

// Create and train a network
const network = NeuralNetwork.createNetwork(trainingData, 10000);

// Use the trained network
const result = network.feedForward([1, 0]);
console.log(`Prediction: ${result}`);
```

## How It Works

The neural network consists of:
1. An input layer (size determined by your input data)
2. A hidden layer (configurable size, default is 4 neurons)
3. An output layer (size determined by your output data)

The network uses:
- Sigmoid activation function: `1 / (1 + Math.exp(-x))`
- Backpropagation for training
- A default learning rate of 0.3

## Example

The included example trains the network to understand the XOR logical operation:

| Input 1 | Input 2 | Output |
|---------|---------|--------|
| 0       | 0       | 0      |
| 0       | 1       | 1      |
| 1       | 0       | 1      |
| 1       | 1       | 0      |

Run the example:

```bash
node main.js
```

Expected output:

