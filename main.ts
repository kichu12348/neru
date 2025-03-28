import Network from './network.ts';

// Example: Let's solve the XOR problem
// XOR truth table:
// 0,0 -> 0
// 0,1 -> 1
// 1,0 -> 1
// 1,1 -> 0

//NAND truth table
// 0,0 -> 1
// 0,1 -> 1
// 1,0 -> 1
// 1,1 -> 0

// Create a network with:
// - 2 inputs (two binary values)
// - 4 hidden neurons (we need at least 2 for XOR, using more for better results)
// - 1 output (binary result)
const network = new Network(2, 8, 1);

// Training data for NAND gate
const trainingData = [
    { inputs: [0, 0], targets: [0] },
    { inputs: [0, 1], targets: [0] },
    { inputs: [1, 0], targets: [0] },
    { inputs: [1, 1], targets: [1] }
];

// Training parameters
let learningRate = 0.3;
const epochs = 15000;
const initialLearningRate = 0.3;

// Training the network
console.log("Training the network...");
for (let i = 0; i < epochs; i++) {
    const shuffled = [...trainingData].sort(() => Math.random() - 0.5);
    
    let totalError = 0;
    for (const example of shuffled) {
        totalError += network.train(example.inputs, example.targets, learningRate);
    }
    
    // Gradually decrease learning rate over time
    learningRate = initialLearningRate * (1 - i / epochs);
    if (learningRate < 0.01) learningRate = 0.01; // Minimum learning rate
    
    // Log progress every 1000 epochs
    if (i % 1000 === 0) {
        console.log(`Epoch ${i}, Error: ${totalError.toFixed(6)}, Learning rate: ${learningRate.toFixed(6)}`);
    }
}

// Testing the network
console.log("\nTesting the network:");
for (const example of trainingData) {
    const prediction = network.forward(example.inputs);
    console.log(`Input: [${example.inputs}], Expected: ${example.targets}, Predicted: ${prediction.map(p => p.toFixed(3))}`);
}

// Function to make a prediction
function predict(input1: number, input2: number): void {
    const output = network.forward([input1, input2])[0];
    console.log(`${input1} NAND ${input2} = ${output.toFixed(3)} (Rounded: ${Math.round(output)})`);
}

// Test all possible inputs
console.log("\nFinal predictions:");
predict(0, 0);
predict(0, 1);
predict(1, 0);
predict(1, 1);
