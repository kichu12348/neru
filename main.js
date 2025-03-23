class Neuron {
  constructor(inputSize) {
    this.weights = Array(inputSize)
      .fill()
      .map(() => Math.random() * 0.2 - 0.1);
    this.bias = Math.random() * 0.2 - 0.1;
  }

  activation(sum) {
    return 1 / (1 + Math.exp(-sum));
  }

  feedForward(inputs) {
    let sum = this.bias;
    for (let i = 0; i < inputs.length; i++) {
      sum += inputs[i] * this.weights[i];
    }
    return this.activation(sum);
  }
}

class Layer {
  constructor(neuronCount, inputSize) {
    this.neurons = Array(neuronCount)
      .fill()
      .map(() => new Neuron(inputSize));
  }

  feedForward(inputs) {
    return this.neurons.map((neuron) => neuron.feedForward(inputs));
  }
}

class NeuralNetwork {
  constructor(inputSize, hiddenSize, outputSize) {
    this.hiddenLayer = new Layer(hiddenSize, inputSize);
    this.outputLayer = new Layer(outputSize, hiddenSize);
    this.learningRate = 0.3;
  }

  feedForward(inputs) {
    const hiddenOutputs = this.hiddenLayer.feedForward(inputs);
    return this.outputLayer.feedForward(hiddenOutputs);
  }

  train(trainingData, epochs) {
    for (let epoch = 0; epoch < epochs; epoch++) {
      for (const data of trainingData) {
        const hiddenOutputs = this.hiddenLayer.feedForward(data.inputs);
        const finalOutputs = this.outputLayer.feedForward(hiddenOutputs);

        const outputErrors = [];
        for (let i = 0; i < finalOutputs.length; i++) {
          const error = data.outputs[i] - finalOutputs[i];
          outputErrors.push(error);
        }

        for (let i = 0; i < this.outputLayer.neurons.length; i++) {
          const neuron = this.outputLayer.neurons[i];
          for (let j = 0; j < hiddenOutputs.length; j++) {
            neuron.weights[j] +=
              this.learningRate *
              outputErrors[i] *
              finalOutputs[i] *
              (1 - finalOutputs[i]) *
              hiddenOutputs[j];
          }
          neuron.bias +=
            this.learningRate *
            outputErrors[i] *
            finalOutputs[i] *
            (1 - finalOutputs[i]);
        }

        const hiddenErrors = [];
        for (let i = 0; i < hiddenOutputs.length; i++) {
          let error = 0;
          for (let j = 0; j < this.outputLayer.neurons.length; j++) {
            error += outputErrors[j] * this.outputLayer.neurons[j].weights[i];
          }
          hiddenErrors.push(error);
        }

        for (let i = 0; i < this.hiddenLayer.neurons.length; i++) {
          const neuron = this.hiddenLayer.neurons[i];
          for (let j = 0; j < data.inputs.length; j++) {
            neuron.weights[j] +=
              this.learningRate *
              hiddenErrors[i] *
              hiddenOutputs[i] *
              (1 - hiddenOutputs[i]) *
              data.inputs[j];
          }
          neuron.bias +=
            this.learningRate *
            hiddenErrors[i] *
            hiddenOutputs[i] *
            (1 - hiddenOutputs[i]);
        }
      }
    }
  }

  static createNetwork(trainingData, epochs = 10000) {
    const inputSize = trainingData[0].inputs.length;
    const outputSize = trainingData[0].outputs.length;
    const hiddenSize = 4;

    const network = new NeuralNetwork(inputSize, hiddenSize, outputSize);
    network.train(trainingData, epochs);
    return network;
  }
}

const trainingData = [
  {
    inputs: [0, 0],
    outputs: [0],
  },
  {
    inputs: [0, 1],
    outputs: [1],
  },
  {
    inputs: [1, 0],
    outputs: [1],
  },
  {
    inputs: [1, 1],
    outputs: [0],
  },
];

function main() {
  const network = NeuralNetwork.createNetwork(trainingData);

  trainingData.forEach((data) => {
    const result = network.feedForward(data.inputs);
    console.log(
      `Input: [${data.inputs}] → Output: ${result} (Expected: ${data.outputs})`
    );
  });
}

main();
