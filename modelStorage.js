const VectorDb = require('./vectorDb');
const fs = require('fs').promises;
const path = require('path');

/**
 * Neural network model storage system
 */
class ModelStorage {
    constructor(options = {}) {
        this.modelsDir = options.modelsDir || './models';
        this.currentModelName = options.modelName || 'default';
        this.vectorDb = new VectorDb({
            storageDir: path.join(this.modelsDir, 'vectors')
        });
        this.modelRegistry = new Map(); // name -> metadata
    }
    
    /**
     * Initialize the storage system
     */
    async initialize() {
        try {
            // Ensure directories exist
            await fs.mkdir(this.modelsDir, { recursive: true });
            
            // Load model registry if exists
            try {
                const registryPath = path.join(this.modelsDir, 'registry.json');
                const registryData = JSON.parse(await fs.readFile(registryPath, 'utf8'));
                this.modelRegistry = new Map(registryData);
            } catch (e) {
                // Registry doesn't exist yet, will be created later
                console.log("No model registry found, creating a new one.");
            }
            
            // Initialize vector database
            await this.vectorDb.load();
            
            return true;
        } catch (error) {
            console.error("Error initializing model storage:", error);
            return false;
        }
    }
    
    /**
     * Save neural network model
     * @param {NeuralNetwork} network - The neural network to save
     * @param {string} name - Optional model name
     * @param {Object} metadata - Optional metadata
     */
    async saveModel(network, name = this.currentModelName, metadata = {}) {
        // Generate model data
        const modelData = {
            name,
            timestamp: Date.now(),
            metadata: {
                ...metadata,
                inputSize: network.inputSize,
                outputSize: network.outputSize,
                hiddenLayers: network.layers.map(l => l.length)
            },
            layers: network.layers.map(layer => 
                layer.map(neuron => ({
                    weights: neuron.weights,
                    bias: neuron.bias
                }))
            ),
            outputLayer: network.outputLayer.map(neuron => ({
                weights: neuron.weights,
                bias: neuron.bias
            }))
        };
        
        // Save to file
        const modelPath = path.join(this.modelsDir, `${name}.json`);
        await fs.writeFile(modelPath, JSON.stringify(modelData, null, 2));
        
        // Update registry
        this.modelRegistry.set(name, {
            name,
            timestamp: modelData.timestamp,
            path: modelPath,
            ...metadata
        });
        
        // Save registry
        const registryPath = path.join(this.modelsDir, 'registry.json');
        await fs.writeFile(
            registryPath, 
            JSON.stringify(Array.from(this.modelRegistry.entries()), null, 2)
        );
        
        // Save weight vectors for similarity search
        await this.saveWeightVectors(network, name);
        
        this.currentModelName = name;
        return modelPath;
    }
    
    /**
     * Save weight vectors for analysis
     */
    async saveWeightVectors(network, modelName) {
        // Save output layer vectors for similarity analysis
        const vectorBatch = [];
        
        // Process output layer neurons
        for (let i = 0; i < network.outputLayer.length; i++) {
            const neuron = network.outputLayer[i];
            const key = `${modelName}:output:${i}`;
            
            vectorBatch.push({
                key,
                vector: neuron.weights,
                metadata: { 
                    type: 'outputNeuron',
                    neuronIndex: i,
                    layer: 'output',
                    modelName,
                    bias: neuron.bias
                }
            });
        }
        
        await this.vectorDb.addVectors(vectorBatch);
    }
    
    /**
     * Load a neural network model
     * @param {string} name - The model name to load
     * @param {NeuralNetwork} network - The network to load into
     */
    async loadModel(name, network) {
        // Get model path
        const modelPath = path.join(this.modelsDir, `${name}.json`);
        
        // Load model data
        const modelData = JSON.parse(await fs.readFile(modelPath, 'utf8'));
        
        // Verify model structure compatibility
        if (modelData.metadata.inputSize !== network.inputSize || 
            modelData.metadata.outputSize !== network.outputSize) {
            console.log(`Resizing network from ${network.inputSize}x${network.outputSize} to ${modelData.metadata.inputSize}x${modelData.metadata.outputSize}`);
            network.resize(modelData.metadata.inputSize, modelData.metadata.outputSize);
        }
        
        // Load hidden layers
        for (let l = 0; l < Math.min(network.layers.length, modelData.layers.length); l++) {
            const layer = network.layers[l];
            const layerData = modelData.layers[l];
            
            for (let n = 0; n < Math.min(layer.length, layerData.length); n++) {
                layer[n].weights = layerData[n].weights;
                layer[n].bias = layerData[n].bias;
            }
        }
        
        // Load output layer
        for (let i = 0; i < Math.min(network.outputLayer.length, modelData.outputLayer.length); i++) {
            network.outputLayer[i].weights = modelData.outputLayer[i].weights;
            network.outputLayer[i].bias = modelData.outputLayer[i].bias;
        }
        
        this.currentModelName = name;
        return true;
    }
    
    /**
     * List available models
     * @returns {Promise<Array<Object>>} - List of models with metadata
     */
    async listModels() {
        return Array.from(this.modelRegistry.values())
            .sort((a, b) => b.timestamp - a.timestamp); // Sort by timestamp descending
    }
    
    /**
     * Find similar models to a given model using weight vectors
     */
    async findSimilarModels(modelName, limit = 5) {
        // Get a sample of weight vectors from the model
        const sampleVectors = await this.vectorDb.findSimilarByKey(
            `${modelName}:output:0`, 
            { limit: 100 }
        );
        
        // Count occurrences of each model
        const modelScores = new Map();
        
        for (const result of sampleVectors) {
            if (result.metadata && result.metadata.modelName) {
                const resultModel = result.metadata.modelName;
                if (resultModel !== modelName) { // Skip the query model itself
                    const currentScore = modelScores.get(resultModel) || 0;
                    modelScores.set(resultModel, currentScore + result.similarity);
                }
            }
        }
        
        // Sort and return top results
        return Array.from(modelScores.entries())
            .map(([model, score]) => ({ model, score }))
            .sort((a, b) => b.score - a.score)
            .slice(0, limit);
    }
}

module.exports = ModelStorage;
