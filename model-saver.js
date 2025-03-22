const tf = require('@tensorflow/tfjs');  // Use pure JS version
const fs = require('fs');
const path = require('path');

// Configure TensorFlow.js
tf.setBackend('cpu');
tf.enableProdMode();

// Constants
const MODEL_PATH = path.join(__dirname, 'model');

// Make sure directory exists
if (!fs.existsSync(MODEL_PATH)) {
  fs.mkdirSync(MODEL_PATH, { recursive: true });
  console.log(`Created directory: ${MODEL_PATH}`);
}

// Create a simple test model
async function createAndSaveModel() {
  console.log('Creating test model...');
  
  // Create a simple model
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 5, inputShape: [3], activation: 'softmax'}));
  
  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  
  console.log('Model created.');
  console.log('Trying to save model...');
  
  try {
    // Use standard Node.js file system to manually save model files
    const modelJSON = model.toJSON();
    
    // Save the model JSON to a file
    fs.writeFileSync(
      path.join(MODEL_PATH, 'model.json'),
      JSON.stringify(modelJSON)
    );
    
    // Save weights separately
    for (const layer of model.layers) {
      if (layer.getWeights && layer.getWeights().length > 0) {
        const layerWeights = layer.getWeights();
        fs.writeFileSync(
          path.join(MODEL_PATH, `layer_${layer.name}_weights.json`),
          JSON.stringify(layerWeights.map(w => w.arraySync()))
        );
      }
    }
    
    console.log('Model metadata saved successfully!');
    console.log(`Files should be in: ${MODEL_PATH}`);
    
    // List files in the directory
    const files = fs.readdirSync(MODEL_PATH);
    console.log('Files in directory:', files);
  } catch (error) {
    console.error('Failed to save model:', error);
  }
}

// Run the test
createAndSaveModel();
