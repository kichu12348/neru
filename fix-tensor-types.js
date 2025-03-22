const tf = require('@tensorflow/tfjs');

console.log('TensorFlow.js Tensor Type Verification');
console.log('=====================================');

// Check TensorFlow.js version
console.log(`Using TensorFlow.js version: ${tf.version.tfjs}`);
console.log('Verifying tensor types and loss function compatibility...\n');

// Create test tensors
const labels = [0, 1, 2, 3, 4]; // Sparse labels
const predictions = [
  [0.1, 0.8, 0.05, 0.05, 0],
  [0.1, 0.2, 0.6, 0.05, 0.05],
  [0.3, 0.3, 0.05, 0.3, 0.05],
  [0.05, 0.05, 0.05, 0.8, 0.05],
  [0.05, 0.05, 0.05, 0.05, 0.8]
];

// Try different tensor types
console.log('Testing sparse categorical cross-entropy with different tensor types:');

try {
  // Test with float32 labels
  console.log('\nTest 1: float32 labels, float32 predictions');
  const floatLabels = tf.tensor1d(labels, 'float32');
  const floatPreds = tf.tensor2d(predictions, [5, 5], 'float32');
  const loss1 = tf.metrics.sparseCategoricalAccuracy(floatLabels, floatPreds);
  console.log('Result:', loss1.dataSync());
  console.log('✓ Success with float32 labels, float32 predictions');
} catch (error) {
  console.log('✗ Error with float32 labels, float32 predictions:', error.message);
}

try {
  // Test with int32 labels
  console.log('\nTest 2: int32 labels, float32 predictions');
  const intLabels = tf.tensor1d(labels, 'int32');
  const floatPreds = tf.tensor2d(predictions, [5, 5], 'float32');
  const loss2 = tf.metrics.sparseCategoricalAccuracy(intLabels, floatPreds);
  console.log('Result:', loss2.dataSync());
  console.log('✓ Success with int32 labels, float32 predictions');
} catch (error) {
  console.log('✗ Error with int32 labels, float32 predictions:', error.message);
}

// Try simple model with sparse categorical cross-entropy
console.log('\nTesting simple model with sparse categorical cross-entropy:');

try {
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 5, inputShape: [3], activation: 'softmax'}));
  
  model.compile({
    optimizer: 'adam',
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy']
  });
  
  const inputs = tf.tensor2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [3, 3], 'float32');
  // Try with float32 labels
  const outputs = tf.tensor1d([0, 1, 2], 'float32');
  
  console.log('Model summary:');
  model.summary();
  
  console.log('\nFitting model with float32 labels...');
  model.fit(inputs, outputs, {epochs: 1}).then(() => {
    console.log('✓ Success: Model trained with float32 labels');
    console.log('\nRecommended fix for Neru: Use float32 for nextWords tensor');
  }).catch(error => {
    console.log('✗ Error:', error.message);
  });
} catch (error) {
  console.log('✗ Error setting up model:', error.message);
}

console.log('\nThis utility script helps diagnose tensor type issues in the Neru model.');
