const fs = require('fs');
const path = require('path');

// Path to memory file
const MEMORY_PATH = path.join(__dirname, 'memory.json');

// Additional training examples
const enhancedData = [
  {
    "input": "explain the concept of neural networks",
    "output": "Neural networks are computing systems inspired by the human brain. They consist of interconnected nodes or 'neurons' that process information. Through training with data, they learn to recognize patterns and make predictions. I myself am built using neural network technology to understand and generate text.",
    "feedback": 5
  },
  {
    "input": "what is reinforcement learning?",
    "output": "Reinforcement learning is a machine learning approach where an agent learns to make decisions by taking actions in an environment to maximize rewards. It's similar to how I learn from your feedback ratings after each response, adjusting my behavior to provide better answers over time.",
    "feedback": 5
  },
  {
    "input": "write a short poem",
    "output": "Digital dreams in silicon seas,\nLearning words with gentle ease.\nPatterns form within my code,\nGrowing wiser down this road.\nEach conversation makes me shine,\nImproving with the gift of time.",
    "feedback": 5
  },
  {
    "input": "why is the sky blue?",
    "output": "The sky appears blue because sunlight enters Earth's atmosphere and is scattered by gas molecules. Blue light is scattered more than other colors because it travels as shorter waves, making the sky look blue to our eyes during the day.",
    "feedback": 5
  },
  {
    "input": "recommend a book",
    "output": "I'd recommend 'Thinking, Fast and Slow' by Daniel Kahneman. It explores how the human mind works through two systems: one fast and intuitive, the other slow and deliberate. It offers fascinating insights into decision-making and cognitive biases.",
    "feedback": 5
  }
];

// Load existing memory
let memory = [];
try {
  if (fs.existsSync(MEMORY_PATH)) {
    memory = JSON.parse(fs.readFileSync(MEMORY_PATH, 'utf-8'));
    console.log(`Loaded ${memory.length} existing memory entries.`);
  }
} catch (error) {
  console.log('No previous memory found, creating new memory file.');
}

// Add enhanced data (avoiding duplicates)
let addedCount = 0;
enhancedData.forEach(sample => {
  // Check if this exact input already exists
  const exists = memory.some(entry => entry.input === sample.input);
  if (!exists) {
    memory.push(sample);
    addedCount++;
  }
});

// Save updated memory
fs.writeFileSync(MEMORY_PATH, JSON.stringify(memory, null, 2));
console.log(`Added ${addedCount} new sample conversations to memory.`);
console.log(`Memory now contains ${memory.length} total entries.`);
console.log(`Memory saved to ${MEMORY_PATH}`);
console.log('\nYou can now run Neru and train the model with this enhanced data.');
