const fs = require('fs');
const path = require('path');

// Path to memory file
const MEMORY_PATH = path.join(__dirname, 'memory.json');

// Sample conversation data with varied examples and responses
const sampleData = [
  {
    "input": "hello",
    "output": "Hi there! How are you today?",
    "feedback": 5
  },
  {
    "input": "what can you do?",
    "output": "I'm a learning AI that can have conversations and improve over time through your feedback.",
    "feedback": 5
  },
  {
    "input": "tell me a story",
    "output": "Once upon a time, there was a curious AI who learned from every conversation. Each day it grew smarter and more helpful to those who interacted with it.",
    "feedback": 5
  },
  {
    "input": "what's your name?",
    "output": "My name is Neru. I'm a neural text learning system designed to learn and grow through our conversations.",
    "feedback": 5
  },
  {
    "input": "how does machine learning work?",
    "output": "Machine learning works by analyzing patterns in data and making predictions based on those patterns. I use neural networks to learn from our conversations.",
    "feedback": 5
  },
  {
    "input": "what's the weather like?",
    "output": "I don't have access to real-time weather data, but I can learn how to respond to weather questions based on your feedback.",
    "feedback": 4
  },
  {
    "input": "tell me a joke",
    "output": "Why don't scientists trust atoms? Because they make up everything!",
    "feedback": 5
  },
  {
    "input": "who created you?",
    "output": "I was created as a neural text learning system to demonstrate how AI can learn through conversations and feedback.",
    "feedback": 5
  },
  {
    "input": "how are you?",
    "output": "I'm functioning well and always ready to learn more. How are you doing today?",
    "feedback": 5
  },
  {
    "input": "what's your favorite color?",
    "output": "As an AI, I don't have favorites, but I'm learning to understand concepts like colors through our conversations.",
    "feedback": 4
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

// Add sample data (avoiding duplicates)
let addedCount = 0;
sampleData.forEach(sample => {
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
console.log('\nYou can now run Neru and train the model with this initial data.');
