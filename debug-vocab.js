const fs = require('fs');
const path = require('path');
const natural = require('natural');

const MEMORY_PATH = path.join(__dirname, 'memory.json');
const VOCAB_PATH = path.join(__dirname, 'vocab.json');

const tokenizer = new natural.WordTokenizer();
const stemmer = natural.PorterStemmer;

// Check if vocabulary exists
if (fs.existsSync(VOCAB_PATH)) {
  console.log('Checking existing vocabulary...');
  const vocabData = JSON.parse(fs.readFileSync(VOCAB_PATH, 'utf-8'));
  console.log(`Loaded vocabulary with ${Object.keys(vocabData.wordIndex).length} words`);
  
  // Show some sample entries
  const entries = Object.entries(vocabData.wordIndex).slice(0, 10);
  console.log('Sample words:', entries);
} else {
  console.log('No vocabulary file found.');
}

// Check memory
if (fs.existsSync(MEMORY_PATH)) {
  const memory = JSON.parse(fs.readFileSync(MEMORY_PATH, 'utf-8'));
  console.log(`Memory contains ${memory.length} entries`);
  
  // Build a fresh vocabulary from memory
  const wordIndex = {};
  const indexWord = {};
  
  memory.forEach(item => {
    const text = item.input + ' ' + item.output;
    const tokens = tokenizer.tokenize(text.toLowerCase());
    const stems = tokens.map(token => stemmer.stem(token));
    
    stems.forEach(word => {
      if (!wordIndex[word]) {
        const index = Object.keys(wordIndex).length + 1;
        wordIndex[word] = index;
        indexWord[index] = word;
      }
    });
  });
  
  console.log(`Fresh vocabulary would have ${Object.keys(wordIndex).length} words`);
  
  // Save this new vocabulary for comparison
  fs.writeFileSync(
    path.join(__dirname, 'debug_vocab.json'),
    JSON.stringify({ wordIndex, indexWord }, null, 2)
  );
  console.log('Debug vocabulary saved to debug_vocab.json');
} else {
  console.log('No memory file found.');
}

console.log('\nTry running this script after fixing the vocabulary issue to verify the changes.');
