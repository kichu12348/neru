class TextProcessor {
    constructor() {
        this.vocabulary = new Map();
        this.nextIndex = 0;
        this.nGramSize = 1; // Default is 1 (single words)
        this.contextMemory = new Map(); // Store context frequencies
        this.categories = new Map(); // Word categories for semantics
        this.wordBlacklist = new Set(['', ' ', 'the', 'a', 'an', 'is', 'are', 'for']);
        this.problemWords = new Map(); // Track words that are causing learning issues
    }
    
    // Set n-gram size
    setNGramSize(n) {
        this.nGramSize = Math.max(1, n);
    }
    
    // Clean and normalize text with better tokenization
    cleanText(text) {
        if (!text || text.trim() === '') return [];
        
        // Better tokenization that preserves word boundaries
        const cleanedText = text.toLowerCase()
            .replace(/\[|\]/g, ' ') // Remove brackets
            .replace(/[^\w\s'-]/g, ' ') // Replace non-word chars with spaces
            .replace(/\s+/g, ' '); // Normalize whitespace
        
        // Split into words and filter empties
        const words = cleanedText.split(' ')
            .filter(w => w && w.trim() !== '')
            .map(w => w.trim());
        
        return words;
    }
    
    // Add new words to vocabulary
    updateVocabulary(text) {
        const words = this.cleanText(text);
        
        words.forEach(word => {
            if (!this.vocabulary.has(word)) {
                this.vocabulary.set(word, this.nextIndex++);
            }
        });
        
        // Track word context for smarter responses
        if (words.length > 1) {
            // Track pairs and triplets
            for (let i = 0; i < words.length - 1; i++) {
                const currentWord = words[i];
                const nextWord = words[i + 1];
                
                // Store word pairs
                if (!this.contextMemory.has(currentWord)) {
                    this.contextMemory.set(currentWord, new Map());
                }
                
                const wordContexts = this.contextMemory.get(currentWord);
                wordContexts.set(nextWord, (wordContexts.get(nextWord) || 0) + 1);
                
                // Store extended context (up to 3 words)
                if (i < words.length - 2) {
                    const thirdWord = words[i + 2];
                    const bigram = `${currentWord} ${nextWord}`;
                    
                    if (!this.contextMemory.has(bigram)) {
                        this.contextMemory.set(bigram, new Map());
                    }
                    
                    const bigramContexts = this.contextMemory.get(bigram);
                    bigramContexts.set(thirdWord, (bigramContexts.get(thirdWord) || 0) + 1);
                }
            }
        }
        
        return words;
    }
    
    // Create an input vector for a word with optional context
    createInputVector(word, contextWord = null) {
        const vocabSize = this.vocabulary.size;
        const vector = Array(vocabSize).fill(0);
        
        if (this.vocabulary.has(word)) {
            vector[this.vocabulary.get(word)] = 1;
            
            // Add context if provided
            if (contextWord && this.vocabulary.has(contextWord)) {
                vector[this.vocabulary.get(contextWord)] = 0.5;
            }
        }
        
        return vector;
    }
    
    // Get probable next words based on context memory
    getContextWords(word, count = 3) {
        if (!this.contextMemory.has(word)) {
            return [];
        }
        
        const contextMap = this.contextMemory.get(word);
        const wordPairs = Array.from(contextMap.entries());
        
        // Sort by frequency, highest first
        wordPairs.sort((a, b) => b[1] - a[1]);
        
        // Return top N words
        return wordPairs.slice(0, count).map(pair => pair[0]);
    }
    
    // Parse explicit training pairs format with improved handling
    parseExplicitPairs(text) {
        try {
            // Check if the text follows the [input, expectedOutput] format
            if (text.trim().startsWith('[') && text.trim().includes(',')) {
                // Extract the input and output parts
                const cleanText = text.replace(/\[|\]/g, '').trim();
                const parts = cleanText.split(',');
                
                if (parts.length >= 2) {
                    const inputWord = parts[0].trim().toLowerCase();
                    
                    // Handle either a single output or an array of outputs
                    let outputWords = [];
                    
                    // Check if second part is an array
                    const outputPart = parts.slice(1).join(',').trim();
                    if (outputPart.startsWith('[') && outputPart.endsWith(']')) {
                        // Parse the array of outputs
                        const outputArray = outputPart.substring(1, outputPart.length - 1).split(',');
                        outputWords = outputArray.map(word => word.trim().toLowerCase()).filter(w => w);
                    } else {
                        // Single output
                        outputWords = [parts[1].trim().toLowerCase()];
                    }
                    
                    // Skip if input or outputs are empty
                    if (!inputWord || outputWords.length === 0) return null;
                    
                    // Create examples for each output
                    const examples = [];
                    
                    // Add both words to vocabulary
                    let allWords = inputWord;
                    outputWords.forEach(word => {
                        allWords += ' ' + word;
                    });
                    this.updateVocabulary(allWords);
                    
                    // Choose a random output for training
                    const randomIndex = Math.floor(Math.random() * outputWords.length);
                    const chosenOutput = outputWords[randomIndex];
                    
                    const vocabSize = this.vocabulary.size;
                    const inputVector = Array(vocabSize).fill(0);
                    const expectedOutputVector = Array(vocabSize).fill(0);
                    
                    if (this.vocabulary.has(inputWord) && this.vocabulary.has(chosenOutput)) {
                        inputVector[this.vocabulary.get(inputWord)] = 1;
                        expectedOutputVector[this.vocabulary.get(chosenOutput)] = 1;
                        
                        examples.push({
                            input: inputVector,
                            expectedOutput: expectedOutputVector,
                            inputWord,
                            outputWord: chosenOutput,
                            allPossibleOutputs: outputWords // Store all possible outputs
                        });
                    }
                    
                    return examples.length > 0 ? examples : null;
                }
            }
        } catch (error) {
            console.error("Error parsing explicit pairs:", error);
        }
        
        return null;
    }
    
    // Convert text to one-hot encoded vectors with n-gram support
    textToInputs(text) {
        // First try to parse as explicit training pair
        const explicitPairs = this.parseExplicitPairs(text);
        if (explicitPairs) {
            return explicitPairs;
        }
        
        // If not an explicit pair, process as normal text
        const words = this.cleanText(text);
        const inputs = [];
        
        for (let i = 0; i < words.length - 1; i++) {
            if (this.vocabulary.has(words[i])) {
                const input = Array(this.vocabulary.size).fill(0);
                input[this.vocabulary.get(words[i])] = 1;
                
                // Add additional context if using n-grams > 1
                if (this.nGramSize > 1 && i > 0) {
                    // For each additional context word, add it to the input with decreasing weight
                    for (let j = 1; j < this.nGramSize && i-j >= 0; j++) {
                        const contextWord = words[i-j];
                        if (this.vocabulary.has(contextWord)) {
                            // Add with lower weight (0.5, 0.25, etc.)
                            input[this.vocabulary.get(contextWord)] = 1 / (2 ** j);
                        }
                    }
                }
                
                // Add next word as expected output
                const expectedOutput = this.getExpectedOutput(words[i+1]);
                
                inputs.push({
                    input,
                    expectedOutput,
                    inputWord: words[i],
                    outputWord: words[i+1]
                });
                
                // Create additional training examples for n-grams (context pairs)
                if (this.nGramSize >= 3 && i >= 1 && i < words.length - 2) {
                    // Create an input that combines current word and previous word
                    // to predict the word after next
                    const bigramInput = Array(this.vocabulary.size).fill(0);
                    bigramInput[this.vocabulary.get(words[i])] = 1;  // Current word
                    bigramInput[this.vocabulary.get(words[i-1])] = 0.5;  // Previous word
                    
                    inputs.push({
                        input: bigramInput,
                        expectedOutput: this.getExpectedOutput(words[i+2]),
                        inputWord: `${words[i-1]} ${words[i]}`,
                        outputWord: words[i+2]
                    });
                }
            }
        }
        
        return inputs;
    }
    
    getExpectedOutput(word) {
        const output = Array(this.vocabulary.size).fill(0);
        if (this.vocabulary.has(word)) {
            output[this.vocabulary.get(word)] = 1;
        }
        return output;
    }
    
    // Convert output vector back to text
    outputToText(output) {
        let maxIndex = 0;
        let maxValue = output[0];
        
        for (let i = 1; i < output.length; i++) {
            if (output[i] > maxValue) {
                maxValue = output[i];
                maxIndex = i;
            }
        }
        
        for (const [word, index] of this.vocabulary.entries()) {
            if (index === maxIndex) return word;
        }
        
        return "(unknown)";
    }
    
    // Get the top N words from an output vector with better scoring
    getTopNWords(output, n = 1) {
        // Create array of [index, value, word] triplets
        const indexedValues = [];
        
        for (let i = 0; i < output.length; i++) {
            for (const [word, index] of this.vocabulary.entries()) {
                if (index === i) {
                    indexedValues.push([i, output[i], word]);
                    break;
                }
            }
        }
        
        // Sort by value in descending order
        indexedValues.sort((a, b) => b[1] - a[1]);
        
        // Filter out blacklisted words if we have enough alternatives
        let filtered = indexedValues.filter(item => !this.wordBlacklist.has(item[2]));
        
        // Fall back to unfiltered if we filtered too much
        if (filtered.length < Math.min(n, indexedValues.length / 2)) {
            filtered = indexedValues;
        }
        
        // Return the top N words
        return filtered.slice(0, n).map(item => item[2]);
    }
    
    // Get vocabulary size
    getVocabularySize() {
        return this.vocabulary.size;
    }

    // Create an example directly from word pair or array
    createExampleFromPair(inputWord, outputWord) {
        // Handle case where outputWord is an array of options
        let outputWords = Array.isArray(outputWord) ? outputWord : [outputWord];
        
        // Make sure words are in vocabulary
        let allWords = inputWord;
        outputWords.forEach(word => {
            allWords += ' ' + word;
        });
        this.updateVocabulary(allWords);
        
        // Choose random output for training
        const randomIndex = Math.floor(Math.random() * outputWords.length);
        const chosenOutput = outputWords[randomIndex];
        
        if (!this.vocabulary.has(inputWord) || !this.vocabulary.has(chosenOutput)) {
            return null;
        }
        
        const vocabSize = this.vocabulary.size;
        const inputVector = Array(vocabSize).fill(0);
        const outputVector = Array(vocabSize).fill(0);
        
        // Set the one-hot encodings
        inputVector[this.vocabulary.get(inputWord)] = 1;
        outputVector[this.vocabulary.get(chosenOutput)] = 1;
        
        return {
            input: inputVector,
            expectedOutput: outputVector,
            inputWord,
            outputWord: chosenOutput,
            allPossibleOutputs: outputWords // Store all possible outputs
        };
    }
    
    // Track problem words that are hard to learn
    trackProblemWord(word, isInput = true) {
        const key = isInput ? `in:${word}` : `out:${word}`;
        const count = this.problemWords.get(key) || 0;
        this.problemWords.set(key, count + 1);
    }
    
    // Get most problematic words
    getProblemWords(limit = 10) {
        return Array.from(this.problemWords.entries())
            .sort((a, b) => b[1] - a[1])
            .slice(0, limit);
    }
    
    // Get input vector for a word with stronger encoding
    getStrongInputVector(word) {
        const vocabSize = this.vocabulary.size;
        const vector = Array(vocabSize).fill(0);
        
        if (this.vocabulary.has(word)) {
            // Strong one-hot encoding (exactly 1.0)
            vector[this.vocabulary.get(word)] = 1.0;
        }
        
        return vector;
    }
}

module.exports = TextProcessor;
