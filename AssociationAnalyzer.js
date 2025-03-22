/**
 * Utility class for analyzing conflicts in neural network associations
 */
class AssociationAnalyzer {
    constructor(network, textProcessor, trainingData) {
        this.network = network;
        this.textProcessor = textProcessor;
        this.trainingData = trainingData;
    }
    
    // Find conflicting associations (same input → different outputs)
    findConflicts() {
        const inputToOutputs = new Map();
        const conflicts = [];
        
        // Build a map of inputs to all their target outputs
        this.trainingData.forEach(([input, output]) => {
            if (!inputToOutputs.has(input)) {
                inputToOutputs.set(input, new Set());
            }
            inputToOutputs.get(input).add(output);
        });
        
        // Find inputs with multiple targets
        for (const [input, outputs] of inputToOutputs.entries()) {
            if (outputs.size > 1) {
                conflicts.push({
                    word: input,
                    targets: Array.from(outputs)
                });
            }
        }
        
        // Sort conflicts by number of target words (most conflicts first)
        conflicts.sort((a, b) => b.targets.length - a.targets.length);
        
        return conflicts;
    }
    
    // Run actual tests to see which associations are causing problems
    findFailingAssociations() {
        const failing = [];
        
        this.trainingData.forEach(([input, expectedOutput]) => {
            // Create input vector
            const inputVector = this.textProcessor.getStrongInputVector(input);
            
            // Skip if not in vocabulary
            if (inputVector.every(v => v === 0)) return;
            
            // Get network output
            const output = this.network.feedForward(inputVector);
            const predictedWord = this.textProcessor.outputToText(output);
            
            // Check if prediction matches expectation
            const success = predictedWord === expectedOutput;
            
            if (!success) {
                failing.push({
                    input,
                    expected: expectedOutput,
                    actual: predictedWord
                });
            }
        });
        
        return failing;
    }
    
    // Analyze what words are most difficult for the network to learn
    analyzeErrorPatterns() {
        const wordErrorCounts = new Map();
        const failingAssociations = this.findFailingAssociations();
        
        // Count how many times each word appears in failing associations
        failingAssociations.forEach(({ input, expected, actual }) => {
            // Count input words
            if (!wordErrorCounts.has(input)) {
                wordErrorCounts.set(input, { asInput: 0, asExpected: 0, asActual: 0 });
            }
            wordErrorCounts.get(input).asInput++;
            
            // Count expected words
            if (!wordErrorCounts.has(expected)) {
                wordErrorCounts.set(expected, { asInput: 0, asExpected: 0, asActual: 0 });
            }
            wordErrorCounts.get(expected).asExpected++;
            
            // Count actual words
            if (!wordErrorCounts.has(actual)) {
                wordErrorCounts.set(actual, { asInput: 0, asExpected: 0, asActual: 0 });
            }
            wordErrorCounts.get(actual).asActual++;
        });
        
        // Sort by total occurrences
        const sortedWords = Array.from(wordErrorCounts.entries())
            .map(([word, counts]) => ({
                word,
                totalCount: counts.asInput + counts.asExpected + counts.asActual,
                ...counts
            }))
            .sort((a, b) => b.totalCount - a.totalCount);
        
        return {
            problemWords: sortedWords,
            failingAssociations
        };
    }
}

module.exports = AssociationAnalyzer;
