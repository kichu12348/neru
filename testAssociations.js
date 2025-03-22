// Utility to test if the network has properly learned associations

class AssociationTester {
    constructor(network, textProcessor) {
        this.network = network;
        this.textProcessor = textProcessor;
        this.testResults = [];
    }
    
    // Test a single association
    testAssociation(inputWord, expectedOutputWord) {
        // Handle multiple possible outputs
        const expectedOutputWords = Array.isArray(expectedOutputWord) ? expectedOutputWord : [expectedOutputWord];
        
        // Create input vector
        const inputVector = this.textProcessor.getStrongInputVector(inputWord);
        
        // Get network output
        const output = this.network.feedForward(inputVector);
        const predictedWord = this.textProcessor.outputToText(output);
        
        // Compare with any of the expected outputs
        const success = expectedOutputWords.includes(predictedWord);
        
        // Save result
        this.testResults.push({
            inputWord,
            expectedOutputWords,
            predictedWord,
            success
        });
        
        return { predictedWord, success };
    }
    
    // Test a list of associations
    testMultiple(associationList) {
        const results = [];
        let successCount = 0;
        
        associationList.forEach(([input, expected]) => {
            const result = this.testAssociation(input, expected);
            results.push({
                input,
                expected,
                predicted: result.predictedWord,
                success: result.success
            });
            
            if (result.success) successCount++;
        });
        
        const successRate = associationList.length > 0 ? 
            (successCount / associationList.length) * 100 : 0;
            
        return { results, successCount, total: associationList.length, successRate };
    }
    
    // Fix failed associations through intensive training
    fixFailedAssociations(maxAttempts = 50) {
        const failed = this.testResults.filter(r => !r.success);
        console.log(`Fixing ${failed.length} failed associations...`);
        
        const fixResults = [];
        
        for (const failedTest of failed) {
            // Choose one expected output for training
            const chosenOutput = failedTest.expectedOutputWords[0];
            console.log(`Training: "${failedTest.inputWord}" → "${chosenOutput}"`);
            
            // Create example
            const example = this.textProcessor.createExampleFromPair(
                failedTest.inputWord, 
                failedTest.expectedOutputWords
            );
            
            if (!example) {
                console.log("  Could not create training example.");
                continue;
            }
            
            // Train until learned
            const result = this.network.trainUntilLearned(
                example.input,
                example.expectedOutput,
                failedTest.inputWord,
                example.outputWord,
                maxAttempts
            );
            
            // Test again
            const testResult = this.testAssociation(
                failedTest.inputWord, 
                failedTest.expectedOutputWords
            );
            
            fixResults.push({
                input: failedTest.inputWord,
                expected: failedTest.expectedOutputWords,
                predicted: testResult.predictedWord,
                success: testResult.success,
                attempts: result.attempts
            });
            
            console.log(`  Result: ${testResult.success ? "Fixed!" : "Still failing"} (${result.attempts} attempts)`);
        }
        
        const fixedCount = fixResults.filter(r => r.success).length;
        console.log(`Fixed ${fixedCount} out of ${fixResults.length} associations.`);
        
        return fixResults;
    }
}

module.exports = AssociationTester;
