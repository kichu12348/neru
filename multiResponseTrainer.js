/**
 * Utility class for training networks with multiple response options
 */
class MultiResponseTrainer {
    constructor(network, textProcessor) {
        this.network = network;
        this.textProcessor = textProcessor;
        this.trainingStats = {
            total: 0,
            success: 0,
            trainedAssociations: new Map()
        };
    }
    
    /**
     * Train the network on an input with multiple possible outputs
     * @param {string} input - Input word or phrase
     * @param {Array<string>|string} outputs - Possible outputs
     * @param {Object} options - Training options
     * @returns {Object} - Training results
     */
    async trainMultiResponse(input, outputs, options = {}) {
        const outputArray = Array.isArray(outputs) ? outputs : [outputs];
        const maxAttempts = options.maxAttempts || 50;
        const rotateResponses = options.rotateResponses !== false;
        const results = [];
        
        console.log(`Training "${input}" with ${outputArray.length} possible responses`);
        
        // First try with randomly selected output
        const example = this.textProcessor.createExampleFromPair(input, outputArray);
        if (!example) {
            console.log("Could not create training example.");
            return { success: false, results };
        }
        
        // Initial training on randomly selected output
        const initialResult = this.network.trainUntilLearned(
            example.input,
            example.expectedOutput,
            input,
            example.outputWord,
            Math.floor(maxAttempts * 0.5) // Use half attempts for initial training
        );
        
        results.push({
            output: example.outputWord,
            success: initialResult.success,
            attempts: initialResult.attempts,
            error: initialResult.error
        });
        
        // If we want to rotate through all responses, train on each
        if (rotateResponses && outputArray.length > 1) {
            // For each additional output, do some light training
            for (let i = 0; i < outputArray.length; i++) {
                // Skip the already trained output
                if (outputArray[i] === example.outputWord) continue;
                
                // Create example for this specific output
                const rotatedExample = this.textProcessor.createExampleFromPair(input, outputArray[i]);
                if (!rotatedExample) continue;
                
                console.log(`  Training rotation ${i+1}/${outputArray.length}: "${input}" → "${outputArray[i]}"`);
                
                // Train with fewer attempts
                const rotationResult = this.network.trainUntilLearned(
                    rotatedExample.input,
                    rotatedExample.expectedOutput,
                    input,
                    rotatedExample.outputWord,
                    Math.floor(maxAttempts * 0.3) // Use fewer attempts for rotations
                );
                
                results.push({
                    output: rotatedExample.outputWord,
                    success: rotationResult.success,
                    attempts: rotationResult.attempts,
                    error: rotationResult.error
                });
            }
        }
        
        // Verify training by testing against all possible outputs
        const testResult = this.testMultiResponse(input, outputArray);
        
        // Track training statistics
        this.trainingStats.total++;
        if (testResult.success) {
            this.trainingStats.success++;
        }
        this.trainingStats.trainedAssociations.set(input, {
            outputs: outputArray,
            success: testResult.success
        });
        
        return {
            success: testResult.success,
            results,
            testResult
        };
    }
    
    /**
     * Test if any of the possible outputs is produced
     * @param {string} input - Input word or phrase
     * @param {Array<string>|string} expectedOutputs - Possible outputs
     * @returns {Object} - Test results
     */
    testMultiResponse(input, expectedOutputs) {
        const outputArray = Array.isArray(expectedOutputs) ? expectedOutputs : [expectedOutputs];
        
        // Create input vector
        const inputVector = this.textProcessor.getStrongInputVector(input);
        
        // Run inference
        const output = this.network.feedForward(inputVector);
        const predictedWord = this.textProcessor.outputToText(output);
        
        // Check if prediction is among expected outputs
        const success = outputArray.includes(predictedWord);
        
        // Get top 3 predictions for diagnostics
        const topN = 3;
        const topPredictions = this.textProcessor.getTopNWords(output, topN);
        
        // Check if any of the expected outputs are in top predictions
        const topNContainsExpected = topPredictions.some(word => outputArray.includes(word));
        
        return {
            input,
            expectedOutputs: outputArray,
            predictedWord,
            success,
            topNContainsExpected,
            topPredictions
        };
    }
    
    /**
     * Train on all entries in training data
     * @param {Array} trainingData - Training data in [input, output(s)] format
     * @param {Object} options - Training options
     * @returns {Object} - Training results
     */
    async trainAll(trainingData, options = {}) {
        const results = [];
        let successful = 0;
        
        for (let i = 0; i < trainingData.length; i++) {
            const [input, outputs] = trainingData[i];
            
            if (options.verbose) {
                console.log(`Training item ${i+1}/${trainingData.length}: "${input}"`);
            }
            
            const result = await this.trainMultiResponse(input, outputs, options);
            results.push(result);
            
            if (result.success) successful++;
            
            // Show progress periodically
            if (options.verbose && (i+1) % 10 === 0) {
                console.log(`Progress: ${i+1}/${trainingData.length}, Success rate: ${(successful/(i+1)*100).toFixed(1)}%`);
            }
        }
        
        const successRate = successful / trainingData.length * 100;
        console.log(`Training complete. Overall success rate: ${successRate.toFixed(1)}%`);
        
        return {
            results,
            successful,
            total: trainingData.length,
            successRate
        };
    }
}

module.exports = MultiResponseTrainer;
