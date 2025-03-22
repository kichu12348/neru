/**
 * Embedding layer for converting word indices to dense vectors
 */
class EmbeddingLayer {
    /**
     * Create an embedding layer
     * @param {number} vocabSize - Size of vocabulary
     * @param {number} embeddingDim - Embedding dimension
     * @param {WordEmbeddings} wordEmbeddingDb - Optional pre-trained embeddings
     */
    constructor(vocabSize, embeddingDim = 50, wordEmbeddingDb = null) {
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
        this.embeddings = [];
        this.wordEmbeddingDb = wordEmbeddingDb;
        
        // Initialize random embeddings
        this.initializeEmbeddings();
    }
    
    /**
     * Initialize embeddings with Xavier/Glorot distribution
     */
    initializeEmbeddings() {
        // Xavier scaling factor
        const scale = Math.sqrt(6 / (this.vocabSize + this.embeddingDim));
        
        this.embeddings = Array(this.vocabSize).fill().map(() => 
            Array(this.embeddingDim).fill().map(() => 
                (Math.random() * 2 - 1) * scale
            )
        );
    }
    
    /**
     * Load pre-trained embeddings from wordEmbeddingDb
     * @param {Map<string, number>} vocabulary - Map of word to index
     */
    async loadPretrainedEmbeddings(vocabulary) {
        if (!this.wordEmbeddingDb) return false;
        
        let loaded = 0;
        
        // For each word in vocabulary, try to load its pre-trained embedding
        for (const [word, index] of vocabulary.entries()) {
            try {
                const vector = await this.wordEmbeddingDb.getWordVector(word, false);
                
                // Only use if dimensions match
                if (vector.length === this.embeddingDim) {
                    this.embeddings[index] = [...vector];
                    loaded++;
                }
            } catch (e) {
                // Word not in pre-trained embeddings
            }
        }
        
        console.log(`Loaded ${loaded} pre-trained word embeddings`);
        return loaded > 0;
    }
    
    /**
     * Look up embeddings for word indices
     * @param {Array<number>} indices - Word indices
     * @returns {Array<Array<number>>} - Embeddings for the indices
     */
    lookup(indices) {
        return indices.map(idx => 
            idx >= 0 && idx < this.vocabSize ? 
                this.embeddings[idx] : 
                Array(this.embeddingDim).fill(0)
        );
    }
    
    /**
     * Forward pass: convert one-hot/index vectors to embeddings
     * @param {Array<number>} input - One-hot encoded input or array of indices
     * @returns {Array<number>} - Concatenated embedding vectors
     */
    forward(input) {
        // If input is one-hot encoded
        if (input.length === this.vocabSize) {
            // Find non-zero elements (could be multiple for context)
            const activeIndices = [];
            const weights = [];
            
            for (let i = 0; i < input.length; i++) {
                if (input[i] > 0) {
                    activeIndices.push(i);
                    weights.push(input[i]); // Preserve input weight
                }
            }
            
            // No active elements, return zero vector
            if (activeIndices.length === 0) {
                return Array(this.embeddingDim).fill(0);
            }
            
            // Look up embeddings and combine weighted
            const embeddings = this.lookup(activeIndices);
            const combined = Array(this.embeddingDim).fill(0);
            
            // Weighted sum of embeddings
            for (let i = 0; i < activeIndices.length; i++) {
                for (let j = 0; j < this.embeddingDim; j++) {
                    combined[j] += embeddings[i][j] * weights[i];
                }
            }
            
            return combined;
        }
        // If input is already indices
        else {
            // Look up embeddings and concatenate
            const embeddings = this.lookup(input);
            return embeddings.flat();
        }
    }
    
    /**
     * Update embeddings during training
     * @param {Array<number>} input - One-hot encoded input
     * @param {Array<number>} gradients - Gradients from next layer
     * @param {number} learningRate - Learning rate
     */
    backward(input, gradients, learningRate = 0.01) {
        // Find active indices (non-zero elements in one-hot)
        const activeIndices = [];
        for (let i = 0; i < input.length; i++) {
            if (input[i] > 0) {
                activeIndices.push(i);
            }
        }
        
        // No active elements, nothing to update
        if (activeIndices.length === 0) return;
        
        // Update embeddings for each active index
        for (const idx of activeIndices) {
            // Each embedding dimension gets updated based on corresponding gradient
            for (let d = 0; d < this.embeddingDim; d++) {
                this.embeddings[idx][d] -= learningRate * gradients[d];
            }
        }
    }
    
    /**
     * Resize the embedding layer for new vocabulary
     * @param {number} newVocabSize - New vocabulary size
     */
    resize(newVocabSize) {
        if (newVocabSize <= this.vocabSize) return;
        
        // Create new embeddings for added vocabulary
        const additionalEmbeddings = Array(newVocabSize - this.vocabSize).fill().map(() =>
            Array(this.embeddingDim).fill().map(() => 
                (Math.random() * 0.2 - 0.1) // Smaller range for new embeddings
            )
        );
        
        this.embeddings = [...this.embeddings, ...additionalEmbeddings];
        this.vocabSize = newVocabSize;
    }
}

module.exports = EmbeddingLayer;
