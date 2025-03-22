const VectorDb = require('./vectorDb');

/**
 * Word embeddings manager that extends VectorDb for word-specific operations
 */
class WordEmbeddings extends VectorDb {
    constructor(options = {}) {
        super({
            dimensions: options.dimensions || 50,
            storageDir: options.storageDir || './embeddings'
        });
        this.defaultVector = options.defaultVector || new Array(this.dimensions).fill(0);
    }
    
    /**
     * Initialize the word embeddings system
     */
    async initialize() {
        try {
            // Make sure storage directory exists
            const fs = require('fs').promises;
            const path = require('path');
            await fs.mkdir(this.storageDir, { recursive: true });
            
            // Try to load existing embeddings
            try {
                await this.load();
            } catch (err) {
                console.log("No existing word embeddings found. Starting fresh.");
            }
            
            return true;
        } catch (error) {
            console.error("Error initializing word embeddings:", error);
            return false;
        }
    }
    
    /**
     * Get vector for a word, creating a random one if not exists
     * @param {string} word - The word to get or create vector for
     * @param {boolean} createIfMissing - Whether to create a missing vector
     * @returns {Promise<Array<number>>} - The word vector
     */
    async getWordVector(word, createIfMissing = true) {
        const result = await this.getVector(word);
        
        if (result) return result.vector;
        
        if (createIfMissing) {
            // Create a new random vector
            const newVector = Array(this.dimensions).fill(0).map(() => (Math.random() - 0.5) * 0.1);
            await this.addVector(word, newVector, { created: Date.now() });
            return newVector;
        }
        
        // Return zero vector if not found and not creating
        return [...this.defaultVector];
    }
    
    /**
     * Find similar words
     * @param {string} word - The word to find similar words for
     * @param {number} limit - Maximum number of results
     * @returns {Promise<Array<{word: string, similarity: number}>>} - Similar words
     */
    async findSimilarWords(word, limit = 10) {
        // First get the word vector
        const vector = await this.getWordVector(word, false);
        
        // Find similar vectors
        const results = await this.findSimilar(vector, { limit, minSimilarity: 0.5 });
        
        // Format results
        return results.map(item => ({
            word: item.key,
            similarity: item.similarity
        }));
    }
    
    /**
     * Update vocabulary with text processor's vocabulary
     * @param {TextProcessor} textProcessor - The text processor instance
     * @returns {Promise<number>} - Number of vectors added/updated
     */
    async updateFromVocabulary(textProcessor) {
        let count = 0;
        
        // Process all words in the vocabulary
        for (const [word] of textProcessor.vocabulary.entries()) {
            if (!this.vectors.has(word)) {
                // Create a new random vector
                const vector = Array(this.dimensions).fill(0).map(() => (Math.random() - 0.5) * 0.1);
                await this.addVector(word, vector, { source: 'vocabulary' });
                count++;
            }
        }
        
        return count;
    }
    
    /**
     * Find word vector centroids for a phrase
     * @param {string} phrase - The phrase to process
     * @returns {Promise<Array<number>>} - The centroid vector
     */
    async getPhraseVector(phrase) {
        if (!phrase || typeof phrase !== 'string') {
            return [...this.defaultVector];
        }
        
        const words = phrase.toLowerCase().trim().split(/\s+/);
        if (words.length === 0) return [...this.defaultVector];
        
        if (words.length === 1) {
            return this.getWordVector(words[0]);
        }
        
        // Get vectors for all words
        const vectors = await Promise.all(words.map(w => this.getWordVector(w, false)));
        
        // Calculate centroid vector
        const centroid = new Array(this.dimensions).fill(0);
        for (const vec of vectors) {
            for (let i = 0; i < this.dimensions; i++) {
                centroid[i] += vec[i] / words.length;
            }
        }
        
        return centroid;
    }
}

module.exports = WordEmbeddings;
