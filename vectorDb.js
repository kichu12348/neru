const fs = require('fs').promises;
const path = require('path');

/**
 * Simple in-memory vector database with async operations
 */
class VectorDb {
    constructor(options = {}) {
        this.vectors = new Map(); // Main storage: key -> vector
        this.metadata = new Map(); // Additional metadata: key -> object
        this.dimensions = options.dimensions || null; // Optional fixed dimensions
        this.storageDir = options.storageDir || './vectordb';
        this.indexedKeys = new Set(); // Keys that are added to search index
        this.ready = Promise.resolve(); // Ready state tracker for async operations
    }

    /**
     * Add a vector to the database
     * @param {string} key - Unique identifier
     * @param {Array<number>} vector - The vector to store
     * @param {Object} metadata - Optional metadata to store with the vector
     * @returns {Promise<boolean>} - Success status
     */
    async addVector(key, vector, metadata = {}) {
        return new Promise((resolve) => {
            // Validate vector dimensions if specified
            if (this.dimensions && vector.length !== this.dimensions) {
                console.warn(`Vector dimension mismatch: expected ${this.dimensions}, got ${vector.length}`);
                // Pad or truncate to match dimensions
                if (vector.length < this.dimensions) {
                    vector = [...vector, ...Array(this.dimensions - vector.length).fill(0)];
                } else {
                    vector = vector.slice(0, this.dimensions);
                }
            }
            
            // Store vector and metadata
            this.vectors.set(key, vector);
            this.metadata.set(key, metadata);
            this.indexedKeys.add(key);
            resolve(true);
        });
    }

    /**
     * Add multiple vectors in batch
     * @param {Array<{key: string, vector: Array<number>, metadata: Object}>} items 
     * @returns {Promise<number>} - Number of vectors added
     */
    async addVectors(items) {
        return new Promise((resolve) => {
            let added = 0;
            
            for (const item of items) {
                if (item.key && item.vector) {
                    this.vectors.set(item.key, item.vector);
                    this.metadata.set(item.key, item.metadata || {});
                    this.indexedKeys.add(item.key);
                    added++;
                }
            }
            
            resolve(added);
        });
    }

    /**
     * Get a vector by key
     * @param {string} key - The vector identifier
     * @returns {Promise<{vector: Array<number>, metadata: Object}|null>}
     */
    async getVector(key) {
        return new Promise((resolve) => {
            if (this.vectors.has(key)) {
                resolve({
                    vector: this.vectors.get(key),
                    metadata: this.metadata.get(key) || {}
                });
            } else {
                resolve(null);
            }
        });
    }

    /**
     * Delete a vector
     * @param {string} key - The vector identifier
     * @returns {Promise<boolean>} - Success status
     */
    async deleteVector(key) {
        return new Promise((resolve) => {
            const existed = this.vectors.has(key);
            this.vectors.delete(key);
            this.metadata.delete(key);
            this.indexedKeys.delete(key);
            resolve(existed);
        });
    }

    /**
     * Calculate cosine similarity between two vectors
     * @param {Array<number>} a - First vector
     * @param {Array<number>} b - Second vector
     * @returns {number} - Similarity score (1=identical, 0=orthogonal, -1=opposite)
     */
    cosineSimilarity(a, b) {
        // Handle empty vectors
        if (a.length === 0 || b.length === 0) return 0;
        
        // Ensure vectors are the same length
        const len = Math.min(a.length, b.length);
        
        // Calculate dot product
        let dotProduct = 0;
        let magnitudeA = 0;
        let magnitudeB = 0;
        
        for (let i = 0; i < len; i++) {
            dotProduct += a[i] * b[i];
            magnitudeA += a[i] * a[i];
            magnitudeB += b[i] * b[i];
        }
        
        magnitudeA = Math.sqrt(magnitudeA);
        magnitudeB = Math.sqrt(magnitudeB);
        
        // Avoid division by zero
        if (magnitudeA === 0 || magnitudeB === 0) return 0;
        
        return dotProduct / (magnitudeA * magnitudeB);
    }

    /**
     * Find nearest neighbors to a query vector
     * @param {Array<number>} queryVector - The query vector
     * @param {Object} options - Search options
     * @param {number} options.limit - Maximum results (default: 10)
     * @param {number} options.minSimilarity - Minimum similarity score (default: 0)
     * @returns {Promise<Array<{key: string, similarity: number, vector: Array<number>, metadata: Object}>>}
     */
    async findSimilar(queryVector, options = {}) {
        const limit = options.limit || 10;
        const minSimilarity = options.minSimilarity || 0;
        
        return new Promise((resolve) => {
            const results = [];
            
            // Compare with all vectors
            for (const key of this.indexedKeys) {
                const vector = this.vectors.get(key);
                const similarity = this.cosineSimilarity(queryVector, vector);
                
                if (similarity >= minSimilarity) {
                    results.push({
                        key,
                        similarity,
                        vector,
                        metadata: this.metadata.get(key) || {}
                    });
                }
            }
            
            // Sort by similarity (highest first)
            results.sort((a, b) => b.similarity - a.similarity);
            
            // Return top results
            resolve(results.slice(0, limit));
        });
    }

    /**
     * Find nearest neighbors to a vector identified by key
     * @param {string} key - The key of the vector to find neighbors for
     * @param {Object} options - Search options
     * @returns {Promise<Array<{key: string, similarity: number, metadata: Object}>>}
     */
    async findSimilarByKey(key, options = {}) {
        const vector = this.vectors.get(key);
        if (!vector) {
            return Promise.resolve([]);
        }
        return this.findSimilar(vector, options);
    }

    /**
     * Save the database to a JSON file
     * @param {string} filename - Optional filename (default: vectordb.json)
     * @returns {Promise<void>}
     */
    async save(filename = 'vectordb.json') {
        const filePath = path.join(this.storageDir, filename);
        
        // Ensure directory exists
        await fs.mkdir(this.storageDir, { recursive: true });
        
        // Convert maps to serializable format
        const data = {
            dimensions: this.dimensions,
            vectors: Array.from(this.vectors.entries()),
            metadata: Array.from(this.metadata.entries()),
            indexedKeys: Array.from(this.indexedKeys)
        };
        
        // Write to file
        await fs.writeFile(filePath, JSON.stringify(data, null, 2));
        return filePath;
    }

    /**
     * Load the database from a JSON file
     * @param {string} filename - Optional filename (default: vectordb.json)
     * @returns {Promise<void>}
     */
    async load(filename = 'vectordb.json') {
        const filePath = path.join(this.storageDir, filename);
        
        try {
            const data = JSON.parse(await fs.readFile(filePath, 'utf8'));
            
            this.dimensions = data.dimensions;
            this.vectors = new Map(data.vectors);
            this.metadata = new Map(data.metadata);
            this.indexedKeys = new Set(data.indexedKeys);
            
            return true;
        } catch (err) {
            if (err.code === 'ENOENT') {
                console.log(`Database file ${filePath} not found. Starting with empty database.`);
                return false;
            } else {
                throw err;
            }
        }
    }

    /**
     * Get database statistics
     * @returns {Promise<Object>} - Statistics about the database
     */
    async getStats() {
        return {
            vectorCount: this.vectors.size,
            indexedCount: this.indexedKeys.size,
            dimensions: this.dimensions,
            memoryUsage: this.estimateMemoryUsage()
        };
    }

    /**
     * Estimate memory usage in bytes
     * @returns {number} - Estimated bytes used
     */
    estimateMemoryUsage() {
        let total = 0;
        
        // Vectors (estimate: 8 bytes per number + overhead)
        for (const vector of this.vectors.values()) {
            total += vector.length * 8 + 40;
        }
        
        // Metadata (rough estimate)
        for (const meta of this.metadata.values()) {
            total += JSON.stringify(meta).length * 2;
        }
        
        // Keys (estimate: 2 bytes per character + overhead)
        for (const key of this.vectors.keys()) {
            total += key.length * 2 + 16;
        }
        
        return total;
    }
}

module.exports = VectorDb;
