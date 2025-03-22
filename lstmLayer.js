/**
 * LSTM-inspired layer for sequence processing
 */
class LSTMLayer {
    /**
     * Create an LSTM layer
     * @param {number} inputSize - Size of input
     * @param {number} hiddenSize - Size of hidden state
     * @param {number} dropoutRate - Dropout rate (0-1)
     */
    constructor(inputSize, hiddenSize, dropoutRate = 0.2) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.dropoutRate = dropoutRate;
        
        // Initialize weights with Xavier/Glorot distribution
        const scale = Math.sqrt(6 / (inputSize + hiddenSize));
        
        // Input gate weights
        this.Wi = Array(hiddenSize).fill().map(() => 
            Array(inputSize).fill().map(() => (Math.random() * 2 - 1) * scale)
        );
        
        // Forget gate weights
        this.Wf = Array(hiddenSize).fill().map(() => 
            Array(inputSize).fill().map(() => (Math.random() * 2 - 1) * scale)
        );
        
        // Cell update weights
        this.Wc = Array(hiddenSize).fill().map(() => 
            Array(inputSize).fill().map(() => (Math.random() * 2 - 1) * scale)
        );
        
        // Output gate weights
        this.Wo = Array(hiddenSize).fill().map(() => 
            Array(inputSize).fill().map(() => (Math.random() * 2 - 1) * scale)
        );
        
        // Hidden state recurrent weights
        this.Ui = Array(hiddenSize).fill().map(() => 
            Array(hiddenSize).fill().map(() => (Math.random() * 2 - 1) * scale * 0.5)
        );
        
        this.Uf = Array(hiddenSize).fill().map(() => 
            Array(hiddenSize).fill().map(() => (Math.random() * 2 - 1) * scale * 0.5)
        );
        
        this.Uc = Array(hiddenSize).fill().map(() => 
            Array(hiddenSize).fill().map(() => (Math.random() * 2 - 1) * scale * 0.5)
        );
        
        this.Uo = Array(hiddenSize).fill().map(() => 
            Array(hiddenSize).fill().map(() => (Math.random() * 2 - 1) * scale * 0.5)
        );
        
        // Biases
        this.bi = Array(hiddenSize).fill(0);
        this.bf = Array(hiddenSize).fill(1.0); // Initialize forget bias to 1 (important!)
        this.bc = Array(hiddenSize).fill(0);
        this.bo = Array(hiddenSize).fill(0);
        
        // State for forward pass
        this.lastInput = null;
        this.lastHidden = null;
        this.lastCell = null;
        
        // Intermediate states
        this.gates = { i: null, f: null, c: null, o: null };
        
        // Reset states
        this.resetState();
    }
    
    /**
     * Reset LSTM state
     */
    resetState() {
        this.lastHidden = Array(this.hiddenSize).fill(0);
        this.lastCell = Array(this.hiddenSize).fill(0);
    }
    
    /**
     * Sigmoid activation function
     * @param {number} x - Input value
     * @returns {number} - Sigmoid output
     */
    sigmoid(x) {
        // Avoid numerical instability
        if (x < -16) return 0;
        if (x > 16) return 1;
        return 1 / (1 + Math.exp(-x));
    }
    
    /**
     * Tanh activation function
     * @param {number} x - Input value
     * @returns {number} - Tanh output
     */
    tanh(x) {
        // Avoid numerical instability
        if (x < -16) return -1;
        if (x > 16) return 1;
        const exp2x = Math.exp(2 * x);
        return (exp2x - 1) / (exp2x + 1);
    }
    
    /**
     * Matrix multiplication
     * @param {Array<Array<number>>} A - Matrix A
     * @param {Array<number>} x - Vector x
     * @returns {Array<number>} - Result vector
     */
    matmul(A, x) {
        const result = Array(A.length).fill(0);
        for (let i = 0; i < A.length; i++) {
            for (let j = 0; j < x.length; j++) {
                result[i] += A[i][j] * x[j];
            }
        }
        return result;
    }
    
    /**
     * Apply dropout to a vector
     * @param {Array<number>} x - Input vector
     * @param {number} rate - Dropout rate (0-1)
     * @param {boolean} training - Whether in training mode
     * @returns {Array<number>} - Output with dropout applied
     */
    applyDropout(x, rate, training) {
        if (!training || rate === 0) return x;
        
        // Create dropout mask
        const mask = Array(x.length).fill().map(() => 
            Math.random() > rate ? 1 / (1 - rate) : 0
        );
        
        // Apply mask
        return x.map((val, i) => val * mask[i]);
    }
    
    /**
     * Forward pass
     * @param {Array<number>} input - Input vector
     * @param {boolean} training - Whether in training mode
     * @returns {Array<number>} - Output vector
     */
    forward(input, training = true) {
        this.lastInput = input;
        
        // Apply dropout to input
        const droppedInput = this.applyDropout(input, this.dropoutRate, training);
        const h_prev = this.lastHidden;
        const c_prev = this.lastCell;
        
        // Input gate
        const i_g = this.matmul(this.Wi, droppedInput)
            .map((val, idx) => val + this.matmul(this.Ui, h_prev)[idx] + this.bi[idx])
            .map(x => this.sigmoid(x));
        
        // Forget gate
        const f_g = this.matmul(this.Wf, droppedInput)
            .map((val, idx) => val + this.matmul(this.Uf, h_prev)[idx] + this.bf[idx])
            .map(x => this.sigmoid(x));
        
        // Cell update
        const c_tilda = this.matmul(this.Wc, droppedInput)
            .map((val, idx) => val + this.matmul(this.Uc, h_prev)[idx] + this.bc[idx])
            .map(x => this.tanh(x));
        
        // Cell state
        const c_t = c_prev.map((val, idx) => f_g[idx] * val + i_g[idx] * c_tilda[idx]);
        
        // Output gate
        const o_g = this.matmul(this.Wo, droppedInput)
            .map((val, idx) => val + this.matmul(this.Uo, h_prev)[idx] + this.bo[idx])
            .map(x => this.sigmoid(x));
        
        // Hidden state
        const h_t = c_t.map((val, idx) => o_g[idx] * this.tanh(val));
        
        // Store gates for backpropagation
        this.gates = { i: i_g, f: f_g, c_new: c_tilda, o: o_g };
        
        // Update state
        this.lastHidden = h_t;
        this.lastCell = c_t;
        
        // Apply dropout to output during training
        return this.applyDropout(h_t, this.dropoutRate, training);
    }
    
    /**
     * Simplified backpropagation for the LSTM layer
     * @param {Array<number>} gradOutput - Gradient from next layer
     * @param {number} learningRate - Learning rate
     * @returns {Array<number>} - Gradient to be propagated to previous layer
     */
    backward(gradOutput, learningRate = 0.01) {
        // This is a simplified version of LSTM backpropagation
        // For a production system, you'd want a more complex implementation
        
        // Calculate gradients for the hidden state
        const gradHidden = gradOutput;
        
        // Calculate gradients for the cell state
        const gradCell = gradHidden.map((val, idx) => 
            val * this.gates.o[idx] * (1 - Math.pow(this.tanh(this.lastCell[idx]), 2))
        );
        
        // Calculate gradients for the gates
        const gradO = gradHidden.map((val, idx) => 
            val * this.tanh(this.lastCell[idx]) * this.gates.o[idx] * (1 - this.gates.o[idx])
        );
        
        const gradI = gradCell.map((val, idx) => 
            val * this.gates.c_new[idx] * this.gates.i[idx] * (1 - this.gates.i[idx])
        );
        
        const gradF = gradCell.map((val, idx) => 
            val * this.lastCell[idx - 1] * this.gates.f[idx] * (1 - this.gates.f[idx])
        );
        
        const gradCNew = gradCell.map((val, idx) => 
            val * this.gates.i[idx] * (1 - Math.pow(this.gates.c_new[idx], 2))
        );
        
        // Calculate input gradients
        const gradInput = Array(this.inputSize).fill(0);
        
        // Add contributions from all gates
        for (let i = 0; i < this.inputSize; i++) {
            for (let h = 0; h < this.hiddenSize; h++) {
                gradInput[i] += gradI[h] * this.Wi[h][i];
                gradInput[i] += gradF[h] * this.Wf[h][i];
                gradInput[i] += gradCNew[h] * this.Wc[h][i];
                gradInput[i] += gradO[h] * this.Wo[h][i];
            }
        }
        
        // Update weights (simplified)
        for (let h = 0; h < this.hiddenSize; h++) {
            for (let i = 0; i < this.inputSize; i++) {
                this.Wi[h][i] -= learningRate * gradI[h] * this.lastInput[i];
                this.Wf[h][i] -= learningRate * gradF[h] * this.lastInput[i];
                this.Wc[h][i] -= learningRate * gradCNew[h] * this.lastInput[i];
                this.Wo[h][i] -= learningRate * gradO[h] * this.lastInput[i];
            }
            
            // Update biases
            this.bi[h] -= learningRate * gradI[h];
            this.bf[h] -= learningRate * gradF[h];
            this.bc[h] -= learningRate * gradCNew[h];
            this.bo[h] -= learningRate * gradO[h];
        }
        
        return gradInput;
    }
}

module.exports = LSTMLayer;
