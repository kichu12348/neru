# Neru - Teaching Guide

## How to Teach Neru

Neru learns through these key methods:

### 1. Conversation + Feedback
The primary way Neru learns is through conversations with you:

1. **Input text:** Provide clear, well-formed sentences
2. **Rate responses:** After each response, rate it from 1-5
   - Higher ratings (4-5) reinforce good responses
   - Lower ratings (1-2) help Neru learn what to avoid

Example:
```
You: What's the weather like today?
Neru: I'm still learning. Please teach me more!
Rate response (1-5, 5 being best): 3
```

### 2. Training Explicitly
After several conversations are stored in memory:

1. Type `train` at the prompt to process collected data
2. Neru will analyze past conversations and improve its model
3. Training is more effective with at least 10-15 conversation examples

```
You: train
Training model...
Epoch 0: loss = 1.2345, accuracy = 0.3456
...
Training complete!
```

### 3. Teaching Best Practices

- **Start simple:** Begin with short, clear sentences
- **Be consistent:** Rate similar responses similarly
- **Provide variety:** Give different types of inputs
- **Be patient:** The model improves gradually over time
- **Correct mistakes:** When Neru gives poor responses, rate them lower
- **Reward good responses:** When responses make sense, rate them higher

### 4. Memory System

All interactions are stored in `memory.json` and used for future learning. More data means better responses over time.
