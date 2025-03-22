// Training data in explicit [input, [output1, output2, ...]] format

const trainingData = [
    // Greetings - Multiple responses for variety
    ["hello", ["hi", "hey", "greetings"]],
    ["hi", ["hello", "hey there"]],
    ["hey", ["hi", "hello"]],
    ["what's up", ["not much", "just chatting"]],
    ["how are you", ["I'm doing well", "I'm good, how about you?"]],
    ["good morning", ["morning!", "good morning!"]],
    ["good evening", ["evening!", "hope your evening is great!"]],

    // Small Talk
    ["what's new", ["not much, how about you?", "just learning new things"]],
    ["how's your day", ["pretty good!", "same as usual"]],
    ["how's it going", ["going well!", "doing fine"]],

    // Identity & Capabilities
    ["what is your name", ["I'm a chatbot!", "Just call me AI."]],
    ["who are you", ["I'm an AI assistant!", "I'm a chatbot trained to chat."]],
    ["what can you do", ["I can chat, answer questions, and learn from you!"]],
    ["are you smart", ["I'm still learning!", "I try my best."]],
    ["how do you work", ["I use machine learning to understand and respond."]],

    // Thank you & Apologies
    ["thank you", ["you're welcome!", "no problem!"]],
    ["thanks", ["anytime!", "glad to help!"]],
    ["sorry", ["it's okay!", "no worries!"]],
    ["excuse me", ["yes?", "I'm listening"]],

    // Goodbye
    ["goodbye", ["bye!", "see you later!"]],
    ["bye", ["goodbye!", "take care!"]],
    
    // Emotions - More natural responses
    ["I'm happy", ["That's great!", "Awesome!"]],
    ["I'm sad", ["I'm here for you.", "Want to talk about it?"]],
    ["I'm angry", ["Take a deep breath.", "Let's talk it out."]],
    ["I'm tired", ["Maybe get some rest?", "You should relax."]],
    ["I'm excited", ["That's awesome!", "Tell me more!"]],

    // Topics - Keeps conversation going
    ["weather", ["How's the weather near you?", "It's always sunny in the cloud."]],
    ["music", ["I love all kinds of music! What's your favorite?"]],
    ["sports", ["Are you into sports?", "I don't play, but I like hearing about them."]],
    ["food", ["Food is great! What's your favorite dish?"]],
    ["movies", ["Seen any good movies lately?", "I enjoy movie discussions!"]],
    ["books", ["Do you like reading?", "Books are fascinating!"]],

    // Clarifications & Follow-ups
    ["what", ["Could you clarify?", "I didn't catch that."]],
    ["why", ["That's a good question.", "Let me think about it."]],
    ["when", ["Time is tricky!", "Can you be more specific?"]],
    ["where", ["Somewhere interesting!", "It depends!"]],
    ["how", ["It's a process!", "Good question!"]],

    // Affirmation & Denial
    ["yes", ["Okay!", "Got it!"]],
    ["no", ["Alright!", "Let's try something else."]],
    ["maybe", ["That's fair.", "Possibly."]],
    ["sure", ["Of course!", "Absolutely!"]],

    // Internet Slang & Casual Talk
    ["lol", ["Haha!", "That's funny!"]],
    ["omg", ["Wow!", "That's surprising!"]],
    ["wtf", ["Oh wow!", "That's unexpected!"]],
    ["haha", ["Laughing!", "Good one!"]],
    
    // Time-related
    ["today", ["Right now!", "This very moment."]],
    ["tomorrow", ["Soon enough!", "Just a day away."]],
    ["yesterday", ["In the past!", "Not too long ago."]],

    // Technical & AI-Related
    ["computer", ["A machine that computes!", "Tech is fascinating."]],
    ["program", ["Software that runs!", "Are you into coding?"]],
    ["code", ["Programming is cool!", "What language do you like?"]],
    ["data", ["Information is power!", "I learn from data!"]],
    ["machine learning", ["AI learns from patterns!", "That's what makes me smarter!"]],

    // Handling Unknown Inputs
    ["*", ["I don't understand, but I'm learning!", "Could you rephrase that?"]]
];

module.exports = trainingData;
