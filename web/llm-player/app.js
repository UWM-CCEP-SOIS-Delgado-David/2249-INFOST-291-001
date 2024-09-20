// app.js

document.addEventListener('DOMContentLoaded', async () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatHistory = document.getElementById('chat-history');
  
    // Display loading message
    appendMessage('bot', 'Loading model, please wait...');
  
    // Load the tokenizer and model
    let tokenizer, model;
    try {
      tokenizer = await window.transformers.AutoTokenizer.from_pretrained('microsoft/Phi-3.5-mini-instruct-onnx');
      model = await window.transformers.AutoModelForCausalLM.from_pretrained('microsoft/Phi-3.5-mini-instruct-onnx', {
        provider: 'webgl' // Use 'webgl' for GPU acceleration if available
      });
    } catch (error) {
      console.error('Error loading model:', error);
      appendMessage('bot', 'Failed to load the chatbot model. Please try again later.');
      return;
    }
  
    // Remove loading message
    removeMessage('Loading model, please wait...');
  
    // Initialize conversation history
    let conversation = [];
  
    // Function to append messages to the chat history
    function appendMessage(sender, text) {
      const messageElem = document.createElement('div');
      messageElem.classList.add('flex', 'flex-col', sender === 'user' ? 'items-end' : 'items-start');
      
      const bubble = document.createElement('div');
      bubble.classList.add(
        'rounded-lg',
        'px-4',
        'py-2',
        'max-w-xs',
        sender === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-300 text-gray-800'
      );
      bubble.innerText = text;
  
      messageElem.appendChild(bubble);
      chatHistory.appendChild(messageElem);
      chatHistory.scrollTop = chatHistory.scrollHeight;
    }
  
    // Function to remove a specific message
    function removeMessage(text) {
      const messages = chatHistory.querySelectorAll('div > div');
      messages.forEach((msg) => {
        if (msg.innerText === text) {
          msg.parentElement.remove();
        }
      });
    }
  
    // Function to generate bot response
    async function generateResponse(userText) {
      // Append user message to conversation
      conversation.push({ role: 'user', content: userText });
  
      // Limit conversation history
      const MAX_HISTORY = 6; // Adjust as needed
      const recentConversation = conversation.slice(-MAX_HISTORY);
  
      // Prepare the prompt
      const prompt = recentConversation
        .map(msg => `${msg.role === 'user' ? 'User' : 'Bot'}: ${msg.content}`)
        .join('\n') + '\nBot:';
  
      // Tokenize input
      let inputs;
      try {
        inputs = await tokenizer.encode(prompt, { add_special_tokens: true });
      } catch (error) {
        console.error('Error tokenizing input:', error);
        return "I'm sorry, I couldn't process your input.";
      }
  
      // Convert inputs to tensor
      const inputTensor = window.transformers.tensorFromArray(inputs, 'int32').reshape([1, inputs.length]);
  
      // Generate response
      let outputs;
      try {
        outputs = await model.generate(inputTensor, {
          max_new_tokens: 50,
          temperature: 0.7,
          top_p: 0.9,
          do_sample: true,
          eos_token_id: tokenizer.eos_token_id
        });
      } catch (error) {
        console.error('Error generating response:', error);
        return "I'm sorry, I couldn't generate a response.";
      }
  
      // Decode the output tokens
      let responseText;
      try {
        responseText = await tokenizer.decode(outputs[0], { skip_special_tokens: true });
      } catch (error) {
        console.error('Error decoding response:', error);
        return "I'm sorry, I couldn't decode the response.";
      }
  
      // Extract the bot response
      const botResponse = responseText.split('Bot:')[1]?.trim() || "I'm not sure how to respond to that.";
  
      // Append bot response to conversation
      conversation.push({ role: 'bot', content: botResponse });
  
      return botResponse;
    }
  
    // Handle form submission
    chatForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const text = userInput.value.trim();
      if (text === '') return;
  
      // Append user message to chat history
      appendMessage('user', text);
      userInput.value = '';
  
      // Show a typing indicator
      appendMessage('bot', 'Typing...');
  
      // Generate and append bot response
      const botReply = await generateResponse(text);
  
      // Remove the typing indicator
      removeMessage('Typing...');
  
      appendMessage('bot', botReply);
    });
  });
  