document.getElementById("send-btn").addEventListener("click", sendMessage);

const chatMessages = document.getElementById("chat-messages");

function sendMessage() {
    const userInput = document.getElementById("user-input");
    const message = userInput.value.trim();

    if (message === "") return;

    // Add user message to the chat
    addMessageToChat(message, "user");

    // Clear input field
    userInput.value = "";

    // Fetch chatbot response
    fetch('/get-response', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
        addMessageToChat(data.response, "bot");
    })
    .catch(error => {
        console.error("Error:", error);
        addMessageToChat("Sorry, I encountered an error.", "bot");
    });
}

function addMessageToChat(message, type) {
    const messageElement = document.createElement("div");
    messageElement.classList.add("message", type);
    messageElement.textContent = message;
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
