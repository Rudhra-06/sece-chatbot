document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');
    const modal = document.getElementById('score-modal');
    const closeModal = document.querySelector('.close-modal');
    const scoresContainer = document.getElementById('scores-container');

    let currentRetrievalData = [];

    const addMessage = (text, sender) => {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender);
        
        const contentDiv = document.createElement('div');
        contentDiv.classList.add('message-content');
        contentDiv.textContent = text;
        messageDiv.appendChild(contentDiv);

        if (sender === 'bot' && currentRetrievalData.length > 0) {
            const scoreBtn = document.createElement('span');
            scoreBtn.classList.add('score-trigger');
            scoreBtn.textContent = 'View retrieval sources & scores';
            scoreBtn.onclick = () => showScores(currentRetrievalData);
            messageDiv.appendChild(scoreBtn);
        }

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    };

    const showScores = (data) => {
        scoresContainer.innerHTML = '';
        data.forEach((item, index) => {
            const card = document.createElement('div');
            card.classList.add('chunk-card');
            card.innerHTML = `
                <div class="score-tag">Similarity Score: ${item.score.toFixed(4)}</div>
                <p>${item.text}</p>
            `;
            scoresContainer.appendChild(card);
        });
        modal.style.display = 'block';
    };

    closeModal.onclick = () => modal.style.display = 'none';
    window.onclick = (event) => {
        if (event.target == modal) modal.style.display = 'none';
    };

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = userInput.value.trim();
        if (!query) return;

        addMessage(query, 'user');
        userInput.value = '';

        // Typing indicator
        const typingDiv = document.createElement('div');
        typingDiv.classList.add('message', 'bot', 'typing');
        typingDiv.textContent = 'Analyzing knowledge base...';
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            });

            const data = await response.json();
            chatMessages.removeChild(typingDiv);

            if (data.answer) {
                currentRetrievalData = data.retrieval_results;
                addMessage(data.answer, 'bot');
            } else {
                addMessage("I'm sorry, I encountered an error. Please try again later.", 'bot');
            }
        } catch (error) {
            chatMessages.removeChild(typingDiv);
            addMessage("Error connecting to the server. Refer to sece.ac.in for info.", 'bot');
            console.error(error);
        }
    });
});
