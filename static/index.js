// Handle image upload via click
document.getElementById('drop-area').addEventListener('click', () => {
    document.getElementById('file-input').click();
});

// Handle file input change
document.getElementById('file-input').addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        previewImage(file);
    }
});

// Handle pasting images
document.addEventListener('paste', (event) => {
    const items = (event.clipboardData || event.originalEvent.clipboardData).items;
    for (const item of items) {
        if (item.kind === 'file') {
            const file = item.getAsFile();
            previewImage(file);
            break;
        }
    }
});

// Preview uploaded or pasted image
function previewImage(file) {
    const img = document.getElementById('preview');
    const reader = new FileReader();
    reader.onload = (event) => {
        img.src = event.target.result;
        img.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

// Chatbot Functionality
async function sendMessage() {
    let inputField = document.getElementById("user-input");
    let userMessage = inputField.value.trim();
    if (!userMessage) return;

    let chatContent = document.getElementById("chat-content");
    chatContent.innerHTML += `<p><b>You:</b> ${userMessage}</p>`;

    const detectedDisease = document.getElementById("detected-disease").innerText;

    const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage, disease: detectedDisease })
    });

    const data = await response.json();
    chatContent.innerHTML += `<p><b>Bot:</b> ${data.response}</p>`;
    inputField.value = "";
}

// Send message on pressing Enter key
document.getElementById("user-input").addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
});
