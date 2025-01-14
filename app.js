async function classifyImage() {
    const fileInput = document.getElementById('fileInput');
    if (!fileInput.files.length) {
        alert('Please select an image file.');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    const response = await fetch('https://<your-api-url>/classify', {
        method: 'POST',
        body: formData,
    });

    const result = await response.json();
    document.getElementById('result').textContent = `Category: ${result.category}, Confidence: ${(result.confidence * 100).toFixed(2)}%`;
}
