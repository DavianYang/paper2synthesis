const fileInput = document.getElementById('fileInput');
const modelSelect = document.getElementById('modelSelect');
const uploadBtn = document.getElementById('uploadBtn');
const loadingDiv = document.getElementById('loading');
const resultMessage = document.getElementById('resultMessage');

async function fetchData(endpoint, formData, selectedModel) {
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }

        const data = await response.json();
        updateResultMessage(data, selectedModel);
    } catch (error) {
        displayError();
    } finally {
        hideLoadingIndicator();
    }
}

function updateResultMessage(data, selectedModel) {
    if (selectedModel === 'binary') {
        const isPositive = data.result === 1;
        resultMessage.textContent = isPositive ? 'Yes' : 'No';
        resultMessage.className = `result-box ${isPositive ? 'result-green' : 'result-red'}`;
    } else {
        const labels = {
            0: "Coastal and marine Ecosystems",
            1: "Human and managed",
            2: "Mountains, snow and ice",
            3: "Rivers, lakes, and soil moisture",
            4: "Terrestrial ES",
        };

        const result = data.result.map(element => labels[element] || "Unknown");
        resultMessage.textContent = result.join(', ');
        resultMessage.className = 'result-box result-green';
    }
}

function displayError() {
    resultMessage.textContent = 'Error';
    resultMessage.className = 'result-box result-red';
}

function hideLoadingIndicator() {
    loadingDiv.classList.add('hidden');
}

uploadBtn.addEventListener('click', async () => {
    resultMessage.textContent = 'Pending';
    resultMessage.className = 'result-box result-gray';
    loadingDiv.classList.remove('hidden');

    if (!fileInput.files[0]) {
        alert('Please select a file before uploading.');
        loadingDiv.classList.add('hidden');
        return;
    }

    const selectedModel = modelSelect.value;
    const endpoint = selectedModel === 'binary' 
        ? 'http://127.0.0.1:8000/predict_binary/' 
        : 'http://127.0.0.1:8000/predict_multilabel/';

    const formData = new FormData();
    formData.append('uploaded_file', fileInput.files[0]);

    await fetchData(endpoint, formData, selectedModel);
});