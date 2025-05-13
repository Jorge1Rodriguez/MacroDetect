document.getElementById('upload-form').addEventListener('submit', async function (e) {
    e.preventDefault();

    const fileInput = document.getElementById('file');
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/predict/', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();

    // Mostrar la máscara
    const maskImage = document.getElementById('mask-image');
    maskImage.src = `data:image/png;base64,${result.mask}`;

    // Mostrar los porcentajes
    const output = document.getElementById('output');
    output.innerHTML = "<h3>Macronutrient Breakdown</h3>";
    for (let [macro, value] of Object.entries(result.percentages)) {
        output.innerHTML += `<p><strong>${macro}:</strong> ${value}%</p>`;
    }
});
