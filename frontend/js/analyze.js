const uploadInput = document.getElementById("uploadInput");
const originalImage = document.getElementById("originalImage");
const maskImage = document.getElementById("maskImage");
const barsContainer = document.getElementById("barsContainer");

uploadInput.addEventListener("change", async () => {
  const file = uploadInput.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  originalImage.src = URL.createObjectURL(file);

  try {
    const response = await fetch("http://127.0.0.1:8000/predict/", {
      method: "POST",
      body: formData
    });

    const result = await response.json();
    maskImage.src = `data:image/png;base64,${result.mask_image}`;

    barsContainer.innerHTML = "";
    for (const [macro, percent] of Object.entries(result.percentages)) {
      const bar = document.createElement("div");
      bar.className = "analysis-bar";
      bar.innerHTML = `
        <div class="bar-container">
          <div class="bar" style="height: ${percent}%;"></div>
        </div>
        <span class="bar-label">${macro}</span>
      `;
      barsContainer.appendChild(bar);
    }
  } catch (err) {
    alert("Prediction failed.");
    console.error(err);
  }
});
