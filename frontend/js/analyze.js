const imageInput = document.getElementById("imageInput");
const resultsSection = document.getElementById("resultsSection");
const originalImage = document.getElementById("originalImage");
const maskImage = document.getElementById("maskImage");
const barsContainer = document.getElementById("barsContainer");
const analyzeBtn = document.getElementById("analyzeBtn");

const macroColors = {
  proteins: "#1f77b4",
  carbohydrates: "#ff7f0e",
  fats: "#2ca02c",
  vegetables: "#d62728",
  other: "#e377c2",
};

let selectedFile = null;

imageInput.addEventListener("change", () => {
  selectedFile = imageInput.files[0];
  if (!selectedFile) return;

  const reader = new FileReader();
  reader.onload = () => {
    originalImage.src = reader.result;
    resultsSection.style.display = "block";
    maskImage.src = "";
    barsContainer.innerHTML = "";
    analyzeBtn.disabled = false;
  };
  reader.readAsDataURL(selectedFile);
});


analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch("https://warrior-assembly-rand-ton.trycloudflare.com/predict/", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("Error al predecir imagen.");
    }

    const data = await response.json();
    maskImage.src = `data:image/png;base64,${data.mask}`;


    sessionStorage.setItem("originalImage", originalImage.src);
    sessionStorage.setItem("maskImage", maskImage.src);
    sessionStorage.setItem("percentages", JSON.stringify(data.percentages));


    barsContainer.innerHTML = "";
    Object.entries(data.percentages).forEach(([macro, percentage]) => {
      const bar = document.createElement("div");
      bar.className = "analysis-bar";
      bar.innerHTML = `
        <div class="bar-container">
          <div class="bar" style="height: ${percentage}%; background-color: ${macroColors[macro]}"></div>
        </div>
        <span class="bar-label">${macro.charAt(0).toUpperCase() + macro.slice(1)}</span>
      `;
      bar.addEventListener("click", () => {
        window.location.href = "details.html";
      });
      barsContainer.appendChild(bar);
    });
  } catch (error) {
    console.error(error);
    alert("Hubo un error al analizar la imagen.");
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze Meal";
  }
});
