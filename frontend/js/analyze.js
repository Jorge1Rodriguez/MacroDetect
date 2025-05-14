const imageInput = document.getElementById("imageInput");
const resultsSection = document.getElementById("resultsSection");
const originalImage = document.getElementById("originalImage");
const maskImage = document.getElementById("maskImage");
const barsContainer = document.getElementById("barsContainer");

const macroColors = {
  proteins: "#1f77b4",
  carbohydrates: "#ff7f0e",
  fats: "#2ca02c",
  vegetables: "#d62728",
  other: "#e377c2",
};

imageInput.addEventListener("change", async () => {
  const file = imageInput.files[0];
  if (!file) return;

  // Mostrar imagen original
  const reader = new FileReader();
  reader.onload = () => {
    originalImage.src = reader.result;
    resultsSection.style.display = "block";
  };
  reader.readAsDataURL(file);

  // Enviar imagen al backend
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("http://127.0.0.1:8000/predict/", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("Error al predecir imagen.");
    }

    const data = await response.json();
    maskImage.src = `data:image/png;base64,${data.mask}`;

    // Guardar para la vista de detalles
    sessionStorage.setItem("originalImage", originalImage.src);
    sessionStorage.setItem("maskImage", maskImage.src);
    sessionStorage.setItem("percentages", JSON.stringify(data.percentages));

    // Renderizar barras
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
  }
});
