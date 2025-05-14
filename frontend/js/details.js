const originalDetailImage = document.getElementById("originalDetailImage");
const maskDetailImage = document.getElementById("maskDetailImage");
const percentagesText = document.getElementById("percentagesText");
const downloadBtn = document.getElementById("downloadBtn");

const macroColors = {
  proteins: "#1f77b4",
  carbohydrates: "#ff7f0e",
  fats: "#2ca02c",
  vegetables: "#d62728",
  other: "#e377c2",
};

function generateLegendedMask(base64Img, percentages) {
  return new Promise((resolve) => {
    const image = new Image();
    image.src = base64Img;
    image.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = image.width;
      canvas.height = image.height + 50;
      const ctx = canvas.getContext("2d");

      ctx.drawImage(image, 0, 0);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, image.height, canvas.width, 50);

      let x = 10;
      Object.entries(percentages).forEach(([macro]) => {
        ctx.fillStyle = macroColors[macro];
        ctx.fillRect(x, image.height + 15, 10, 10);
        ctx.fillStyle = "#000";
        ctx.font = "12px Arial";
        ctx.fillText(macro, x + 15, image.height + 25);
        x += ctx.measureText(macro).width + 45;
      });

      resolve(canvas.toDataURL("image/png"));
    };
  });
}

window.addEventListener("DOMContentLoaded", async () => {
  const original = sessionStorage.getItem("originalImage");
  const mask = sessionStorage.getItem("maskImage");
  const percentages = JSON.parse(sessionStorage.getItem("percentages"));

  if (!original || !mask || !percentages) {
    alert("No hay datos disponibles. Por favor analiza una imagen primero.");
    window.location.href = "analyze.html";
    return;
  }

  originalDetailImage.src = original;
  maskDetailImage.src = mask;

  const parts = Object.entries(percentages).map(
    ([macro, perc]) => `${macro.charAt(0).toUpperCase() + macro.slice(1)}: ${perc}%`
  );
  percentagesText.textContent = parts.join(" • ");

  downloadBtn.addEventListener("click", async () => {
    const result = await generateLegendedMask(mask, percentages);
    const a = document.createElement("a");
    a.href = result;
    a.download = "analysis_result.png";
    a.click();
  });
});
