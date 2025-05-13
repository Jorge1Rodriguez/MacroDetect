const input = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const result = document.getElementById("result");

input.addEventListener("change", async () => {
  const file = input.files[0];
  if (!file) return;

  preview.innerHTML = `<img src="${URL.createObjectURL(file)}" style="max-width:300px"/>`;

  const formData = new FormData();
  formData.append("file", file);

  result.innerHTML = "Analyzing...";

  const res = await fetch("http://127.0.0.1:8000/predict/", {
    method: "POST",
    body: formData,
  });

  const data = await res.json();

  result.innerHTML = "<h3>Macronutrient Breakdown:</h3>";
  for (const [macro, value] of Object.entries(data.percentages)) {
    result.innerHTML += `<p>${macro}: ${value}%</p>`;
  }
});
