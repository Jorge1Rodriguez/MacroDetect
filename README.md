# Macronutrient Analyzer

A web application that uses computer vision to segment food items in a photo and analyze their macronutrient distribution (proteins, carbohydrates, fats, vegetables, and others).

---

## Project Structure

```
Macronutirent_Analyzer/
│
├── backend/                  # FastAPI backend with PyTorch model
│   ├── main.py
│   └── utils/
│       ├── __init__.py
│       └── processing.py
|   └── model/
|       └── unet_resnet50.pth   # <-- Place the downloaded model here
│
├── frontend/                 # HTML/CSS/JS static frontend
│   ├── index.html
│   ├── analyze.html
│   ├── details.html
│   ├── about.html
│   ├── contact.html
│   ├── css/
│   │   └── styles.css
│   └── js/
│       ├── analyze.js
│       └── details.js
|   └── assets/
│       ├── aboutImage1.jpg
│       ├── aboutImage2.jpg
|       ├── aboutImage3.jpg
|       ├── HomeImage.jpg
|       ├── logo.png
|       └── qrgithub.png
│
|
│
├── requirements.txt
└── venv/                     # Python virtual environment (local)
```

---

## Requirements

- Python 3.9 or newer
- pip
- A modern web browser (Chrome, Firefox, Edge, etc.)

---

## 1. Clone the repository

```bash
git clone https://github.com/Jorge1Rodriguez/MacroDetect.git
cd macronutrient-analyzer
```

---

## 2. Create and activate a virtual environment

```bash
python -m venv venv
```

- On **Windows**:

```bash
venv\Scripts\activate
```

- On **macOS/Linux**:

```bash
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Download the model

Download the model file from the following link and place it inside the `/model` directory:

```
https://www.mediafire.com/file/nds8ricg0q9axpt/unet_resnet50.pth/file
```

---

## 5. Run the backend

From the root project directory:

- On **Windows**:

```bash
set PYTHONPATH=./backend
uvicorn backend.main:app --reload
```

- On **macOS/Linux**:

```bash
export PYTHONPATH=./backend
uvicorn backend.main:app --reload
```

Once running, the backend will be available at:

```
http://127.0.0.1:8000
```

---

## 6. Run the frontend

To test the frontend, open `frontend/index.html` in your browser manually **or** serve it using Python's built-in server:

```bash
cd frontend
python -m http.server 8080
```

Visit:

```
http://localhost:8080
```

> You can also use the **Live Server** extension in VS Code for live reloading.

---

## 7. How it works

1. Go to the "Analyze" page and upload a food image (JPG or PNG).
2. The backend processes the image with the AI model.
3. The UI shows the segmentation mask and a bar chart with macronutrient percentages.
4. Clicking on the chart navigates to a detailed view.
5. You can download the labeled segmentation mask from the "Details" page.

---


