# 🎮 TicTacBot

TicTacBot is an AI-powered physical Tic-Tac-Toe game. It uses YOLO for grid detection, Teachable Machine for piece classification, and a smart Python AI engine to challenge you turn-by-turn via a sleek Streamlit web app!

---

## ✨ Features
* **Live Camera or File Upload:** Capture your physical board live using your webcam/upload a photo.
* **Computer Vision:** YOLO locates the 9 individual grid cells, and Teachable Machine classifies your piece (`X`).
* **Smart AI Opponent:** Powered by the `xo` Python library to calculate optimal moves for the AI (`O`).
* **Interactive UI:** Built with Streamlit, includes beginner-friendly instructions and other built-in Streamlit features.

---

## 🛠️ Tech Stack
* **Python**
* **Streamlit** (Web App UI)
* **Ultralytics YOLO** (Grid Cell Detection)
* **Teachable Machine / tf-keras** (Symbol Classification)
* **`xo`** (Tic-Tac-Toe AI Logic)

---

## 📂 Project Structure
```text
TicTacBot/
├── app.py              # Main Streamlit web application
├── requirements.txt    # Required Python packages
├── best.pt             # Trained YOLO model weights
├── keras_model.h5      # Teachable Machine model file
└── labels.txt          # Model class labels
