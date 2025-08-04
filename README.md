# WhisperChart

> **A next-generation, interactive, scenario-based forecasting tool for traders and learners.**

WhisperChart helps you build intuition for market moves using deep learning, real-time news sentiment, and interactive scenario exploration. It's designed to help beginner traders develop a "feel" for the market by showing likely paths, confidence ribbons, and the live impact of news.

## 🏗 Features
- Multi-step price prediction with LSTM or Transformers
- Live chart with uncertainty ribbon and sentiment overlay
- Real-time news sentiment (FinBERT integration)
- Interactive UI (Streamlit or Gradio)
- Notebook demos for rapid prototyping

## 🚀 Getting Started

1. **Clone this repo:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/whisperchart.git
    cd whisperchart
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the app (Streamlit example):**
    ```bash
    cd app
    streamlit run app.py
    ```

4. **Try the notebooks:**
    - Open `notebooks/whisperchart_dev.ipynb` in Colab, JupyterLab, or VSCode.

5. **Deploy to Hugging Face Spaces or Streamlit Cloud:**
    - Connect your GitHub repo for instant web demos.

## 🗂️ Structure

```
app/        # Main app UI code
notebooks/  # Prototyping and dev experiments
models/     # Trained models and scalers
data/       # Optional: Sample CSVs or demo data
```

---

**Roadmap:**
- [ ] Add “What-if” sentiment slider to UI
- [ ] Switch to Transformer for even less lag
- [ ] Collect real user feedback
- [ ] Integrate with broker API for live trading (optional)

**License:** MIT

---

*Demo screenshots and more docs coming soon!*
