# WhisperChart

> **A next-generation, interactive, scenario-based forecasting tool for traders and learners.**

WhisperChart helps you build intuition for market moves using deep learning, real-time news sentiment, and interactive scenario exploration. It's designed to help beginner traders develop a "feel" for the market by showing likely paths, confidence ribbons, and the live impact of news.

## 🏗 Features
- Multi-step price prediction with LSTM or Transformers
- Fast, interactive lightweight charts for real-time data visualization
- Live market data from Alpaca API with customizable timeframes
- Real-time quotes with bid/ask spreads and volume
- Multi-timezone support with automatic market hours detection
- Auto-refresh functionality for live trading scenarios
- Interactive UI (Streamlit) with responsive design
- Notebook demos for rapid prototyping

## 🚀 Getting Started

1. **Clone this repo:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/whisperchart.git
    cd whisperchart
    ```

2. **Create a virtual environment (recommended):**
    ```bash
    python3.11 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4. **Set up Alpaca API credentials:**
    ```bash
    cd app
    cp .streamlit/secrets.toml.example .streamlit/secrets.toml
    # Edit .streamlit/secrets.toml with your Alpaca API keys
    ```
    Get free API keys from [Alpaca Markets](https://app.alpaca.markets/)

5. **Run the app:**
    ```bash
    streamlit run app.py
    ```

6. **Try the notebooks:**
    - Open `notebooks/whisperchart_dev.ipynb` in Colab, JupyterLab, or VSCode.

7. **Deploy to Hugging Face Spaces or Streamlit Cloud:**
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
