# Supply Chain Analytics Dashboard

This project converts the original EDA and modeling notebooks into a modular Streamlit application.

## Project structure

```text
dashboard/
│
├── run.py              # Streamlit main page and navigation
├── data_processing.py  # Data loading and preprocessing
├── eda.py              # EDA and risk-analysis chart functions
├── modeling.py         # Machine-learning training and evaluation functions
├── requirements.txt    # Python dependencies
└── README.md           # Running instructions
```

## How to run

1. Put `DataCoSupplyChainDataset.csv` in the same folder as `run.py`.


2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run run.py
```

