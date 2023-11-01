# Stock Market Prediction using LSTM

## Overview

This project uses a Long Short-Term Memory (LSTM) neural network to predict stock market trends, focusing on Apple Inc. (AAPL) as an example. The model is trained on historical stock price data and aims to predict future price trends.

## Model Performance

The model has been evaluated using the following metrics:

- Mean Absolute Error (MAE): 1.925
- Mean Squared Error (MSE): 5.371
- Root Mean Squared Error (RMSE): 2.318
- R-squared (R2) Score: 0.927

These metrics provide insights into the accuracy and performance of the model. A lower MAE, MSE, and RMSE indicates a better fit to the data, while a higher R2 score indicates a stronger correlation between predicted and actual prices.

## Project Files

- `stock_market_prediction.ipynb`: Jupyter Notebook containing the code for data preprocessing, model training, and evaluation.
- `keras_model.h5`: The trained LSTM model saved in HDF5 format.
- `README.md`: This document summarizing the project.

## Dependencies

To run the project, you'll need the following dependencies:

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- yfinance
- TensorFlow
- scikit-learn

## Usage

1. Clone this repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Open and run the `stock_market_prediction.ipynb` notebook to train and evaluate the LSTM model.
4. The model's performance metrics will be displayed in the notebook.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
