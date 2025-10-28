 Mixture-Based Machine Learning Analysis to Predict Fouling Release

This repository implements a machine learning pipeline designed to predict fouling release (FR) activity in antifouling coatings, leveraging insights from newly developed mixture descriptors. The approach integrates combinatorial feature generation, machine learning modeling, and model evaluation to establish predictive models for fouling release.

## 📄 Overview

The study focuses on developing quantitative structure–activity relationship (QSAR) models to correlate mixture descriptors with fouling release activity, specifically targeting algae removal at 110 kPa of the coating. The methodology encompasses:

- **Combinatorial Feature Generation:** Utilizing [`combinatorixPy`](https://github.com/your-repo/combinatorixPy) to create a comprehensive set of mixture descriptors.
- **Data Preprocessing:** Scaling, normalization, and handling missing values to prepare features for modeling.
- **Feature Selection:** Employing statistical methods and model-based importance to identify the most influential descriptors.
- **Model Training:** Implementing multiple machine learning algorithms (Decision Trees, Random Forest, Linear Models, SVR) to develop predictive models.
- **Model Evaluation:** Assessing model performance using metrics (R², MAE, RMSE) and visualization techniques such as:
  - Correlation plots (predicted vs. experimental)
  - Williams plots (leverage vs. standardized residuals)
  - Accumulated Local Effects (ALE) plots

## ⚙️ Installation

Clone this repository:

```bash
git clone https://github.com/Rahel-mahini/FOULING_RELEASE_MLOPS.git
cd FOULING_RELEASE_MLOPS

pip install -r requirements.txt

python main.py

```


##  Pipeline Overview

The pipeline executes the following steps:

1. **Generate combinatorial mixture descriptors** using `combinatorixPy`.
2. **Split the dataset** into training and test sets.
3. **Preprocess and normalize features** for modeling.
4. **Select top features**: first top 1000 descriptors, then refine to top 3.
5. **Train multiple machine learning models** including Decision Trees, Random Forests, Ridge, Lasso, and SVR.
6. **Evaluate models and generate plots**:
   - Correlation plot
   - Williams plot
   - ALE (Accumulated Local Effects) plot
7. **Save the trained model** for future predictions.

## 📊 Outputs

- **Selected Features**: CSV files for top 1000 and top 3 descriptors.
- **Model Results**: CSV with metrics (R², MAE, MSE, RMSE) for all trained models.
- **Plots**: Saved in `outputs/plots/` (correlation, Williams, ALE).
- **Trained Models**: Pickled models ready for serving.

## 📚 References

- **Preprint Paper**: [Mixture Descriptors for Fouling Release Prediction](https://www.preprints.org/frontend/manuscript/2560b1d015207b4a6b2fa40763041ae9/download_pub)
- **combinatorixPy**: Tool for combinatorial descriptor generation. (https://www.sciencedirect.com/science/article/pii/S2352711025000275)
- **Scikit-learn**: [https://scikit-learn.org](https://scikit-learn.org)

## 🔧 Contributing

Contributions are welcome! Please open an issue or pull request with improvements, bug fixes, or feature requests.

## 📄 License

This project is licensed under the GNU GENERAL PUBLIC LICENSE. See the LICENSE file for details.