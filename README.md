  **Don’t Swim in Data: Real-Time Microbial Forecasting for New Zealand Recreational Waters**
                                                                          

Traditional water quality monitoring, which relies on infrequent sampling and 48-hour lab- oratory delays, fails to capture rapid contamination fluctuations, leaving recreational water users exposed to health risks. To address this critical gap, we developed two novel machine learning frameworks for real-time forecasting of Enterococci concentrations in Canterbury, New Zealand.

The Probabilistic Forecasting Framework employs an ensemble of quantile regression models (covering the 5th to 98th percentiles), a gradient boosting meta-learner, and Conformalized Quantile Regression (CQR) to produce both accurate point forecasts and calibrated 90% prediction intervals. This approach captures the full range of contamination scenarios, enabling proactive, risk-based water quality management.

The Matrix Decomposition Framework uses Non-negative Matrix Factorization (NMF) to separate complex spatio-temporal water quality data into interpretable latent factors, which are then modeled with multi-target Random Forests. This method enhances inter- pretability and generalization, particularly for new monitoring sites with limited historical data.

Evaluated on a comprehensive dataset (2021–2024, 15 sites, 1047 samples, 100 exceedance events), the Probabilistic Framework achieved an overall exceedance sensitivity of 67.0% (rising to 75.7% in 2023–2024), a precautionary sensitivity of 77.0%, and a specificity of 92.3%, with a WMAPE of 17.2% during exceedance events. The Matrix Decomposition Framework delivered comparable performance, with an exceedance sensitivity of 61.0%, a precaution- ary sensitivity of 74.0%, a specificity of 90.6%, and a WMAPE of 20.3%. Together, these frameworks not only exceed USGS guidelines but also outperform traditional operational methods and standard ML benchmarks (e.g., linear regression, logistic regression, decision trees, and multi-layer perceptrons), while displaying highly competitive performance relative to state-of-the-art systems such as Auckland’s Safeswim.

SHAP analysis confirmed that short-term rainfall and wind conditions are the primary drivers of contamination, aligning with hydrological principles. A complete forecasting system—comprising a real-time data pipeline with automated validation and an interactive analytics dashboard—has been deployed in a staging environment, demonstrating both operational feasibility and the potential for broader applications in environmental risk management.

![image](https://github.com/user-attachments/assets/34ca8a44-e750-4812-8eee-008ffa6e5d38)

![image](https://github.com/user-attachments/assets/15ccf87c-44d2-477d-aa45-f6d348320aa7)

![beach](https://github.com/user-attachments/assets/683af48f-c1f4-418d-9eba-fc57e0815c32)
