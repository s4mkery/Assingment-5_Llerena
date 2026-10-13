# Assingment-5_Llerena

Despite extensive hyperparameter tuning of tree-based architectures (max depth = 4, min samples leaf = 15), Linear Regression consistently outperformed both Decision Trees and Random Forests, achieving the highest explanatory power R^2 = [0.45, 0.5] and the lowest prediction error MEA(49)

For a step function to closely approximate a smooth gradient, an extensive sample size and deep hierarchical branching are required. With only 353 training observations and a regularization constraint of min_samples_leaf = 15, the decision tree is restricted to approximately 8 to 12 terminal leaves. Consequently, the model can only output 8 to 12 discrete values across a wide continuous target range (25 to 346), assigning identical predictions to clinically heterogeneous individuals mapped to the same leaf. Relaxing this constraint allows the model to overfit sample-specific noise, leading to catastrophic test-set degradation (R^2<0).
