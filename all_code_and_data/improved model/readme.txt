this codes is a part of our comment paper: Comment on ‘Multi-Class Stress Detection Through Heart Rate Variability: A Deep Neural Network Based Study’ could replicate the data leakage problem of the target paper 

J. A. Mortensen et al., "Multi-Class Stress Detection Through Heart Rate Variability: A Deep Neural Network Based
Study," IEEE Access, vol. 11, pp. 57470–57480, 2023.

the code feature_direction_swellonly demonstrate the directional change of each feature in conditions between no stress and stressed set, which serve as the criteria of feature nomalization and selection in our model (as we selected 6 features from 34)

the rest three codes are used to reproduce our model performance after our feature engineering, run the code, get X_train X_test Y_train Y_test, open the matlab classificationLearner and click 'train all models' you could get the same results as reported. You could also get a glance of our results from our picture.

