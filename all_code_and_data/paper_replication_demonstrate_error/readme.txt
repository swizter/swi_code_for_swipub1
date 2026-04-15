this codes is a part of our comment paper: Comment on ‘Multi-Class Stress Detection Through Heart Rate Variability: A Deep Neural Network Based Study’ could replicate the data leakage problem of the target paper 

J. A. Mortensen et al., "Multi-Class Stress Detection Through Heart Rate Variability: A Deep Neural Network Based
Study," IEEE Access, vol. 11, pp. 57470–57480, 2023.

no split and split refer to two dataset split condition: subject independent and subject dependent split, in which the previous one will induce inflated perfect high accuracy in validation set, namely the data leakage problem. The latter one will show perfect performance in training and test set (as data leaked inside) but the validation performance will drop sharply to around 50%

the 8 files are for 1D CNN and the rest two marked as 'other models' provides data for MATLAB classificationtoolbox training, you just have to load the data and train all model, you could get the result as the figures shows, which represent the results from all other models (e.g., KNN Tree Linear regiression and linear SVM)

the corresponding four data files records the performance of each dataset, and the trained 1D CNN model itself