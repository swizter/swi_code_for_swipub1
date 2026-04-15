% =========================================================
% Replication of original paper — 1D CNN (no subject split, binary classification)
% Data: SWELL-KW only, 34 features
% Labels: no stress=0, interruption+time pressure=1 (stressed)
% Split: 80/20 window-level random split (erroneous method from original paper)
% Expected result: close to 100%, due to data leakage
% =========================================================

clear; clc;
rng(42);

% ---- 34 features --------------------------------------------
feature_cols = {'MEAN_RR','MEDIAN_RR','SDRR','RMSSD','SDSD','SDRR_RMSSD', ...
    'HR','pNN25','pNN50','SD1','SD2','KURT','SKEW', ...
    'MEAN_REL_RR','MEDIAN_REL_RR','SDRR_REL_RR','RMSSD_REL_RR', ...
    'SDSD_REL_RR','SDRR_RMSSD_REL_RR','KURT_REL_RR','SKEW_REL_RR', ...
    'VLF','VLF_PCT','LF','LF_PCT','LF_NU','HF','HF_PCT','HF_NU', ...
    'TP','LF_HF','HF_LF','sampen','higuci'};

n_feat = numel(feature_cols);  % 34

% ---- Load SWELL-KW ----------------------------------------
fprintf('Loading SWELL-KW dataset...\n');
swell = readtable('C:\Users\swizt\Desktop\research\PHD\PHD_project\HRV_stress_project\swell_dataset\regression\data\all_subject.csv');

% ---- Binary labels: no stress=0, others=1 ----------------------
swell.label = zeros(height(swell), 1);
swell.label(strcmp(swell.condition, 'interruption'))  = 1;
swell.label(strcmp(swell.condition, 'time pressure')) = 1;

% Extract features + labels, remove missing values
data = swell(:, [feature_cols, {'label'}]);
data = rmmissing(data);

X = table2array(data(:, feature_cols));
Y = data.label;

fprintf('Total samples: %d\n', numel(Y));
fprintf('Class distribution — no_stress: %d  stress: %d\n', sum(Y==0), sum(Y==1));

% ---- Window-level random 80/20 split (no subject isolation) ---------------------
N       = numel(Y);
perm    = randperm(N);
n_train = round(0.8 * N);
idx_train = perm(1:n_train);
idx_test  = perm(n_train+1:end);

X_train = X(idx_train, :);
Y_train = Y(idx_train);
X_test  = X(idx_test,  :);
Y_test  = Y(idx_test);

fprintf('\nTraining set: %d samples   Test set: %d samples\n', numel(Y_train), numel(Y_test));

% ---- Normalize to [0,1] ----------------------------------------
X_min   = min(X_train, [], 1);
X_max   = max(X_train, [], 1);
X_range = X_max - X_min;
X_range(X_range == 0) = 1;

X_train_norm = (X_train - X_min) ./ X_range;
X_test_norm  = (X_test  - X_min) ./ X_range;

% ---- Split out 5% validation set -----------------------------------------
N_tr    = size(X_train_norm, 1);
n_val   = round(0.05 * N_tr);
idx_val = N_tr - n_val + 1 : N_tr;
idx_tr  = 1 : N_tr - n_val;

X_val_norm   = X_train_norm(idx_val, :);
Y_val        = Y_train(idx_val);
X_train_norm = X_train_norm(idx_tr, :);
Y_train      = Y_train(idx_tr);

N_train = size(X_train_norm, 1);
N_val   = size(X_val_norm,   1);
N_test  = size(X_test_norm,  1);

% ---- Convert to cell array (34 channels, 1 time step) -------------------
X_train_cell = mat2cell(X_train_norm', n_feat, ones(1, N_train))';
X_val_cell   = mat2cell(X_val_norm',   n_feat, ones(1, N_val))';
X_test_cell  = mat2cell(X_test_norm',  n_feat, ones(1, N_test))';

% Convert labels to categorical
class_names = {'no_stress', 'stress'};
Y_train_cat = categorical(Y_train, [0,1], class_names);
Y_val_cat   = categorical(Y_val,   [0,1], class_names);
Y_test_cat  = categorical(Y_test,  [0,1], class_names);

% ---- 1D CNN architecture (consistent with original paper, output changed to 2 classes) ---------------
numClasses = 2;

layers = [
    sequenceInputLayer(n_feat, 'MinLength', 1, 'Name', 'input')
    convolution1dLayer(2, 64, 'Padding', 'same', 'Name', 'conv1d')
    reluLayer('Name', 'relu')
    fullyConnectedLayer(16, 'Name', 'dense_1')
    globalMaxPooling1dLayer('Name', 'global_pool')
    fullyConnectedLayer(numClasses, 'Name', 'dense_out')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% ---- Training options ---------------------------------------------
max_epochs = 40;

fprintf('\nNumber of epochs: %d\n', max_epochs);
fprintf('Estimated iterations per epoch: %d\n', ceil(N_train / 64));

options = trainingOptions('adam', ...
    'MaxEpochs',           max_epochs, ...
    'MiniBatchSize',       64, ...
    'Shuffle',             'every-epoch', ...
    'ValidationData',      {X_val_cell, Y_val_cat}, ...
    'ValidationFrequency', max(1, ceil(N_train/64)), ...
    'Verbose',             true, ...
    'Plots',               'training-progress');

% ---- Train ------------------------------------------------
fprintf('\n=== Start training 1D CNN (no subject split, binary classification) ===\n\n');
net = trainNetwork(X_train_cell, Y_train_cat, layers, options);

% ---- Test set evaluation ------------------------------------------
fprintf('\n=== Test Set Evaluation ===\n');
Y_pred = classify(net, X_test_cell);

acc = mean(Y_pred == Y_test_cat);
C   = confusionmat(Y_test_cat, Y_pred, 'Order', categorical(class_names));

prec_vals = zeros(numClasses, 1);
rec_vals  = zeros(numClasses, 1);
f1_vals   = zeros(numClasses, 1);
mcc_vals  = zeros(numClasses, 1);

for c = 1:numClasses
    TP = C(c, c);
    FP = sum(C(:, c)) - TP;
    FN = sum(C(c, :)) - TP;
    TN = sum(C(:))    - TP - FP - FN;

    prec_vals(c) = TP / max(TP + FP, 1);
    rec_vals(c)  = TP / max(TP + FN, 1);
    f1_vals(c)   = 2 * prec_vals(c) * rec_vals(c) / max(prec_vals(c) + rec_vals(c), eps);

    denom = sqrt(double((TP+FP)) * double((TP+FN)) * double((TN+FP)) * double((TN+FN)));
    if denom == 0, mcc_vals(c) = 0;
    else,          mcc_vals(c) = double(TP*TN - FP*FN) / denom;
    end
end

fprintf('\n');
fprintf('========================================================\n');
fprintf('  1D CNN Binary Classification  —  No Subject Split (Erroneous Method)\n');
fprintf('========================================================\n');
fprintf('Accuracy  : %.4f  (%.2f%%)\n', acc, acc*100);
fprintf('Precision : %.4f  (macro-avg)\n', mean(prec_vals));
fprintf('Recall    : %.4f  (macro-avg)\n', mean(rec_vals));
fprintf('F1-score  : %.4f  (macro-avg)\n', mean(f1_vals));
fprintf('MCC       : %.4f  (macro-avg)\n', mean(mcc_vals));
fprintf('\n--- Per-class details ---\n');
fprintf('%-12s  Precision  Recall  F1-score\n', 'Class');
for c = 1:numClasses
    fprintf('%-12s  %.4f     %.4f   %.4f\n', class_names{c}, prec_vals(c), rec_vals(c), f1_vals(c));
end
fprintf('\n--- Confusion Matrix (rows=true, cols=predicted) ---\n');
disp(array2table(C, 'RowNames', class_names, 'VariableNames', class_names));
fprintf('========================================================\n');
fprintf('[!] High accuracy is caused by data leakage, not true generalization.\n');
fprintf('========================================================\n');
