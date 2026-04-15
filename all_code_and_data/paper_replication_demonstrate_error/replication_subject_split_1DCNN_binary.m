% =========================================================
% Correct evaluation — 1D CNN (subject-level split, binary classification)
% Data: SWELL-KW only, 34 features
% Labels: no stress=0, interruption+time pressure=1 (stressed)
% Split: 80% subjects for training, 20% subjects for testing, no subject overlap
% Expected result: significantly lower than no-split version, reflecting true cross-subject generalization
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

% Extract features + labels + subject_id, remove missing values
data = swell(:, [feature_cols, {'subject_id', 'label'}]);
data = rmmissing(data);

fprintf('Total samples: %d\n', height(data));
fprintf('Class distribution — no_stress: %d  stress: %d\n', sum(data.label==0), sum(data.label==1));

% ---- Subject-level 80/20 split --------------------------------------
all_subjects = unique(data.subject_id);
n_subjects   = numel(all_subjects);
fprintf('\nTotal subjects: %d\n', n_subjects);

perm_subj      = randperm(n_subjects);
n_train_subj   = round(0.8 * n_subjects);
train_subjects = all_subjects(perm_subj(1:n_train_subj));
test_subjects  = all_subjects(perm_subj(n_train_subj+1:end));

fprintf('Training subjects (%d): %s\n', numel(train_subjects), num2str(train_subjects'));
fprintf('Test subjects (%d): %s\n', numel(test_subjects),  num2str(test_subjects'));

train_mask = ismember(data.subject_id, train_subjects);
test_mask  = ismember(data.subject_id, test_subjects);

X_train = table2array(data(train_mask, feature_cols));
Y_train = data.label(train_mask);
X_test  = table2array(data(test_mask,  feature_cols));
Y_test  = data.label(test_mask);

fprintf('\nTraining set: %d samples (%d subjects)\n', numel(Y_train), numel(train_subjects));
fprintf('Test set: %d samples (%d subjects, zero overlap)\n', numel(Y_test), numel(test_subjects));

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
max_epochs = 80;

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
fprintf('\n=== Start training 1D CNN (subject-level split, binary classification) ===\n\n');
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
fprintf('  1D CNN Binary Classification  —  Subject-Level Split (Correct Evaluation)\n');
fprintf('  Training subjects: %d   Test subjects: %d\n', numel(train_subjects), numel(test_subjects));
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
fprintf('[!] This result reflects the model''s true generalization to unseen subjects.\n');
fprintf('    The gap vs. the no-split version quantifies the extent of data leakage.\n');
fprintf('========================================================\n');
