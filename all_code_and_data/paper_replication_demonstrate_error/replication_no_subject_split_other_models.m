% =========================================================
% Replication of erroneous paper method — Version 1: window-level random split (no subject isolation)
% No normalization preprocessing, direct 80/20 random window split
% Expected result: close to 100%, but entirely an overfitting artifact
% =========================================================

clear; clc;
feature_cols = {'HR','HF_NU','LF_HF','RMSSD','TP','pNN25'};
rng(42);

% ---- Load SWELL ------------------------------------------
swell = readtable('C:\Users\swizt\Desktop\research\PHD\PHD_project\HRV_stress_project\swell_dataset\regression\data\all_subject.csv');
swell.condition(strcmp(swell.condition,'no stress'))     = {0};
swell.condition(strcmp(swell.condition,'interruption'))  = {1};
swell.condition(strcmp(swell.condition,'time pressure')) = {1};
swell.condition = cell2mat(swell.condition);

% ---- Load WESAD ------------------------------------------
wesad = readtable('C:\Users\swizt\Desktop\research\PHD\PHD_project\HRV_stress_project\wesad_dataset\all_subjects.csv');
wesad = wesad(strcmp(wesad.condition,'baseline')|strcmp(wesad.condition,'stress'),:);
wesad.condition_num = double(strcmp(wesad.condition,'stress'));
wesad.condition     = wesad.condition_num;

% ---- Merge (no normalization) --------------------------------
keep_cols = [feature_cols, {'condition'}];
combined  = [swell(:,keep_cols); wesad(:,keep_cols)];
combined  = rmmissing(combined);

% ---- Window-level random 80/20 split (erroneous method) -----------------------
N      = height(combined);
perm   = randperm(N);
n_train = round(0.8 * N);

X_train = table2array(combined(perm(1:n_train),        feature_cols));
Y_train = combined.condition(perm(1:n_train));
X_test  = table2array(combined(perm(n_train+1:end),    feature_cols));
Y_test  = combined.condition(perm(n_train+1:end));

fprintf('[Replicating erroneous method] Window-level random split (no subject isolation, no normalization)\n');
fprintf('Total windows: %d\n', N);
fprintf('Train: %d windows   Test: %d windows\n', numel(Y_train), numel(Y_test));
fprintf('Train label distribution: no_stress=%d  stress=%d\n', sum(Y_train==0), sum(Y_train==1));
fprintf('Test label distribution:  no_stress=%d  stress=%d\n', sum(Y_test==0),  sum(Y_test==1));
fprintf('\nExpected: accuracy close to 100%% (data leakage — same subjects in train and test)\n');
fprintf('Feature order: %s\n', strjoin(feature_cols,', '));
