% =========================================================
% Replication of erroneous paper method — Version 2: subject-level split (no normalization)
% Same lack of normalization preprocessing, but data split 80/20 by subject
% Expected result: ACC drops sharply to ~50%, revealing true generalization ability
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
wesad.subject_id    = wesad.subjectId;

% ---- Re-encode subject IDs to avoid ID conflicts between datasets ------------------
sw_subjs = unique(swell.subject_id);
we_subjs = unique(wesad.subject_id);
offset   = numel(sw_subjs);
sw_map   = containers.Map(num2cell(sw_subjs), num2cell(1:numel(sw_subjs)));
we_map   = containers.Map(num2cell(we_subjs), num2cell((1:numel(we_subjs))+offset));
swell.subject_id = arrayfun(@(x) sw_map(x), swell.subject_id);
wesad.subject_id = arrayfun(@(x) we_map(x), wesad.subject_id);

% ---- Merge (no normalization) --------------------------------
keep_cols = [feature_cols, {'subject_id','condition'}];
combined  = [swell(:,keep_cols); wesad(:,keep_cols)];
combined  = rmmissing(combined);

% ---- Subject-level 80/20 split (correct split method, but still no normalization) ---------
all_subjs = unique(combined.subject_id);
N         = numel(all_subjs);
perm      = randperm(N);
n_train   = round(0.8 * N);
tr_subjs  = all_subjs(perm(1:n_train));
te_subjs  = all_subjs(perm(n_train+1:end));

tr_mask = ismember(combined.subject_id, tr_subjs);
te_mask = ismember(combined.subject_id, te_subjs);

X_train = table2array(combined(tr_mask, feature_cols));
Y_train = combined.condition(tr_mask);
X_test  = table2array(combined(te_mask, feature_cols));
Y_test  = combined.condition(te_mask);

fprintf('[Control experiment] Subject-level split (no normalization)\n');
fprintf('Total subjects: %d (SWELL %d + WESAD %d)\n', N, numel(sw_subjs), numel(we_subjs));
fprintf('Train: %d windows (%d subjects)   Test: %d windows (%d subjects)\n',...
    numel(Y_train), numel(tr_subjs), numel(Y_test), numel(te_subjs));
fprintf('Train label distribution: no_stress=%d  stress=%d\n', sum(Y_train==0), sum(Y_train==1));
fprintf('Test label distribution:  no_stress=%d  stress=%d\n', sum(Y_test==0),  sum(Y_test==1));
fprintf('\nExpected: ACC drops to ~50%% (no normalization — individual differences mask stress signal)\n');
fprintf('Feature order: %s\n', strjoin(feature_cols,', '));
