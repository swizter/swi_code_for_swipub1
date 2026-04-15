% Combined Dataset + Absolute Deviation Features
% Two datasets normalized separately, take absolute values, merge and split by subjects 80/20
% Core assumption: stress = deviation from baseline (regardless of direction), |X_norm| eliminates directional differences
% Output: X_train, Y_train, X_test, Y_test

clear; clc;
feature_cols = {'HR','HF_NU','LF_HF','RMSSD','TP','pNN25'};
rng(42);

% ---- Load and Normalize SWELL ----------------------------------
swell = readtable('C:\Users\swizt\Desktop\research\PHD\PHD_project\HRV_stress_project\swell_dataset\regression\data\all_subject.csv');
swell.condition(strcmp(swell.condition,'no stress'))     = {0};
swell.condition(strcmp(swell.condition,'interruption'))  = {1};
swell.condition(strcmp(swell.condition,'time pressure')) = {1};
swell.condition = cell2mat(swell.condition);
swell = normalize_subjects(swell, 'subject_id', feature_cols, 0);

% ---- Load and Normalize WESAD ----------------------------------
wesad = readtable('C:\Users\swizt\Desktop\research\PHD\PHD_project\HRV_stress_project\wesad_dataset\all_subjects.csv');
wesad = wesad(strcmp(wesad.condition,'baseline')|strcmp(wesad.condition,'stress'),:);
wesad.condition_num = double(strcmp(wesad.condition,'stress'));
wesad.condition     = wesad.condition_num;
wesad.subject_id    = wesad.subjectId;
wesad = normalize_subjects(wesad, 'subject_id', feature_cols, 0);

% ---- Take Absolute Values --------------------------------------------
for f = 1:numel(feature_cols)
    swell.(feature_cols{f}) = abs(swell.(feature_cols{f}));
    wesad.(feature_cols{f}) = abs(wesad.(feature_cols{f}));
end

% ---- Remap Subject IDs to Avoid ID Conflicts Between Datasets ----------------
sw_subjs = unique(swell.subject_id);
we_subjs = unique(wesad.subject_id);
offset   = numel(sw_subjs);

sw_map = containers.Map(num2cell(sw_subjs), num2cell(1:numel(sw_subjs)));
we_map = containers.Map(num2cell(we_subjs), num2cell((1:numel(we_subjs)) + offset));

swell.subject_id = arrayfun(@(x) sw_map(x), swell.subject_id);
wesad.subject_id = arrayfun(@(x) we_map(x), wesad.subject_id);

% ---- Merge ------------------------------------------------
keep_cols = [feature_cols, {'subject_id','condition'}];
combined  = [swell(:, keep_cols); wesad(:, keep_cols)];
combined  = rmmissing(combined);

% ---- Split by Subjects 80/20 -----------------------------------
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

fprintf('Combined Dataset (Absolute Deviation Features)\n');
fprintf('Total Subjects: %d (SWELL %d + WESAD %d)\n', N, numel(sw_subjs), numel(we_subjs));
fprintf('Training: %d windows (%d subjects)  Testing: %d windows (%d subjects)\n',...
    numel(Y_train), numel(tr_subjs), numel(Y_test), numel(te_subjs));
fprintf('Training set label distribution: no_stress=%d  stress=%d\n', sum(Y_train==0), sum(Y_train==1));
fprintf('Test set label distribution: no_stress=%d  stress=%d\n', sum(Y_test==0),  sum(Y_test==1));
fprintf('Feature order: %s\n', strjoin(feature_cols,', '));

% ---- Helper Functions --------------------------------------------
function data = normalize_subjects(data, subj_col, feat_cols, bl_val)
    subjects = unique(data.(subj_col));
    for i = 1:numel(subjects)
        s_mask  = data.(subj_col) == subjects(i);
        s_data  = data(s_mask, :);
        bl_mask = s_data.condition == bl_val;
        X_bl    = table2array(s_data(bl_mask, feat_cols));
        mu = mean(X_bl, 1, 'omitnan');
        sd = std(X_bl,  0, 1);  sd(sd < 1e-8) = 1;
        X_all = (table2array(s_data(:, feat_cols)) - mu) ./ sd;
        for f = 1:numel(feat_cols)
            data.(feat_cols{f})(s_mask) = X_all(:, f);
        end
    end
end
