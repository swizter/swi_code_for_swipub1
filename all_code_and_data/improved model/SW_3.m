% SWELL 内部验证：80%被试训练，20%被试测试
% 输出：X_train, Y_train, X_test, Y_test

clear; clc;
feature_cols = {'HR','HF_NU','LF_HF','RMSSD','TP','pNN25'};
rng(42);

swell = load_swell();
swell = normalize_subjects(swell, 'subject_id', feature_cols, 'numeric', 0);

subjects = unique(swell.subject_id);
N        = numel(subjects);
perm     = randperm(N);
n_train  = round(0.8 * N);
tr_subjs = subjects(perm(1:n_train));
te_subjs = subjects(perm(n_train+1:end));

tr_mask = ismember(swell.subject_id, tr_subjs);
te_mask = ismember(swell.subject_id, te_subjs);

X_train = table2array(swell(tr_mask, feature_cols));
Y_train = swell.condition(tr_mask);
X_test  = table2array(swell(te_mask, feature_cols));
Y_test  = swell.condition(te_mask);

[X_train,Y_train] = remove_nan(X_train,Y_train);
[X_test, Y_test]  = remove_nan(X_test, Y_test);

fprintf('SW→SW | 训练：%d窗口（%d被试），测试：%d窗口（%d被试）\n',...
    numel(Y_train),numel(tr_subjs),numel(Y_test),numel(te_subjs));
fprintf('特征顺序：%s\n', strjoin(feature_cols,', '));

% ---- 共享函数 ----
function data = load_swell()
    d = 'C:\Users\swizt\Desktop\research\PHD\PHD_project\HRV_stress_project\swell_dataset\regression\data';
    data = readtable(fullfile(d,'all_subject.csv'));
    data.condition(strcmp(data.condition,'no stress'))     = {0};
    data.condition(strcmp(data.condition,'interruption'))  = {1};
    data.condition(strcmp(data.condition,'time pressure')) = {2};
    data.condition = cell2mat(data.condition);
end

function data = load_wesad()
    d = 'C:\Users\swizt\Desktop\research\PHD\PHD_project\HRV_stress_project\wesad_dataset';
    data = readtable(fullfile(d,'all_subjects.csv'));
    data = data(strcmp(data.condition,'baseline')|strcmp(data.condition,'stress'),:);
    data.condition_num = zeros(height(data),1);
    data.condition_num(strcmp(data.condition,'stress')) = 1;
    data.condition = data.condition_num;
    data.subject_id = data.subjectId;
end

function data = normalize_subjects(data, subj_col, feat_cols, ~, bl_val)
    subjects     = unique(data.(subj_col));
    subj_numeric = isnumeric(data.(subj_col));
    cond_numeric = isnumeric(bl_val);
    for i = 1:numel(subjects)
        s = subjects(i);
        if subj_numeric, s_mask = data.(subj_col)==s;
        else,            s_mask = strcmp(data.(subj_col),s); end
        s_data = data(s_mask,:);
        if cond_numeric, bl_mask = s_data.condition==bl_val;
        else,            bl_mask = strcmp(s_data.condition,bl_val); end
        X_bl = table2array(s_data(bl_mask,feat_cols));
        mu = mean(X_bl,1,'omitnan');
        sd = std(X_bl,0,1); sd(sd<1e-8)=1;
        X_all = (table2array(s_data(:,feat_cols))-mu)./sd;
        for f=1:numel(feat_cols), data.(feat_cols{f})(s_mask)=X_all(:,f); end
    end
end

function [X,Y] = remove_nan(X,Y)
    v=~any(isnan(X),2); X=X(v,:); Y=Y(v);
end
