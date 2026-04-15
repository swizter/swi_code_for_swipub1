% =========================================================
% SWELL Dataset Feature Direction Analysis - 6 Key HRV Metrics
% Purpose: Calculate changes of the following metrics from
%          no_stress → stress condition:
%          HR, HF_NU, LF/HF ratio, RMSSD, TP, pNN25
% =========================================================

clear; clc; close all;

% -------- Load Data ----------------------------------------
data_dir  = 'C:\Users\swizt\Desktop\research\PHD\PHD_project\HRV_stress_project\swell_dataset\regression\data';
data      = readtable(fullfile(data_dir, 'all_subject.csv'));

% -------- Define Key Features ------------------------------------
selected_features = {'HR', 'HF_NU', 'HF_LF', 'RMSSD', 'TP', 'pNN25'};
feature_display   = {'Heart Rate (HR)', 'Normalized HF Power (HF-NU)', ...
                     'HF/LF Ratio', 'RMSSD', 'Total Power (TP)', 'pNN25'};

% Check if features exist in dataset
all_cols = data.Properties.VariableNames;
feature_cols = {};
valid_features = {};
for i = 1:length(selected_features)
    if ismember(selected_features{i}, all_cols)
        feature_cols{end+1} = selected_features{i};
        valid_features{end+1} = feature_display{i};
    else
        fprintf('Warning: Feature %s not found\n', selected_features{i});
    end
end

n_features = length(feature_cols);
fprintf('Successfully loaded %d features\n', n_features);

subjects   = unique(data.subject_id);
N_subjects = numel(subjects);
fprintf('Total number of subjects: %d\n\n', N_subjects);

% -------- Baseline Normalization and Calculate Mean Feature Values ------
% Normalize each subject's features using no-stress baseline (z-score)
cond_labels = {'no stress', 'stress'};

mean_no_stress = nan(N_subjects, n_features);
mean_stress    = nan(N_subjects, n_features);

for s = 1:N_subjects
    subj_mask    = data.subject_id == subjects(s);
    subj_data    = data(subj_mask, :);

    no_stress_mask = strcmp(subj_data.condition, 'no stress');
    stress_mask    = strcmp(subj_data.condition, 'interruption') | ...
                     strcmp(subj_data.condition, 'time pressure');

    for f = 1:n_features
        col = feature_cols{f};

        % Calculate baseline (no-stress) mean and std
        baseline_values = subj_data.(col)(no_stress_mask);
        baseline_mean   = mean(baseline_values, 'omitnan');
        baseline_std    = std(baseline_values, 'omitnan');

        % z-score normalization using baseline
        if baseline_std > 0
            subj_data_normalized = (subj_data.(col) - baseline_mean) / baseline_std;
        else
            subj_data_normalized = subj_data.(col) - baseline_mean;  % If std=0, just center
        end

        % Calculate mean z-scores for each condition
        mean_no_stress(s, f) = mean(subj_data_normalized(no_stress_mask), 'omitnan');
        mean_stress(s, f)    = mean(subj_data_normalized(stress_mask),    'omitnan');
    end
end

fprintf('\n========== Baseline-Normalized (Z-Score) Analysis ==========\n');

% -------- Calculate Change Magnitude and Direction Consistency -----
delta      = mean_stress - mean_no_stress;  % Positive = increased under stress
pct_change = delta ./ abs(mean_no_stress) * 100;

% Proportion of subjects with consistent direction
pct_positive = sum(delta > 0, 1) / N_subjects * 100;  % Increased under stress
pct_negative = sum(delta < 0, 1) / N_subjects * 100;  % Decreased under stress
consistency  = max(pct_positive, pct_negative);        % Consistency index

fprintf('====== Direction Consistency Summary - 6 Key Features ======\n');
fprintf('%-30s  %10s  %10s  %12s  %12s\n', 'Feature', 'Increase %', 'Decrease %', 'Consistency %', 'Mean Change %');
fprintf('%s\n', repmat('-', 1, 85));
for f = 1:n_features
    fprintf('%-30s  %10.1f  %10.1f  %12.1f  %12.1f\n', ...
        valid_features{f}, pct_positive(f), pct_negative(f), ...
        consistency(f), mean(pct_change(:, f), 'omitnan'));
end

% Recommended features (consistency >= 70%)
recommended_idx = consistency >= 70;
fprintf('\nFeatures with consistency >= 70%%:');
if any(recommended_idx)
    fprintf(' %s', valid_features{recommended_idx});
else
    fprintf(' None');
end
fprintf('\n\n');

%% target figure
% ========== SECTION: Generate 6 Independent Feature Figures ==========
% Run this section to generate 6 independent figures for feature plotting
% Each figure can be saved and arranged in Visio

% Define units for each feature based on typical value ranges
feature_units = {'bpm', '(norm)', '(ratio)', 'ms', 'ms^2', '%'};

for f = 1:n_features
    % Create individual figure for each feature
    figure('Name', sprintf('Feature %d: %s', f, valid_features{f}), ...
           'Position', [100 + (f-1)*50, 100 + (f-1)*50, 600, 500]);
    hold on;

    % Plot individual subject trajectories
    for s = 1:N_subjects
        if delta(s, f) > 0
            col = [0.8 0.2 0.2];  % Red: increased under stress
        else
            col = [0.2 0.5 0.8];  % Blue: decreased under stress
        end
        plot([1 2], [mean_no_stress(s, f), mean_stress(s, f)], ...
            '-o', 'Color', [col 0.5], 'LineWidth', 2.5, 'MarkerSize', 5);
    end

    % Plot mean trajectory (bold line)
    mean_baseline = mean(mean_no_stress(:, f), 'omitnan');
    mean_stress_val = mean(mean_stress(:, f), 'omitnan');
    plot([1 2], [mean_baseline, mean_stress_val], ...
        'k-o', 'LineWidth', 4.5, 'MarkerSize', 9);

    xlim([0.7 2.3]);
    ylim auto;

    % Keep only left Y-axis, display 3 ticks (top, middle, bottom)
    axis on;
    xticks([]);  % Remove X-axis tick labels
    xticklabels([]);
    set(gca, 'box', 'off');  % Remove border
    grid off;

    % Set Y-axis to display only 3 ticks (top, middle, bottom)
    y_min = min(ylim);
    y_max = max(ylim);
    y_mid = (y_min + y_max) / 2;
    yticks([y_min, y_mid, y_max]);
    set(gca, 'FontSize', 9);

    % Add feature name and consistency info below
    y_pos = y_min - 0.15 * (y_max - y_min);
    text(1.5, y_pos, valid_features{f}, ...
        'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold');

    % Add title at the bottom of figure (clearer feature name)
    subtitle(sprintf('%s (%.0f%% consistency)', valid_features{f}, consistency(f)), ...
        'FontSize', 14, 'FontWeight', 'bold');
end

fprintf('Generated %d independent figures for paper arrangement.\n', n_features);

%% other figure
% -------- Figure 2: Boxplot of Baseline-Normalized Changes with Consistency --------
figure('Name', 'Baseline-Normalized Feature Changes', 'Position', [100 100 900 500]);

% Define feature short names for x-axis
feature_shortnames = {'HR', 'HF-NU', 'HF/LF', 'RMSSD', 'TP', 'pNN25'};

% Calculate normalized change (stress - baseline for each subject)
delta_normalized = mean_stress - mean_no_stress;

% Create boxplot with no outlier markers
bp = boxplot(delta_normalized, 'Labels', feature_shortnames, 'Whisker', 1.5, 'Symbol', '');
hold on;

% Increase linewidth of boxplot elements
set(bp, 'LineWidth', 2.5);

% Add reference line at y=0 (baseline)
yline(0, 'k--', 'LineWidth', 2);

% Add consistency percentage text at y=7 position for all features
for f = 1:n_features
    % Display consistency percentage at fixed y position
    text(f, 7, sprintf('%.0f%%', consistency(f)), ...
        'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', ...
        'Color', [0.2 0.2 0.2]);
end

ylabel('Difference of zscored index', 'FontSize', 12);
set(gca, 'FontSize', 11);
grid off;
box off;

% Set x-axis tick labels to show feature names
set(gca, 'XTick', 1:n_features);
set(gca, 'XTickLabel', feature_shortnames);
set(gca, 'TickDir', 'out');  % Ticks point outward, reducing visual clutter

% Hide only the x-axis line, keep the labels and ticks
ax = gca;
% ax.XAxis.LineWidth = 0;  % Make x-axis line invisible (width 0)

% ylim auto;
ylim([-7,8]);
% Tighten x-axis spacing for very compact layout (boxplots closer together)
xlim([0.2, n_features + 0.8]);

fprintf('========================================\n');
fprintf('Analysis complete! Generated 3 figures.\n');
fprintf('========================================\n');
