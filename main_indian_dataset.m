clear all
close all 
classified_image_big = imread('C:\Users\Fujitsu\Desktop\code_simul\Indian Pine\NS-line\19920612_AVIRIS_IndianPine_NS-line_gr.tif');
file_name_big = 'C:\Users\Fujitsu\Desktop\code_simul\Indian Pine\NS-line\19920612_AVIRIS_IndianPine_NS-line.prj';
real_image_big = imread('C:\Users\Fujitsu\Desktop\code_simul\aviris_hyperspectral_data\19920612_AVIRIS_IndianPine_NS-line.tif');
classified_image = imread('C:\Users\Fujitsu\Desktop\code_simul\Indian Pine\Site3_Project_and_Ground_Reference_Files\19920612_AVIRIS_IndianPine_Site3_gr.tif');
file_name = 'C:\Users\Fujitsu\Desktop\code_simul\Indian Pine\Site3_Project_and_Ground_Reference_Files\19920612_AVIRIS_IndianPine_Site3.prj';
real_image = imread('C:\Users\Fujitsu\Desktop\code_simul\aviris_hyperspectral_data\19920612_AVIRIS_IndianPine_Site3.tif');
class_labels = classified_image;
num_components = 16;
k = 1;
num_classes = 5;

MC = 10;
dm = 7;
overall_accuracy_saem_gmm = zeros(MC, dm);
overall_accuracy_saem = zeros(MC, dm);
overall_accuracy_mice = zeros(MC, dm);
overall_accuracy_mean_cond = zeros(MC, dm);
overall_accuracy_mean = zeros(MC, dm);
overall_accuracy_multiple_em = zeros(MC, dm);
overall_accuracy_mice_forest = zeros(MC, dm);

class_labels = double(class_labels);
real_image = double(real_image(30:115, 25:90, :));
classified_image = double(classified_image(30:115, 25:90));
real_image = real_image(:, :, 20:10:220);
min_value = min(real_image(:));
max_value = max(real_image(:));
classified_image = classified_image + 1;

for i = 1:86
    for j = 1:66
        if classified_image(i, j) == 3
            classified_image(i, j) = 2;
        elseif classified_image(i, j) == 7
            classified_image(i, j) = 3;
        elseif classified_image(i, j) == 11
            classified_image(i, j) = 4;
        elseif classified_image(i, j) == 12
            classified_image(i, j) = 5;
        end
    end
end

[numRows, numCols, numBands] = size(real_image);
data_reshaped = reshape(real_image, [numRows * numCols, numBands]);
spectralData_double = data_reshaped;

% Split data into training and test sets (75% training, 25% test)
for mc = 1:MC
    n = size(spectralData_double, 1);
    nb_train_samples = round(0.75 * n);
    random_indices = randperm(n);
    
    for i = 1:dm
        tic
        train_data = spectralData_double(random_indices(1:nb_train_samples), :);
        train_labels = classified_image(random_indices(1:nb_train_samples));
        test_data = spectralData_double(random_indices(nb_train_samples+1:end), :);
        test_labels = classified_image(random_indices(nb_train_samples+1:end));

        M1 = ones(size(train_data));
        M2 = ones(size(test_data));
        nanIndices1 = rand(size(train_data)) < 0.1 * i;  % Randomly choose 10% indices as NaN
        nanIndices2 = rand(size(test_data)) < 0.1 * i;  % Randomly choose 10% indices as NaN

        M1(nanIndices1) = NaN;
        M2(nanIndices2) = NaN;

        train_data_miss = train_data .* M1;
        test_data_miss = test_data .* M2;

        % Remove rows containing only NaNs
        nan_rows1 = all(isnan(train_data_miss), 2);
        train_data_miss = train_data_miss(~nan_rows1, :);
        train_labels = train_labels(~nan_rows1);
        nan_rows2 = all(isnan(test_data_miss), 2);
        test_data_miss = test_data_miss(~nan_rows2, :);
        test_labels = test_labels(~nan_rows2);

        % Train a multiclass logistic regression model (SAEM_GMM)
        nb_Sample_mcmc = 20;
        num_components_gmm = 3;
        num_components = 1;

        [beta_estimated_SAEM_gmm, w_SAEM_gmm, mu_SAEM_gmm, Sigma_SAEM_gmm] = M_SAEM(train_data_miss, train_labels, num_components_gmm);
        [y_predict_saem_gmm] = saem_gmm_predict(test_data_miss, mu_SAEM_gmm, Sigma_SAEM_gmm, w_SAEM_gmm, beta_estimated_SAEM_gmm, nb_Sample_mcmc, num_classes);
        
        % Calculate overall accuracy for SAEM-GMM
        mat_conf_saem_gmm = confusionmat(y_predict_saem_gmm, test_labels');
        correct_predictions_saem_gmm = sum(diag(mat_conf_saem_gmm));
        total_observations_saem_gmm = sum(mat_conf_saem_gmm(:));
        overall_accuracy_saem_gmm(mc, i) = correct_predictions_saem_gmm / total_observations_saem_gmm;

        % Train a multiclass logistic regression model (SAEM)
        [beta_estimated_SAEM, w_SAEM, mu_SAEM, Sigma_SAEM] = M_SAEM(train_data_miss, train_labels, num_components);
        [y_predict_saem] = saem_gmm_predict(test_data_miss, mu_SAEM, Sigma_SAEM, w_SAEM, beta_estimated_SAEM, nb_Sample_mcmc, num_classes);
        
        % Calculate overall accuracy for SAEM
        mat_conf_saem = confusionmat(y_predict_saem, test_labels');
        correct_predictions_saem = sum(diag(mat_conf_saem));
        total_observations_saem = sum(mat_conf_saem(:));
        overall_accuracy_saem(mc, i) = correct_predictions_saem / total_observations_saem;

        % Train a multiclass logistic regression model (MeanImputation)
        [logistic_model_mean] = MeanImputation_new(train_data_miss, train_labels);
        [X_mean_test] = MeanImputation_test(test_data_miss);
        % Prediction on test data using MeanImputation
        y_pred_probabilities_MEAN = mnrval(logistic_model_mean, X_mean_test);
        % Predictions are probabilities for each class
        [~, y_pred_logistic_MEAN] = max(y_pred_probabilities_MEAN, [], 2);
        
        % Calculate overall accuracy for MeanImputation
        mat_conf_mean = confusionmat(y_pred_logistic_MEAN, test_labels');
        correct_predictions_mean = sum(diag(mat_conf_mean));
        total_observations_mean = sum(mat_conf_mean(:));
        overall_accuracy_mean(mc, i) = correct_predictions_mean / total_observations_mean;

        % Train a multiclass logistic regression model (MICE_forest)
        [logistic_model_MICE_forest] = MICE_forest_new(train_data_miss, train_labels);
        [X_mice_test_forest] = MICE_test_forest(test_data_miss);
        % Prediction on test data using MICE_forest
        y_pred_probabilities_MICE_forest = mnrval(logistic_model_MICE_forest, X_mice_test_forest);
        % Predictions are probabilities for each class
        [~, y_pred_logistic_MICE_forest] = max(y_pred_probabilities_MICE_forest, [], 2);

        % Calculate overall accuracy for MICE_forest
        mat_conf_mice_forest = confusionmat(y_pred_logistic_MICE_forest, test_labels');
        correct_predictions_mice_f = sum(diag(mat_conf_mice_forest));
        total_observations_mice_f = sum(mat_conf_mice_forest(:));
        overall_accuracy_mice_forest(mc, i) = correct_predictions_mice_f / total_observations_mice_f;

        % Train and predict using SVM with MICE_forest imputation
        [svm_mice_forest] = MICE_forest_svm(train_data_miss, train_labels);
        [X_mice_test_forest] = MICE_test_forest(test_data_miss);
        y_pred_svm = predict(svm_mice_forest, X_mice_test_forest);
        
        % Calculate overall accuracy for SVM
        mat_conf_svm = confusionmat(y_pred_svm, test_labels');
        correct_predictions_svm = sum(diag(mat_conf_svm));
        total_observations_svm = sum(mat_conf_svm(:));
        overall_accuracy_svm_mice(mc, i) = correct_predictions_svm / total_observations_svm;

        % Train and predict using KNN with MICE_forest imputation
        [knn_mice_forest] = MICE_forest_knn(train_data_miss, train_labels);
        [X_mice_test_forest] = MICE_test_forest(test_data_miss);
        y_pred_knn = predict(knn_mice_forest, X_mice_test_forest);

        % Calculate overall accuracy for KNN
        mat_conf_knn = confusionmat(y_pred_knn, test_labels');
        correct_predictions_knn = sum(diag(mat_conf_knn));
        total_observations_knn = sum(mat_conf_knn(:));
        overall_accuracy_knn_mice(mc, i) = correct_predictions_knn / total_observations_knn;
% Training and prediction for RF with MICE-forest imputation
[mice_ff] = MICE_forest_forest(train_data_miss, train_labels);
[X_mice_test_forest] = MICE_test_forest(test_data_miss);
y_pred_ff = predict(mice_ff, X_mice_test_forest);
y_pred_ff = str2double(y_pred_ff);  % Convert predictions to double if necessary

% Calculate overall accuracy for MICE_forest (RF)
mat_conf_ff = confusionmat(y_pred_ff, test_labels');
correct_predictions_ff = sum(diag(mat_conf_ff));
total_observations_ff = sum(mat_conf_ff(:));
overall_accuracy_ff_mice(mc, i) = correct_predictions_ff / total_observations_ff;

disp(i)
toc
end
disp(mc)
end

save pred

        
    

