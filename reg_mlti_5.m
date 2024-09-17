
clear all
close all 
classified_image_big = imread('C:\Users\Fujitsu\Desktop\code_simul\Indian Pine\NS-line\19920612_AVIRIS_IndianPine_NS-line_gr.tif');
nom_du_fichier_big = 'C:\Users\Fujitsu\Desktop\code_simul\Indian Pine\NS-line\19920612_AVIRIS_IndianPine_NS-line.prj';
reel_image_big =imread( 'C:\Users\Fujitsu\Desktop\code_simul\aviris_hyperspectral_data\19920612_AVIRIS_IndianPine_NS-line.tif');
classified_image = imread('C:\Users\Fujitsu\Desktop\code_simul\Indian Pine\Site3_Project_and_Ground_Reference_Files\19920612_AVIRIS_IndianPine_Site3_gr.tif');
nom_du_fichier = 'C:\Users\Fujitsu\Desktop\code_simul\Indian Pine\Site3_Project_and_Ground_Reference_Files\19920612_AVIRIS_IndianPine_Site3.prj';
reel_image =imread( 'C:\Users\Fujitsu\Desktop\code_simul\aviris_hyperspectral_data\19920612_AVIRIS_IndianPine_Site3.tif');
classe=  classified_image;
num_components = 	16 ; 
k              =    1 ;
num_claas      = 5 ;

%correspondance_classes_big = [0,1; 1,1; 2,1; 3,2; 4,2; 5,2; 6,2; 7,2; 8,2; 9,3; 10,3; 11,3; 12,3; 13,3; 14,3; 15,4; 16,4; 17,4; 18,4; 19,4; 20,4; 21,4; 22,4; 23,5; 24,5; 25,5; 26,5; 27,5; 28,5;29,5; 30,5; 31,5; 32,5; 33,5; 34,5; 35,5; 36,5; 37,6; 38,6; 39,6; 40,6; 41,6; 42,6; 43,6; 44,6; 45,6; 46,6; 47,7; 48,7; 50,7; 51,7; 52,7; 53,7; 54,7; 55,7; 56,7];
   % Correspondance des classes (de 1 à 8)
%correspondance_classes = [0, 1; 1, 1; 2, 2; 3, 2; 4, 3; 5, 3; 6, 3; 7, 3; 8, 4; 9, 4; 10, 4; 11, 4; 12, 5; 13, 5; 14, 5; 15, 6; 16, 6];
%correspondance_classes = [0, 1; 1, 2; 2, 3; 3, 4; 4,5; 5, 6; 6, 7; 7, 8; 8, 9; 9, 10; 10, 11; 11, 12; 12, 13; 13, 14; 14, 15; 15, 16; 16, 17];


%correspondance_classes = [0, 1; 1, 1; 2, 1; 3, 2; 4, 2; 5, 2; 6, 3; 7, 3; 8, 3; 9, 3; 10, 4; 11, 4; 12, 4; 13, 4; 14, 5; 15, 5; 16, 6];


%correspondance_classes = correspondance_classes_big;
%classified_image       = classified_image_big;
%reel_image             = reel_image_big ;
MC = 10 ;
dm = 7; 
overall_accuracy_saem_gmm    = zeros(MC,dm);
overall_accuracy_saem        = zeros(MC,dm);
overall_accuracy_mice        = zeros(MC,dm);
overall_accuracy_mean_cond   = zeros(MC,dm);
overall_accuracy_mean        = zeros(MC,dm);
overall_accuracy_multiple_em = zeros(MC,dm);
overall_accuracy_mice_forest = zeros(MC,dm);


     
classe           = double(classe);
reel_image       = double(reel_image(30:115,25:90,:));
classified_image = double(classified_image(30:115,25:90));
reel_image       = reel_image(:,:,20:10:220);
min_value = min(reel_image(:));
max_value = max(reel_image(:));
classified_image = classified_image +1;

for i  = 1  :86
    for j = 1 : 66
        if classified_image(i ,j) == 3
            classified_image(i ,j) = 2 ;
        else if classified_image(i ,j) == 7
               classified_image(i ,j) = 3 ;
            else if classified_image(i ,j) == 11
                      classified_image(i ,j) = 4 ;
                else if classified_image(i ,j) == 12
                        classified_image(i ,j) = 5 ;
                    end
                end
            end
        end
    end
end

         
 [numRows, numCols, numBands] = size(reel_image);
data_reshaped = reshape(reel_image, [numRows * numCols, numBands]);
spectralData_double =  data_reshaped;
% Diviser les données en ensembles d'apprentissage et de test (75% d'apprentissage, 25% de test)
for mc = 1 : MC
    
n = size(spectralData_double, 1);
nb_echantillons_apprentissage = round(0.75 * n);
indices_aleatoires = randperm(n);
    for i = 1 : 7
       tic
donnees_apprentissage    = spectralData_double(indices_aleatoires(1:nb_echantillons_apprentissage), :);
etiquettes_apprentissage = classified_image(indices_aleatoires(1:nb_echantillons_apprentissage));
donnees_test             = spectralData_double(indices_aleatoires(nb_echantillons_apprentissage+1:end), :);
etiquettes_test          = classified_image(indices_aleatoires(nb_echantillons_apprentissage+1:end));


    M1 = ones(size(donnees_apprentissage));
    M2 = ones(size(donnees_test));
    nanIndices1 = rand(size(donnees_apprentissage)) < 0.1*i ;  % Randomly choose 10% indices as NaN
    nanIndices2 = rand(size(donnees_test)) < 0.1*i ;  % Randomly choose 10% indices as NaN

    M1(nanIndices1) = NaN;
    M2(nanIndices2) = NaN;
   
    donnees_apprentissage_miss  = donnees_apprentissage .* M1;
    donnees_test_miss           = donnees_test .* M2;
    
 % Supprimer les lignes contenant uniquement des NaN
    nan_rows1                       =   all(isnan(donnees_apprentissage_miss), 2);
    donnees_apprentissage_miss      =   donnees_apprentissage_miss(~nan_rows1, :);
    etiquettes_apprentissage        =   etiquettes_apprentissage(~nan_rows1);
    nan_rows2                       =   all(isnan(donnees_test_miss), 2);
    donnees_test_miss               =   donnees_test_miss(~nan_rows2, :);
    etiquettes_test                 =   etiquettes_test(~nan_rows2);
    

% Entraîner un modèle de régression logistique multiclasse(SAEM_GMM)
nb_Sample_mcmc = 20;
num_components_gmm = 3 ;
num_components     = 1 ;

[beta_estimated_SAEM_gmm, w_SAEM_gmm, mu_SAEM_gmm, Sigma_SAEM_gmm] =  M_SAEM(donnees_apprentissage_miss, etiquettes_apprentissage, num_components_gmm);
[y_predict_saem_gmm]                                       =  saem_gmm_predict(donnees_test_miss, mu_SAEM_gmm, Sigma_SAEM_gmm,w_SAEM_gmm, beta_estimated_SAEM_gmm, nb_Sample_mcmc,num_claas);
% calcule de OV pour saem-gmm
 mat_conf_saem_gmm = confusionmat(y_predict_saem_gmm,etiquettes_test');
 correct_predictions_saem_gmm = sum(diag(mat_conf_saem_gmm));
 total_observations_saem_gmm = sum(mat_conf_saem_gmm(:));
 overall_accuracy_saem_gmm(mc,i) = correct_predictions_saem_gmm / total_observations_saem_gmm;


% Entraîner un modèle de régression logistique multiclasse(SAEM)
[beta_estimated_SAEM, w_SAEM, mu_SAEM, Sigma_SAEM]       =  M_SAEM(donnees_apprentissage_miss, etiquettes_apprentissage, num_components);
[y_predict_saem]                                 =  saem_gmm_predict(donnees_test_miss, mu_SAEM, Sigma_SAEM,w_SAEM, beta_estimated_SAEM, nb_Sample_mcmc,num_claas);
% calcule de OV pour saem

 mat_conf_saem = confusionmat(y_predict_saem,etiquettes_test');
 correct_predictions_saem = sum(diag(mat_conf_saem));
 total_observations_saem = sum(mat_conf_saem(:));
 overall_accuracy_saem(mc,i) = correct_predictions_saem / total_observations_saem;


% Entraîner un modèle de régression logistique multiclasse(MeanImputation)

[logistic_model_mean]        = MeanImputation_new(donnees_apprentissage_miss,etiquettes_apprentissage);
[X_mean_test]                = MeanImputation_test(donnees_test_miss);
% Prédiction sur les données de test_SAEM
y_pred_probabilities_MEAN = mnrval(logistic_model_mean, X_mean_test);

% Les prédictions sont sous forme de probabilités pour chaque classe

[~, y_pred_logistic_MEAN] = max(y_pred_probabilities_MEAN, [], 2);
% calcule de OV pour mean-imputation 

 mat_conf_mean = confusionmat(y_pred_logistic_MEAN,etiquettes_test');
 correct_predictions_mean = sum(diag(mat_conf_mean));
 total_observations_mean = sum(mat_conf_mean(:));
 overall_accuracy_mean(mc,i) = correct_predictions_mean / total_observations_mean;


% Entraîner un modèle de régression logistique multiclasse(MICE_Imputation)

%[logistic_model_MICE]      = MICE_new(donnees_apprentissage_miss,etiquettes_apprentissage);
%[X_mice_test]              = MICE_test(donnees_test_miss);

% Prédiction sur les données de test_MICE
%y_pred_probabilities_MICE = mnrval(logistic_model_MICE, X_mice_test);

% Les prédictions sont sous forme de probabilités pour chaque classe

%[~, y_pred_logistic_MICE] = max(y_pred_probabilities_MICE, [], 2);

% calcule de OV pour mice-regression_lineaire 

 %mat_conf_mice = confusionmat(y_pred_logistic_MICE,etiquettes_test');
 %correct_predictions_mice = sum(diag(mat_conf_mice));
% total_observations_mice = sum(mat_conf_mice(:));
 %overall_accuracy_mice(mc,i) = correct_predictions_mice / total_observations_mice;



   % Entraîner un modèle de régression logistique multiclasse(MICE_fores)

[logistic_model_MICE_forest]      = MICE_forest_new(donnees_apprentissage_miss,etiquettes_apprentissage);
[X_mice_test_forest]              = MICE_test_forest(donnees_test_miss);

% Prédiction sur les données de test_MICE
y_pred_probabilities_MICE_forest = mnrval(logistic_model_MICE_forest, X_mice_test_forest);

% Les prédictions sont sous forme de probabilités pour chaque classe

[~, y_pred_logistic_MICE_forest] = max(y_pred_probabilities_MICE_forest, [], 2); 

% calcule de OV pour mice-regression_lineaire 

 mat_conf_mice_forest = confusionmat(y_pred_logistic_MICE_forest,etiquettes_test');
 correct_predictions_mice_f = sum(diag(mat_conf_mice_forest));
 total_observations_mice_f = sum(mat_conf_mice_forest(:));
 overall_accuracy_mice_forest(mc,i) = correct_predictions_mice_f / total_observations_mice_f;






% Entraîner un modèle de régression logistique multiclasse(mean_cond_Imputation)

[logistic_model_mean_cond]  = mean_cond_new(donnees_apprentissage_miss,etiquettes_apprentissage);
[x_test_mu_cond]            = mean_cond_test(donnees_test_miss);
% Prédiction sur les données de test_MEAN_COND
y_pred_probabilities_MEAN_COND = mnrval(logistic_model_mean_cond, x_test_mu_cond);

% Les prédictions sont sous forme de probabilités pour chaque classe

[~, y_pred_logistic_mean_cond] = max(y_pred_probabilities_MEAN_COND, [], 2);
% calcule de OV pour mean-cond

 mat_conf_mean_cond = confusionmat(y_pred_logistic_mean_cond,etiquettes_test');
 correct_predictions_mean_cond = sum(diag(mat_conf_mean_cond));
 total_observations_mean_cond = sum(mat_conf_mean_cond(:));
 overall_accuracy_mean_cond(mc,i) = correct_predictions_mean_cond / total_observations_mean_cond;



[logistic_model_Mltiple_EM] = Multiple_EM_new(donnees_apprentissage_miss,etiquettes_apprentissage);
[x_imputed_mlt_em_test]          = Multiple_EM_test(donnees_test_miss);
% Prédiction sur les données de test_multiple_em
y_pred_probabilities_Mltiple_EM = mnrval(logistic_model_Mltiple_EM, x_imputed_mlt_em_test);

% Les prédictions sont sous forme de probabilités pour chaque classe

[~, y_pred_logistic_Mltiple_EM ] = max(y_pred_probabilities_Mltiple_EM, [], 2);

% calcule de OV pour multiple-em

 mat_conf_mean_multiple_em = confusionmat(y_pred_logistic_Mltiple_EM,etiquettes_test');
 correct_predictions_multiple_em = sum(diag(mat_conf_mean_multiple_em));
 total_observations_multiple_em = sum(mat_conf_mean_multiple_em(:));
 overall_accuracy_multiple_em(mc,i) = correct_predictions_multiple_em / total_observations_multiple_em;
 
 
 toc
end
disp(mc)
end


save ind2


