%% Generate dataset
MC = 100 ;
dm = 8 ; 
num_claas  = 5 ;
dim = 5 ;
Ratio_Miss = 1 ;
nb_Sample_mcmc = 20 ;
num_components_gmm = 3 ;
num_component_saem     = 1 ;
overall_accuracy_saem_gmm    = zeros(MC,dm);
overall_accuracy_saem        = zeros(MC,dm);
overall_accuracy_mice        = zeros(MC,dm);
overall_accuracy_mean_cond   = zeros(MC,dm);
overall_accuracy_mean        = zeros(MC,dm);
overall_accuracy_multiple_em = zeros(MC,dm);
overall_accuracy_mice_forest = zeros(MC,dm);
overall_accuracy_noNA        = zeros(MC,dm);
beta_estimated_SAEM_gmm_mc          = zeros(dim+1,num_claas-1,MC);
beta_estimated_SAEM_mc              = zeros(dim+1,num_claas-1,MC);
beta_estimated_mice_mc              = zeros(dim+1,num_claas-1,MC);
beta_estimated_mean_cond_mc         = zeros(dim+1,num_claas-1,MC);
beta_estimated_mean_mc              = zeros(dim+1,num_claas-1,MC);
beta_estimated_multiple_em_mc       = zeros(dim+1,num_claas-1,MC);
beta_estimated_mice_forest_mc       = zeros(dim+1,num_claas-1,MC);
p = dim;
% Define GMM parameters
K = 3;
mu1 = [1, 5, 8, 4, 2];
sigma1 = 2*[4, 0.5, 0.2, 0.1, 0.3; 0.5, 3, 0.1, 0.3, 0.2; 0.2, 0.1, 2, 0.4, 0.5; 0.1, 0.3, 0.4, 3, 0.2; 0.3, 0.2, 0.5, 0.2, 5];

mu2 = [0, 0, 0, 0, 0];
sigma2 = 2*[2, 0.5, 0.3, 0.2, 0.1; 0.5, 1, 0.2, 0.4, 0.3; 0.3, 0.2, 2, 0.5, 0.2; 0.2, 0.4, 0.5, 2, 0.1; 0.1, 0.3, 0.2, 0.1, 3];

mu3 = [-1, -7, -3, -8, -5];
sigma3 = 2*[1.2, 0.1, 0.3, 0.4, 0.2; 0.1, 2.3, 0.4, 0.2, 0.3; 0.3, 0.4, 1, 0.1, 0.5; 0.4, 0.2, 0.1, 3, 0.4; 0.2, 0.3, 0.5, 0.4, 2];
rand_mu = rand(1,p);
% Define modified GMM parameters for test data
mu1_test = mu1 + 3*rand_mu;  % Slightly shift the mean for test data
mu2_test = mu2 - 3*rand_mu; 
mu3_test = mu3 + 3*rand_mu; 
n_MH    =100 ;

rand_Sigma = rand(p) ;
sigma1_test = sigma1 + diag(rand_Sigma).*eye(p);  % Slightly change the covariance for test data
sigma2_test = sigma2 + diag(rand_Sigma).*eye(p);
sigma3_test = sigma3 + diag(rand_Sigma).*eye(p);





W = [0.4, 0.3, 0.3];

num_components = length(W);
n = 1400;  % number of subjects
p = 5;     % number of explanatory variables
n_test = 250 ;

for mc = 1 : MC 

gm = gmdistribution([mu1; mu2;mu3], cat(3, sigma1, sigma2,sigma3), W);
gm2 = gmdistribution([mu1_test; mu2_test;mu3_test], cat(3, sigma1_test, sigma2_test,sigma3_test), W);
X1 = random(gm, n);
X2 = random(gm2, n_test);
X3 = random(gm, n_test);
beta1 = [0        , 1    ,  1   , 0.5    , 0.3   ,  0.4];
beta2 = [-0.5     ,-0.2  , -0.5 , 0.6    , 0.3   ,-0.2 ];
beta3 = [0.2      , 0.3  , -0.4 , 0.4    , -0.2  , 1   ];
beta4 = [0.1      , -0.5 , 0.1  ,-0.1    , 1     , -0.3];

beta_true = [beta1; beta2 ;beta3 ;beta4]';


  eta1 = beta1(1) + X1 * beta1(2:end)'; % regression coeff
  eta2 = beta2(1) + X1 * beta2(2:end)'; % regression coeff
  eta3 = beta3(1) + X1 * beta3(2:end)'; % regression coeff
  eta4 = beta4(1) + X1 * beta4(2:end)'; % regression coeff
  
  eta1_test = beta1(1) + X2 * beta1(2:end)'; % regression coeff
  eta2_test = beta2(1) + X2 * beta2(2:end)'; % regression coeff
  eta3_test = beta3(1) + X2 * beta3(2:end)'; % regression coeff
  eta4_test = beta4(1) + X2 * beta4(2:end)'; % regression coeff
  
  
    %eta3 = beta3(1) + x * beta3(2:end)'; % regression coeff
    pdf_logistic1 = exp(eta1) ./ (1 + exp(eta1) + exp(eta2)+ exp(eta3)+ exp(eta4)); % pdf from the logistic distribution for y=1
    pdf_logistic2 = exp(eta2) ./ (1 +exp(eta1) + exp(eta2)+ exp(eta3)+ exp(eta4)); % pdf from the logistic distribution for y=2
    pdf_logistic3 = exp(eta3) ./ (1 + exp(eta1) + exp(eta2)+ exp(eta3)+ exp(eta4)); % pdf from the logistic distribution for y=3
    pdf_logistic4 = exp(eta4) ./ (1 + exp(eta1) + exp(eta2)+ exp(eta3)+ exp(eta4)); % pdf from the logistic distribution for y=4
    pdf_logistic5 = 1 ./ (1 + exp(eta1) + exp(eta2)+ exp(eta3)+ exp(eta4)); % pdf from the logistic distribution for y=5

     %eta3 = beta3(1) + x * beta3(2:end)'; % regression coeff
    pdf_logistic1_test = exp(eta1_test) ./ (1 + exp(eta1_test) + exp(eta2_test)+ exp(eta3_test)+ exp(eta4_test)); % pdf from the logistic distribution for y=1
    pdf_logistic2_test = exp(eta2_test) ./ (1 +exp(eta1_test) + exp(eta2_test)+ exp(eta3_test)+ exp(eta4_test)); % pdf from the logistic distribution for y=2
    pdf_logistic3_test = exp(eta3_test) ./ (1 + exp(eta1_test) + exp(eta2_test)+ exp(eta3_test)+ exp(eta4_test)); % pdf from the logistic distribution for y=3
    pdf_logistic4_test = exp(eta4_test) ./ (1 + exp(eta1_test) + exp(eta2_test)+ exp(eta3_test)+ exp(eta4_test)); % pdf from the logistic distribution for y=4
    pdf_logistic5_test = 1 ./ (1 + exp(eta1_test) + exp(eta2_test)+ exp(eta3_test)+ exp(eta4_test)); % pdf from the logistic distribution for y=5

    % Generate y based on the probabilities
    y      = zeros(n, 1);
    R      = zeros(n, K);
    y_test = zeros(n_test, 1);
    R_test = zeros(n_test, K);
    for i = 1:n
        P = [pdf_logistic1(i), pdf_logistic2(i), pdf_logistic3(i), pdf_logistic4(i), pdf_logistic5(i)];
        y(i) = randsample([1, 2, 3, 4, 5], 1, true, P);
        R(i,y(i)) = 1; 
    end
   for i = 1:n_test
        P_test = [pdf_logistic1_test(i), pdf_logistic2_test(i), pdf_logistic3_test(i), pdf_logistic4_test(i), pdf_logistic5_test(i)];
        y_test(i) = randsample([1, 2, 3, 4, 5], 1, true, P_test);
        R_test(i,y(i)) = 1; 
    end
    beta_estimated = [beta1; beta2 ;beta3; beta4];
    
 mu_SAEM =    [mu1; mu2;mu3] ;
 Sigma_SAEM  =   cat(3, sigma1, sigma2,sigma3) ;
    
    
 
nb_echantillons_apprentissage = round(0.75* n);
indices_aleatoires = randperm(n);
   
       
donnees_apprentissage    = X1(indices_aleatoires(1:nb_echantillons_apprentissage), :);
etiquettes_apprentissage = y(indices_aleatoires(1:nb_echantillons_apprentissage));
donnees_test             = X1(indices_aleatoires(nb_echantillons_apprentissage+1:end), :);
etiquettes_test          = y(indices_aleatoires(nb_echantillons_apprentissage+1:end));
% donnees_test    = X2 ; 
% etiquettes_test = y_test ; 
 for i = 1 : Ratio_Miss
    M1 = ones(size(donnees_apprentissage));
    M2 = ones(size(donnees_test));
    nanIndices1 = rand(size(donnees_apprentissage)) < 0.3*i ;  
    nanIndices2 = rand(size(donnees_test)) < 0.3*i ;  

    M1(nanIndices1) = NaN;
    M2(nanIndices2) = NaN;
   
    donnees_apprentissage_miss  = donnees_apprentissage .* M1;
    donnees_test_miss           = donnees_test .* M2;
    etiquettes_apprentissage_nona = etiquettes_apprentissage ; 
    etiquettes_test_nona          = etiquettes_test ;
% Remove rows containing only NaNs
    nan_rows1                       =   all(isnan(donnees_apprentissage_miss), 2);
    donnees_apprentissage_miss      =   donnees_apprentissage_miss(~nan_rows1, :);
    etiquettes_apprentissage        =   etiquettes_apprentissage(~nan_rows1);
    nan_rows2                       =   all(isnan(donnees_test_miss), 2);
    donnees_test_miss               =   donnees_test_miss(~nan_rows2, :);
    etiquettes_test                 =   etiquettes_test(~nan_rows2);
    

 
%% No NA
tic
 [beta_estimated_nona, w_nona, mu_nona, Sigma_nona,stats_nona.se(:,:,mc)] =  M_SAEM(donnees_apprentissage, etiquettes_apprentissage_nona, num_components_gmm);
 temp_nona(mc) = toc ;
 beta_estimated_nona_mc(:,:,mc) =  beta_estimated_nona;
 
 
% Prediction on the NO NA test data

y_pred_probabilities_noNA = mnrval(beta_estimated_nona, donnees_test);

% Predictions are in the form of probabilities for each class

 [~, y_pred_logistic_noNA] = max(y_pred_probabilities_noNA, [], 2);
% calcule de OV pour NO NA 

  mat_conf_noNA = confusionmat(y_pred_logistic_noNA,etiquettes_test_nona');
  correct_predictions_noNA = sum(diag(mat_conf_noNA));
  total_observations_NONA = sum(mat_conf_noNA(:));
  overall_accuracy_noNA(mc,i) = correct_predictions_noNA / total_observations_NONA;
  y_predict_mean_mc(mc,:)              = y_pred_logistic_noNA;
  Mat_conf_NONA(:,:,mc)  =  mat_conf_noNA ;

%% Complete Case
tic
[beta_CC(:,:,mc), stats_cc.se(:,:,mc), y_prediction_CC] = CC(donnees_apprentissage_miss, etiquettes_apprentissage, donnees_test_miss) ;
temp_CC(mc) = toc ;


 % calcule de OV pour CC
 %mat_conf_CC = confusionmat(y_prediction_CC,etiquettes_test');
 %correct_predictions_CC = sum(diag(mat_conf_CC));
 %total_observations_CC = sum(mat_conf_CC(:));
 %overall_accuracy_CC(mc,i) = correct_predictions_CC / total_observations_CC; 


%% SAEM-GMM
tic
  [beta_estimated_SAEM_gmm, w_SAEM_gmm, mu_SAEM_gmm, Sigma_SAEM_gmm,stats_saem_gmm.se(:,:,mc)] =  M_SAEM(donnees_apprentissage_miss, etiquettes_apprentissage, num_components_gmm);
 
  [y_predict_saem_gmm]                                       =  saem_gmm_predict(donnees_test_miss,mu_SAEM_gmm,Sigma_SAEM_gmm,w_SAEM_gmm, beta_estimated_SAEM_gmm, nb_Sample_mcmc,num_claas);

 temp_saem_gmm(mc) = toc ;

% Calculate OV for SAEM-GMM
  mat_conf_saem_gmm                 = confusionmat(y_predict_saem_gmm,etiquettes_test');
  correct_predictions_saem_gmm      = sum(diag(mat_conf_saem_gmm));
  total_observations_saem_gmm       = sum(mat_conf_saem_gmm(:));
  overall_accuracy_saem_gmm(mc,i)   = correct_predictions_saem_gmm / total_observations_saem_gmm;
  beta_estimated_SAEM_gmm_mc(:,:,mc) = beta_estimated_SAEM_gmm;
  mu_SAEM_gmm_mc(:,:,mc)              = mu_SAEM_gmm ;
  Sigma_SAEM_gmm_mc(:,:,:,mc)         =Sigma_SAEM_gmm ;
%% SAEM
% Train a multiclass logistic regression model (SAEM)
tic
 [beta_estimated_SAEM, w_SAEM, mu_SAEM, Sigma_SAEM,stats_saem.se(:,:,mc)]       =  M_SAEM(donnees_apprentissage_miss, etiquettes_apprentissage, num_component_saem);
 [y_predict_saem]                                 =  saem_gmm_predict(donnees_test_miss, mu_SAEM , Sigma_SAEM,w_SAEM, beta_estimated_SAEM , nb_Sample_mcmc,num_claas);
 temp_saem(mc) = toc ;

% Compute OV for SAEM

  mat_conf_saem                  = confusionmat(y_predict_saem,etiquettes_test');
  correct_predictions_saem       = sum(diag(mat_conf_saem));
  total_observations_saem        = sum(mat_conf_saem(:));
  overall_accuracy_saem(mc,i)    = correct_predictions_saem / total_observations_saem;
  beta_estimated_SAEM_mc(:,:,mc) =  beta_estimated_SAEM;
  Mat_conf_saem(:,:,mc)          =  mat_conf_saem;
  Sigma_SAEM_mc(:,:,:,mc)        =  Sigma_SAEM;
  mu_SAEM_mc(:,:,mc)             =  mu_SAEM;
%% MEAN-IMPUTATION

% Train a multiclass logistic regression model (Mean Imputation)

tic 
 [logistic_model_mean,stats_mean.se(:,:,mc)]        = MeanImputation_new(donnees_apprentissage_miss,etiquettes_apprentissage);
 [X_mean_test]                = MeanImputation_test(donnees_test_miss);
 temp_mean(mc) = toc ;

% Prediction on the test data for MEAN-IMPUTATION 

y_pred_probabilities_MEAN = mnrval(logistic_model_mean, X_mean_test);

% Predictions are in the form of probabilities for each class

 [~, y_pred_logistic_MEAN] = max(y_pred_probabilities_MEAN, [], 2);
 
% Compute OV for mean imputation

  mat_conf_mean = confusionmat(y_pred_logistic_MEAN,etiquettes_test');
  correct_predictions_mean = sum(diag(mat_conf_mean));
  total_observations_mean = sum(mat_conf_mean(:));
  overall_accuracy_mean(mc,i) = correct_predictions_mean / total_observations_mean;
  beta_estimated_mean_mc (:,:,mc) = logistic_model_mean ;
 % y_predict_mean_mc(mc,:)              = y_pred_logistic_MEAN;
 % Mat_conf_mean(:,:,mc)           =  mat_conf_mean ;




%% MICE

% Train a multiclass logistic regression model (MICE Imputation)
tic
 [logistic_model_MICE,stats_mice.se(:,:,mc)]      = MICE_new(donnees_apprentissage_miss,etiquettes_apprentissage);
 [X_mice_test]              = MICE_test(donnees_test_miss);
 temp_mice(mc) = toc ;

% Prediction on the test data for MICE

y_pred_probabilities_MICE = mnrval(logistic_model_MICE, X_mice_test);

% Predictions are in the form of probabilities for each class

 [~, y_pred_logistic_MICE] = max(y_pred_probabilities_MICE, [], 2);

% Compute OV for MICE linear regression

  mat_conf_mice = confusionmat(y_pred_logistic_MICE,etiquettes_test');
  correct_predictions_mice = sum(diag(mat_conf_mice));
  total_observations_mice = sum(mat_conf_mice(:));
  overall_accuracy_mice(mc,i) = correct_predictions_mice / total_observations_mice;
  beta_estimated_mice_mc(:,:,mc) = logistic_model_MICE ;
  y_predict_mice_mc(mc,:)              = y_pred_logistic_MICE;
   Mat_conf_mice(:,:,mc)  =  mat_conf_mice ;
 %% MISS-FOREST
% Train a multiclass logistic regression model (MISS_forest)
tic
 [logistic_model_MICE_forest,stats_miss_f.se(:,:,mc)]      = MICE_forest_new(donnees_apprentissage_miss,etiquettes_apprentissage);
 [X_mice_test_forest]              = MICE_test_forest(donnees_test_miss);
 temp_mice_forest(mc) = toc ;

% Prediction on the test_MICE data
y_pred_probabilities_MICE_forest = mnrval(logistic_model_MICE_forest, X_mice_test_forest);

% The predictions are in the form of probabilities for each class

 [~, y_pred_logistic_MICE_forest] = max(y_pred_probabilities_MICE_forest, [], 2); 

% Calculate OV for miss-FOREST

  mat_conf_mice_forest = confusionmat(y_pred_logistic_MICE_forest,etiquettes_test');
  correct_predictions_mice_f = sum(diag(mat_conf_mice_forest));
  total_observations_mice_f = sum(mat_conf_mice_forest(:));
  overall_accuracy_mice_forest(mc,i) = correct_predictions_mice_f / total_observations_mice_f;
  beta_estimated_mice_forest_mc (:,:,mc)= logistic_model_MICE_forest ;
  y_predict_mice_forest_mc(mc,:)              = y_pred_logistic_MICE_forest;
  Mat_conf_miss_forest(:,:,mc)  =  mat_conf_mice_forest ;


%% Mean_Cond_Imputation
% Train a multiclass logistic regression model (mean_cond_Imputation)
tic
 [logistic_model_mean_cond,stats_mean_cond.se(:,:,mc)]  = mean_cond_new(donnees_apprentissage_miss,etiquettes_apprentissage);
 [x_test_mu_cond]            = mean_cond_test(donnees_test_miss);
 temp_mean_cond(mc) = toc ;


% Prediction on the test data for MEAN_COND
y_pred_probabilities_MEAN_COND = mnrval(logistic_model_mean_cond, x_test_mu_cond);

% Predictions are in the form of probabilities for each class

 [~, y_pred_logistic_mean_cond] = max(y_pred_probabilities_MEAN_COND, [], 2);
 
% calculation of OV for mean-cond

  mat_conf_mean_cond = confusionmat(y_pred_logistic_mean_cond,etiquettes_test');
  correct_predictions_mean_cond = sum(diag(mat_conf_mean_cond));
  total_observations_mean_cond = sum(mat_conf_mean_cond(:));
  overall_accuracy_mean_cond(mc,i) = correct_predictions_mean_cond / total_observations_mean_cond;
  beta_estimated_mean_cond_mc (:,:,mc) = logistic_model_mean_cond ;
  y_predict_mean_cond_mc(mc,:)              = y_pred_logistic_mean_cond;

%% Multiple_EM 
 tic
 [logistic_model_Mltiple_EM,stats_mlt_em.se(:,:,mc)] = Multiple_EM_new(donnees_apprentissage_miss,etiquettes_apprentissage);
 [x_imputed_mlt_em_test]          = Multiple_EM_test(donnees_test_miss);
 temp_mlt_em(mc) = toc ;

% Prediction on the test_multiple_em data

y_pred_probabilities_Mltiple_EM = mnrval(logistic_model_Mltiple_EM, x_imputed_mlt_em_test);

% Predictions are in the form of probabilities for each class
 [~, y_pred_logistic_Mltiple_EM ] = max(y_pred_probabilities_Mltiple_EM, [], 2);

% Calculation of OV for multiple-em
  mat_conf_mean_multiple_em = confusionmat(y_pred_logistic_Mltiple_EM,etiquettes_test');
  correct_predictions_multiple_em = sum(diag(mat_conf_mean_multiple_em));
  total_observations_multiple_em = sum(mat_conf_mean_multiple_em(:));
  overall_accuracy_multiple_em(mc,i) = correct_predictions_multiple_em / total_observations_multiple_em;
  beta_estimated_multiple_em_mc (:,:,mc) = logistic_model_Mltiple_EM ;
  y_predict_mlt_em_mc(mc,:)              = y_pred_logistic_Mltiple_EM;

 
 

 
end
disp(mc)
%save pred_diff2
end
                     
    
% Sélectionner les valeurs d'overall accuracy pour mc=1:MC et i=1
mc_range = 1:MC;  % plage de valeurs pour mc
i = 1;  % Sélection du i correspondant

% Extraire les accuracies pour chaque méthode
accuracy_mean = overall_accuracy_mean(mc_range, i);
accuracy_saem = overall_accuracy_saem(mc_range, i);
accuracy_saem_gmm = overall_accuracy_saem_gmm(mc_range, i);
accuracy_mice = overall_accuracy_mice(mc_range, i);
accuracy_miss_forest = overall_accuracy_mice_forest(mc_range, i);

% Créer une matrice regroupant toutes les accuracies
accuracy_data = [accuracy_mean(:), accuracy_mice(:), accuracy_miss_forest(:),  accuracy_saem(:) ,accuracy_saem_gmm(:)];

% Noms des estimateurs
estimator_names = {'Mean', 'MICE','MissForest', 'SAEM', 'SAEM-GMM'};
% Créer les boxplots
figure;
boxplot(accuracy_data, 'Labels', estimator_names);
ylabel('Overall Accuracy');
grid on;

