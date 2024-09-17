%% Generate dataset
MC = 1 ;
dm = 1; 
num_claas  = 5;
dim = 5 ;
nb_Sample_mcmc = 20;
num_components_gmm = 3 ;
num_component_saem     = 1 ;
overall_accuracy_saem_gmm    = zeros(MC,dm);
overall_accuracy_saem        = zeros(MC,dm);
overall_accuracy_mice        = zeros(MC,dm);
overall_accuracy_mean_cond   = zeros(MC,dm);
overall_accuracy_mean        = zeros(MC,dm);
overall_accuracy_multiple_em = zeros(MC,dm);
overall_accuracy_mice_forest = zeros(MC,dm);
beta_estimated_SAEM_gmm_mc          = zeros(dim+1,num_claas-1,MC);
beta_estimated_SAEM_mc              = zeros(dim+1,num_claas-1,MC);
beta_estimated_mice_mc              = zeros(dim+1,num_claas-1,MC);
beta_estimated_mean_cond_mc         = zeros(dim+1,num_claas-1,MC);
beta_estimated_mean_mc              = zeros(dim+1,num_claas-1,MC);
beta_estimated_multiple_em_mc       = zeros(dim+1,num_claas-1,MC);
beta_estimated_mice_forest_mc       = zeros(dim+1,num_claas-1,MC);

% Définition du mélange GMM :
K = 3;
mu1 = [1, 1, 1, 1, 1];
sigma1 = [4, 0.5, 0.2, 0.1, 0.3; 0.5, 3, 0.1, 0.3, 0.2; 0.2, 0.1, 2, 0.4, 0.5; 0.1, 0.3, 0.4, 3, 0.2; 0.3, 0.2, 0.5, 0.2, 5];

mu2 = [4.6, 4.6, 5.6, 6.6, 7.6];
sigma2 = [2, 0.5, 0.3, 0.2, 0.1; 0.5, 1, 0.2, 0.4, 0.3; 0.3, 0.2, 2, 0.5, 0.2; 0.2, 0.4, 0.5, 2, 0.1; 0.1, 0.3, 0.2, 0.1, 3];

mu3 = [-3.7, -3.7, -5.7, -4.7, -1.7];
sigma3 = [1.2, 0.1, 0.3, 0.4, 0.2; 0.1, 2.3, 0.4, 0.2, 0.3; 0.3, 0.4, 1, 0.1, 0.5; 0.4, 0.2, 0.1, 3, 0.4; 0.2, 0.3, 0.5, 0.4, 2];
tau_SA_control = [ 0.4, 0.7 ,1];
W = [0.3, 0.2, 0.5];

num_components = length(W);
n = 1250;  % number of subjects
p = 5;     % number of explanatory variables




gm = gmdistribution([mu1; mu2;mu3], cat(3, sigma1, sigma2,sigma3), W);
X1 = random(gm, n);

beta1 = [0    , 1 ,  2   , 2.5   , 3   ,  4];
beta2 = [-1   ,-2 , -2.5 , 0.6   , 0.3 ,-0.2];
beta3 = [2    , 3 , -4   , 4     , -2  , 1 ];
beta4 = [1    , -0.5 , 1  ,-1    , 1   , -3];

beta_true = [beta1; beta2;beta3;beta4]';


  eta1 = beta1(1) + X1 * beta1(2:end)'; % regression coeff
  eta2 = beta2(1) + X1 * beta2(2:end)'; % regression coeff
  eta3 = beta3(1) + X1 * beta2(2:end)'; % regression coeff
  eta4 = beta4(1) + X1 * beta2(2:end)'; % regression coeff

    
    
    %eta3 = beta3(1) + x * beta3(2:end)'; % regression coeff
    pdf_logistic1 = exp(eta1) ./ (1 + exp(eta1) + exp(eta2)+ exp(eta3)+ exp(eta4)); % pdf from the logistic distribution for y=1
    pdf_logistic2 = exp(eta2) ./ (1 +exp(eta1) + exp(eta2)+ exp(eta3)+ exp(eta4)); % pdf from the logistic distribution for y=2
    pdf_logistic3 = exp(eta3) ./ (1 + exp(eta1) + exp(eta2)+ exp(eta3)+ exp(eta4)); % pdf from the logistic distribution for y=3
    pdf_logistic4 = exp(eta4) ./ (1 + exp(eta1) + exp(eta2)+ exp(eta3)+ exp(eta4)); % pdf from the logistic distribution for y=4
    pdf_logistic5 = 1 ./ (1 + exp(eta1) + exp(eta2)+ exp(eta3)+ exp(eta4)); % pdf from the logistic distribution for y=5

    % Generate y based on the probabilities
    y = zeros(n, 1);
    R = zeros(n, K);

    for i = 1:n
        P = [pdf_logistic1(i), pdf_logistic2(i), pdf_logistic3(i), pdf_logistic4(i), pdf_logistic5(i)];
        y(i) = randsample([1, 2, 3, 4, 5], 1, true, P);
        R(i,y(i)) = 1; 
    end
    
nb_echantillons_apprentissage = round(0.75 * n);
indices_aleatoires = randperm(n);
for k = 1 : length(tau_SA_control )

    for i = 1 : 1
       
donnees_apprentissage    = X1(indices_aleatoires(1:nb_echantillons_apprentissage), :);
etiquettes_apprentissage = y(indices_aleatoires(1:nb_echantillons_apprentissage));
donnees_test             = X1(indices_aleatoires(nb_echantillons_apprentissage+1:end), :);
etiquettes_test          = y(indices_aleatoires(nb_echantillons_apprentissage+1:end));


    M1 = ones(size(donnees_apprentissage));
    M2 = ones(size(donnees_test));
    nanIndices1 = rand(size(donnees_apprentissage)) < 0.1*i ;  
    nanIndices2 = rand(size(donnees_test)) < 0.1*i ;  

    M1(nanIndices1) = NaN;
    M2(nanIndices2) = NaN;
    end  
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

[beta_estimated_nona(:,:,:,k), w_nona, mu_nona, Sigma_nona,stats_nona.se] =  M_SAEM_bihavor(X1, y,tau_SA_control(k), num_components_gmm);


[beta_estimated_SAEM_gmm(:,:,:,k), w_SAEM_gmm, mu_SAEM_gmm, Sigma_SAEM_gmm,stats_saem_gmm.se] =  M_SAEM_bihavor(donnees_apprentissage_miss, etiquettes_apprentissage,tau_SA_control(k), num_components_gmm);
%[y_predict_saem_gmm]                                       =  saem_gmm_predict(donnees_test_miss, mu_SAEM_gmm, Sigma_SAEM_gmm,w_SAEM_gmm, beta_estimated_SAEM_gmm, nb_Sample_mcmc,num_claas);


 

% Entraîner un modèle de régression logistique multiclasse(SAEM)

[beta_estimated_SAEM(:,:,:,k), w_SAEM, mu_SAEM, Sigma_SAEM,stats_saem.se]       =  M_SAEM_bihavor(donnees_apprentissage_miss, etiquettes_apprentissage,tau_SA_control(k), num_component_saem);
%[y_predict_saem]                                 =  saem_gmm_predict(donnees_test_miss, mu_SAEM, Sigma_SAEM,w_SAEM, beta_estimated_SAEM, nb_Sample_mcmc,num_claas);

% calcule de OV pour saem

 
end
end

% Nombre d'itérations
iter_max = size(beta_estimated_SAEM, 3);

% Boucle sur chaque k
for k = 1:3
    figure; % Crée une nouvelle figure pour chaque k

    % Tracé de l'évolution de beta_estimated_SAEM(1,1,iter)
    subplot(2,1,1);
    plot(1:iter_max, squeeze(beta_estimated_SAEM(1,1,:,k)), 'b-', 'LineWidth', 2);
    title(['Evolution de beta\_estimated\_SAEM pour k = ' num2str(k)]);
    xlabel('Itérations');
    ylabel('Valeur estimée');
    grid on;

    % Tracé de l'évolution de beta_estimated_SAEM_gmm(1,1,iter)
    subplot(2,1,2);
    plot(1:iter_max, squeeze(beta_estimated_SAEM_gmm(1,1,:,k)), 'r-', 'LineWidth', 2);
    title(['Evolution de beta\_estimated\_SAEM\_gmm pour k = ' num2str(k)]);
    xlabel('Itérations');
    ylabel('Valeur estimée');
    grid on;

    % Légende
    legend('SAEM', 'SAEM\_gmm');
end    

% Valeurs spécifiques de tau (adaptez selon vos données)
% Valeurs spécifiques de tau (adaptez selon vos données)
tau_values = [0.4, 0.7, 1.0];

% Création d'une figure avec 6 sous-graphiques
figure;

% Cellules pour stocker les handles des axes
axes_handles = zeros(1, 6);

for k = 1:3
    % Graphique pour beta_estimated_SAEM
    subplot(2, 3, k);
    plot(1:iter_max, squeeze(beta_estimated_SAEM(1,1,:,k)), 'b-', 'LineWidth', 2);
    grid on;
    hold on; % Garder les graphiques existants
    
    % Stocker le handle de l'axe pour la liaison
    axes_handles(k) = gca;

    % Ajouter le tau en tant que titre
    title(['\tau = ' num2str(tau_values(k))]);

    % Graphique pour beta_estimated_SAEM_gmm sans pointillés
    subplot(2, 3, k+3);
    plot(1:iter_max, squeeze(beta_estimated_SAEM_gmm(1,1,:,k)), 'r-', 'LineWidth', 2);
    grid on;
    hold on; % Garder les graphiques existants
    
    % Stocker le handle de l'axe pour la liaison
    axes_handles(k+3) = gca;
end

% Ajouter une légende pour SAEM
legend(axes_handles(1), 'SAEM', 'Location', 'eastoutside');

% Ajouter une légende pour SAEM_gmm
legend(axes_handles(4), 'SAEM_gmm', 'Location', 'eastoutside');

% Lier les axes des graphiques
linkaxes(axes_handles, 'xy');