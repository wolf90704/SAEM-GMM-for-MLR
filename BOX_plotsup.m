a=1 ;
b= 74;


l   = 5 ; 
col = 1;  %  62 bon
% Exemple de données fictives (remplacez-les par vos propres données)
%beta_estimated_CC_box = beta_estimated_CC_mc(l, col, a:b) - beta_true (l,col);
beta_estimated_mean_cond_mc_box = beta_estimated_mean_cond_mc(l, col,a:b) - beta_true (l,col);
beta_estimated_mean_mc_box = beta_estimated_mean_mc(l, col, a:b)- beta_true (l,col);
%beta_estimated_nona_mc_box = beta_estimated_nona_mc(l, col, a:b)- beta_true (l,col);
beta_estimated_mice_forest_mc_box = beta_estimated_mice_forest_mc(l, col, a:b)- beta_true (l,col);
beta_estimated_mice_mc_box = beta_estimated_mice_mc(l, col, a:b)- beta_true (l,col);
beta_estimated_multiple_em_mc_box = beta_estimated_multiple_em_mc(l, col, a:b)- beta_true (l,col);
beta_estimated_SAEM_mc_box = beta_estimated_SAEM_mc(l, col, a:b)- beta_true (l,col);
beta_estimated_SAEM_st_mc_box = beta_estimated_SAEM_gmm_mc(l, col, a:b)- beta_true (l,col);

% Remodeler les données beta
%beta_estimated_CC_box           = reshape(beta_estimated_CC_box, [], 1);
%beta_estimated_nona_mc_box      = reshape(beta_estimated_nona_mc_box, [], 1);
beta_estimated_mean_cond_mc_box = reshape(beta_estimated_mean_cond_mc_box, [], 1);
beta_estimated_mean_mc_box = reshape(beta_estimated_mean_mc_box, [], 1);
beta_estimated_mice_forest_mc_box = reshape(beta_estimated_mice_forest_mc_box, [], 1);
beta_estimated_mice_mc_box = reshape(beta_estimated_mice_mc_box, [], 1);
beta_estimated_multiple_em_mc_box = reshape(beta_estimated_multiple_em_mc_box, [], 1);
beta_estimated_SAEM_mc_box = reshape(beta_estimated_SAEM_mc_box, [], 1);
beta_estimated_SAEM_st_mc_box = reshape(beta_estimated_SAEM_st_mc_box, [], 1);

% Combinez les données dans une matrice
data_matrix = [  beta_estimated_mean_mc_box, ...
     beta_estimated_mice_mc_box,beta_estimated_mice_forest_mc_box, ...
    beta_estimated_SAEM_mc_box,beta_estimated_SAEM_st_mc_box];

% Créez le boxplot
figure;
boxplot(data_matrix, 'Labels', {  'Mean', 'Mice','Miss_Forest','SAEM', 'SAEM_gmm'});


ylabel('Bias of $\hat{\beta}_{23}$', 'Interpreter', 'latex');

% Ajouter la ligne horizontale rouge à y = 0
hold on;
y_zero = 0;
line(get(gca, 'XLim'), [y_zero, y_zero], 'Color', 'red', 'LineStyle', '--', 'LineWidth', 1);
hold off;

ls = 1 ;
cols = 1 ;
s_mean_cond      =zeros(6,4,b);
s_mean           =zeros(6,4,b);
s_mice           =zeros(6,4,b);
s_miss_forest     =zeros(6,4,b);
s_mlt_em         =zeros(6,4,b);
s_nona           =zeros(6,4,b);
s_SAEM           =zeros(6,4,b);
s_SAEM_GMM       =zeros(6,4,b);

 
for i = a:b
    s_CC(:, :, i)          = stats_nona.se(i).se;
 %   s_mean_cond(:, :, i)   = stats_mean_cond.se(i).se;
    s_mean(:, :, i)        = stats_mean.se(i).se;
    s_mice(:, :, i)        = stats_mice.se(i).se;
    s_miss_forest(:, :, i) = stats_mice_forest.se(i).se;
%    s_mlt_em(:, :, i)      = stats_mlt_em.se(i).se;
    s_nona(:, :, i)        = stats_nona.se(i).se;
    s_SAEM(:, :, i)        = stats_saem.se(i).se;
    s_SAEM_GMM(:, :, i)    = stats_saem_gmm.se(i).se;
     
end


% Remodeler les données d'écart-type
stats_CC_se_box = reshape(s_CC(l, col, a:b), [], 1);
%stats_mean_cond_se_box = reshape(s_mean_cond(l, col,  a:b), [], 1);
stats_mean_se_box = reshape(s_mean(l, col, a:b), [], 1);
stats_mice_se_box = reshape(s_mice(l, col,  a:b), [], 1);
stats_mice_forest_se_box = reshape(s_miss_forest(l, col,  a:b), [], 1);
%stats_mlt_em_se_box = reshape(s_mlt_em(l, col, a:b), [], 1);
stats_nona_se_box = reshape(s_nona(l, col,  a:b), [], 1);
stats_saem_se_box = reshape(s_SAEM(l, col,  a:b), [], 1);
stats_saem_gmm_se_box = reshape(s_SAEM_GMM(l, col,  a:b), [], 1);

% Combinez les données dans une matrice
data_matrix_se = [stats_nona_se_box,stats_mean_se_box, stats_mice_se_box, stats_mice_forest_se_box, ...
     stats_saem_se_box, stats_saem_gmm_se_box];

% Créez le boxplot pour les écarts-types
figure;
boxplot(data_matrix_se, 'Labels', {'NONA','Mean', 'MICE', 'Miss_Forest', ...
      'SAEM', 'SAEM_gmm'});
title('Boxplot des écarts-types pour différents estimateurs');
xlabel('Estimateurs');
ylabel('Valeurs des écarts-types');




% Create the boxplots for the bias and estimated standard errors
figure;

% Create the boxplot for the bias
subplot(2, 1, 1);
boxplot(data_matrix, 'Labels', { 'no NA' , 'Mean', 'Mice','Miss_Forest','SAEM', 'SAEM_gmm'});
ylabel('Bias of $\hat{\beta}_{23}$', 'Interpreter', 'latex');

% Add the red horizontal line at y = 0
hold on;
y_zero = 0;
line(get(gca, 'XLim'), [y_zero, y_zero], 'Color', 'red', 'LineStyle', '--', 'LineWidth', 1);
hold off;

% Create the boxplot for the estimated standard errors
subplot(2, 1, 2);
boxplot(data_matrix_se, 'Labels', {'no NA','Mean', 'Mice', 'Miss_Forest', 'SAEM', 'SAEM_gmm'});
ylabel('Standard Error of $\hat{\beta}_{23}$', 'Interpreter', 'latex');

% Add the red points for empirical standard deviation of $\hat{\beta}_{32}$
hold on;
red_points = std(data_matrix);
scatter(1:numel(red_points), red_points, 'r', 'filled');
hold off;

% Adjust other settings as needed