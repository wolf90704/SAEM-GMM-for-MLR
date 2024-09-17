function [coverage_mean, length_mean] = calculate_coverage(true_betas, beta_estimates, stat)
% pour 95prc sa masse pour un estimateur asym gauss se trouve entre +1.96se
% et -1.96se
    % Initialisation des résultats
    [p, k, num_monte_carlo] = size(beta_estimates);
    coverage_results = zeros(p, k, num_monte_carlo);
    length_results = zeros(p, k, num_monte_carlo);

    % Boucle sur les itérations de Monte Carlo
    for mc = 1:num_monte_carlo
        % Bêtas estimés pour cette itération
        beta_est = beta_estimates(:,:,mc);
        se       = stat.se(mc).se ;
        % Erreurs standard

        if length(beta_est) == length(se)
            % Calcul des intervalles de confiance
            ci_lower = beta_est - 1.96 * se; % 1.96 for a 95% confidence interval
            ci_upper = beta_est + 1.96 * se;

            % Vérification de la couverture et calcul de la longueur de l'intervalle
            for i = 1:p
                for j = 1:k
                    if i <= p && j <= k
                        if true_betas(i, j) >= ci_lower(i, j) && true_betas(i, j) <= ci_upper(i, j)
                            coverage_results(i, j, mc) = 1;
                        end
                        length_results(i, j, mc) = abs(ci_upper(i, j) - ci_lower(i, j));
                    end
                end
            end
        end
    end

    % Calcul de la couverture moyenne et de la longueur moyenne de l'intervalle
    coverage_mean = mean(coverage_results, 3) * 100;
    length_mean = mean(length_results, 3) * 100;
end