function [expectations theta] = GaussMixt(X,k,theta,itermax)
[N dim] = size(X);




handles.estep = @GMM_E_Step;
handles.mstep = @GMM_M_Step;

densE_st = GMM_densE_st(X,theta);
logLike = sum(log(sum(repmat(theta.tau,N,1).*densE_st,2)));
%fprintf('Log-Likelihood after initialization: %.3f\n',logLike)

[expectations theta] = EM_General(X,[],theta,handles,itermax);

end
