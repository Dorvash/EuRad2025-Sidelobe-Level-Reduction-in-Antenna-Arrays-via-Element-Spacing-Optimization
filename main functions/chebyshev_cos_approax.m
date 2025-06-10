function [der_g, g] = chebyshev_cos_approax(lb, ub, N, X0, coeff_mat)

X0 = permute(X0, [3, 1, 2]);

c = chebyshev_cos_coeffs(lb, ub, N, X0);

g = pagemtimes(coeff_mat, c);

der_g = (0:N).' .* g;
der_g(end,:,:) = [];


end