function coeffs = chebyshev_cos_coeffs(a, b, N, dm)
%CHEBYSHEV_EXP_APPROX Approximates exp(x) using Chebyshev polynomials.
%
%   coeffs = chebyshev_exp_coeffs(a, b, N)
%
%   Inputs:
%       a         - Lower bound of the interval.
%       b         - Upper bound of the interval.
%       N         - Degree of the Chebyshev polynomial.
%
%   Output:
%       coeffs        - Chebyshev coefficients used in the approximation.

    % Ensure that the interval [a, b] is valid
    if a >= b
        error('Invalid interval: a must be less than b.');
    end

    % Step 1: Compute Chebyshev nodes
    k = (0:N).';
    theta_k = (2 * k + 1) * pi / (2 * (N + 1));
    t_k = cos(theta_k);

    % Map Chebyshev nodes to x nodes in [a, b]
    x_k = ((b - a) * t_k + (b + a)) / 2;

    % Step 2: Evaluate exp(x) at Chebyshev nodes
    f_k = cos(x_k + dm);

    % Step 3: Compute Chebyshev coefficients
    c = zeros(N + 1, size(f_k,2), size(f_k,3));
    for n = 0:N
        if n == 0
            c(n + 1,: , :) = (1 / (N + 1)) * sum(f_k);
        else
            c(n + 1, :, :) = (2 / (N + 1)) * sum(f_k .* cos(n * theta_k));
        end
    end

    coeffs = c; % Save coefficients for output

end
