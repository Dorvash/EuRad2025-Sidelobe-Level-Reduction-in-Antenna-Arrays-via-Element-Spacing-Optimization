%%  PreProcessing
clear, close all
clc
%%  Initializing

ScreenSize          = get(0,'ScreenSize');

addpath('../main functions/')

theta               = -90:0.1:90;
theta_t_vec         = -30:10:30;
M_vec               = [16, 24, 32];
fc                  = 77e9;
c                   = 3e8;
lambda              = c/fc;

E_theta             = cosd(theta);
d_min               = lambda / 2;

delta               = 1i * 2 * pi / lambda * sind(theta);

q_vec               = (0:2).';

% - Optimization Setup
iter_max            = 500;
threshold           = 0;

% - PreAllocation
P                   = zeros(length(M_vec),length(theta), length(theta_t_vec));
P_dB_all            = zeros(length(M_vec),length(theta), length(theta_t_vec));
PSL_mat             = inf(length(M_vec), length(theta_t_vec));
ISL_mat             = inf(length(M_vec), length(theta_t_vec));
d_total_cell        = cell(size(M_vec));
M_iter              = zeros(size(M_vec));
m_cnt               = 0;
d_cell              = cell(length(M_vec),1);
d_conv_mat          = zeros(length(M_vec), iter_max);

%%  Chebyshev's Approximation Parameters

step_bound          = lambda / 16;
app_bound           = 2 * 2*pi/lambda * step_bound;

ub                  = app_bound;
lb                  = -app_bound;
N                   = 4;

syms x
t                   = (2 * x - (ub + lb)) / (ub - lb);
T_sym_prime         = chebyshevT(0:N, x);
T_sym               = subs(T_sym_prime, x, t);

coeff_cell          = arrayfun(@(y) double(coeffs(y + x^(N+1), 'all')), T_sym, 'UniformOutput', false);
coeff_mat_prime     = double(cell2mat(coeff_cell.'));
coeff_mat           = flipud(coeff_mat_prime(:,2:end).');

%%  Optimization

% - mu Optimization
P_des               = cell2mat(arrayfun(@(t) designPDes(theta, t+[0 0]), theta_t_vec, 'UniformOutput', false));
mu                  = mean(cosd(theta.') .* P_des) / mean(cosd(theta));
P_des               = sum(E_theta.' .* P_des ./ mu,2);

for M = M_vec

    m_cnt = m_cnt + 1;

    cnt_break = 0;

    % - d Optimization
    d_ini = (0:M-1).' * lambda / 2;
    A_ini = E_theta .* exp(delta .* d_ini);
    d = d_ini;
    A = E_theta .* A_ini;

    w_t = exp(1i*2*pi*d_ini/lambda.*sind(theta_t_vec));

    objfun = @(A,x) (norm(sum(abs((A)' * x).^2,2) - P_des, 2)^2);

    EPS = inf((N-1) + 2,1);
    EPS(end-1) = -step_bound;
    EPS(end) = step_bound;

    root_mask = zeros((N-1) + 2,1);
    root_mask(end) = 1;
    root_mask(end-1) = 1;

    clearvars d_total
    d_total(:,1) = d_ini / lambda;

    % Main loop
    for iter = 1:iter_max

        for m = 1:M

            D = repmat(d, 1, (N-1)+2);

            mask = double((1:M)' ~= m);

            alpha_t = (A.*mask).' * conj(w_t.*mask);
            beta = sum(conj(w_t(m,:)) .* conj(alpha_t),2);
            gamma = sum(abs(alpha_t).^2 + 1, 2);

            b0 = gamma.^2 + 2*abs(beta).^2 - 2*gamma.* P_des + P_des.^2;
            b1 = exp((delta.') * d(m)) .* (2.*beta.*gamma - 2*beta.*P_des);
            b2 = exp(2*(delta.') * d(m)) .* beta.^2;
            b_vec = [b0 b1 b2].';

            X0 = q_vec * 2 * pi / lambda * d(m) .* sind(theta) + angle(b_vec);
            der_g = chebyshev_cos_approax(lb, ub, N, X0, coeff_mat);

            c_kq = permute(2 * abs(b_vec) .* q_vec * 2 * pi / lambda .* sind(theta), [3, 1, 2]);
            p_vec = sum(c_kq .* der_g, [2, 3]);

            eps_hat = roots(flip(p_vec));
            root_mask(1:(N-1)) = double(imag(eps_hat) < 1e-8 & abs(real(eps_hat)) <= step_bound);
            EPS(1:(N-1)) = real(eps_hat);

            if m > 1
                d_mask = (d(m) + EPS - d(m - 1)) + 1e-6*lambda >= d_min;
            else
                d_mask = true(size(EPS));
            end

            root_mask = root_mask & d_mask;

            D(m, :) = D(m, :) + EPS.';
            A_cell = arrayfun(@(i) exp(D(:, i) * delta), 1:size(D, 2), 'UniformOutput', false);

            inner_objfunVal = cellfun(@(a)objfun(a,w_t), A_cell);

            idx = find(root_mask);
            [~, minIdx] = min(inner_objfunVal(idx));

            d(m) = d(m) + EPS(idx(minIdx));
            A = E_theta .* exp(delta .* d);
            w_t = exp(1i*2*pi*d/lambda.*sind(theta_t_vec));

        end

        d_total(:,iter+1) = sort(d - min(d)) / lambda;

        if iter >= 2 && norm(d_total(:,iter) - d_total(:,iter-1)) <= threshold
            cnt_break = cnt_break + 1;
        end

        if cnt_break == 3
            break
        end

    end

    % Optimized Beampattern
    P(m_cnt,:,:) = abs((A)' * w_t).^2;
    P_dB_all(m_cnt, :, :) = 10*log10(P(m_cnt,:,:) ./ max(P(m_cnt,:,:)));

    % PSL and ISL Calculation
    for i = 1:length(theta_t_vec)

        [PSL, ISL] = sidelobelevel(P_dB_all(m_cnt, :, i));

        PSL_mat(m_cnt, i) = PSL;
        ISL_mat(m_cnt, i) = ISL;

    end

    % Antenna Configuration
    d_cell{m_cnt} = d;
    d_total_cell{m_cnt} = d_total;
    M_iter(m_cnt) = iter;

end

%% Comparision

theta_t         = 30;
M_comp          = 16;

d_oraizi        = [-5.134 -4.272 -3.44 -2.616 -1.992 -1.436 -0.793 -0.252 ...
    0.252 0.793 1.436 1.992 2.616 3.44 4.272 5.134].' * lambda;
d_aslan         = [-4.53 -3.74 -2.73 -2.19 -1.56 -1.03 -0.53 -0.03 ...
    0.47 0.97 1.47 2.02 2.65 3.25 3.95 4.49].' * lambda;
d_ula           = (0:M_comp-1).' * lambda / 2;
d_proposed      = d_cell{M_vec == M_comp};

A_oraizi        = E_theta .* exp(delta .* d_oraizi);
A_proposed      = E_theta .* exp(delta .* d_proposed);
A_aslan         = E_theta .* exp(delta .* d_aslan);
A_ula           = E_theta .* exp(delta .* d_ula);

w_oraizi        = exp(1i*2*pi*d_oraizi/lambda.*sind(theta_t));
w_proposed      = exp(1i*2*pi*d_proposed/lambda.*sind(theta_t));
w_aslan         = exp(1i*2*pi*d_aslan/lambda.*sind(theta_t));
w_ula           = exp(1i*2*pi*d_ula/lambda.*sind(theta_t));

P_dB_oraizi     = 10*log10(abs((A_oraizi)' * w_oraizi).^2 ./ max(abs((A_oraizi)' * w_oraizi).^2));
P_dB_proposed   = 10*log10(abs((A_proposed)' * w_proposed).^2 ./ max(abs((A_proposed)' * w_proposed).^2));
P_dB_aslan      = 10*log10(abs((A_aslan)' * w_aslan).^2 ./ max(abs((A_aslan)' * w_aslan).^2));
P_dB_ula        = 10*log10(abs((A_ula)' * w_ula).^2 ./ max(abs((A_ula)' * w_ula).^2));

[PSL_oraizi, ISL_oraizi]        = sidelobelevel(P_dB_oraizi);
HPBW_idx                        = find(P_dB_oraizi >= -3);
HPBW_oraizi                     = diff([theta(HPBW_idx(1)); theta(HPBW_idx(end))]);

[PSL_proposed, ISL_proposed]    = sidelobelevel(P_dB_proposed);
HPBW_idx                        = find(P_dB_proposed >= -3);
HPBW_proposed                   = diff([theta(HPBW_idx(1)); theta(HPBW_idx(end))]);

[PSL_ula, ISL_ula]              = sidelobelevel(P_dB_ula);
HPBW_idx                        = find(P_dB_ula >= -3);
HPBW_ula                        = diff([theta(HPBW_idx(1)); theta(HPBW_idx(end))]);

[PSL_aslan, ISL_aslan]          = sidelobelevel(P_dB_aslan);
HPBW_idx                        = find(P_dB_aslan >= -3);
HPBW_aslan                      = diff([theta(HPBW_idx(1)); theta(HPBW_idx(end))]);

SLL_proposed                    = findpeaks(P_dB_proposed);
SLL_aslan                       = findpeaks(P_dB_aslan);
SLL_oraizi                      = findpeaks(P_dB_oraizi);
SLL_ula                         = findpeaks(P_dB_ula);
SLL_proposed(SLL_proposed == 0) = [];
SLL_aslan(SLL_aslan == 0)       = [];
SLL_oraizi(SLL_oraizi == 0)     = [];
SLL_ula(SLL_ula == 0)           = [];
MSL_proposed                    = mean(SLL_proposed);
MSL_oraizi                      = mean(SLL_oraizi);
MSL_aslan                       = mean(SLL_aslan);
MSL_ula                         = mean(SLL_ula);

%%  Results

% - Plots
% Beampattern Response
PosVec = {'northwest', 'north', 'northeast'};
for m_cnt = 1:length(M_vec)

    diff_d = diff(d_total_cell{m_cnt}, 1, 2);
    d_conv = vecnorm(sqrt(sum(diff_d.^2, 1)), 2, 1);
    d_conv_mat(m_cnt,1:M_iter(m_cnt)) = d_conv;

    if m_cnt ~= 2
        figure('Name',['Beampattern Response ', '(M = ', num2str(M_vec(m_cnt)), ')'],'NumberTitle','off',...
            'Position', [0 0 floor(ScreenSize(3)/3.5) floor(ScreenSize(3)/3.5)])
        hold on
        movegui(PosVec{m_cnt})
        plot(theta, squeeze(P_dB_all(m_cnt,:,:)), 'LineWidth', 1.5)
        yline(max(PSL_mat(m_cnt,:)), 'LineWidth', 1.5, 'LineStyle', '--')
        plot(theta, 10*log10(E_theta.^2 / max(E_theta.^2)), 'LineWidth', 2, 'LineStyle', ':', 'color', 'magenta')
        xlabel('Angle (degree $^\circ$)','Interpreter','latex','FontSize',12')
        ylabel('Normalized Beampattern (dB)','Interpreter','latex','FontSize',12')
        xlim([min(theta) max(theta)])
        xticks(min(theta):30:max(theta))
        ylim([-35 0])
        grid minor
        box on
        text(-55, max(PSL_mat(m_cnt,:)) + 1.5, [num2str(sprintf('%.2f', max(PSL_mat(m_cnt,:)))), ' dB'], 'Interpreter', 'latex', 'FontSize', 12, 'HorizontalAlignment', 'center');
        legend({'$\theta_t = -30^{\circ}$', '$\theta_t = -20^{\circ}$', ...
            '$\theta_t = -10^{\circ}$', '$\theta_t = 0^{\circ}$', ...
            '$\theta_t = 10^{\circ}$', '$\theta_t = 20^{\circ}$', ...
            '$\theta_t = 30^{\circ}$', 'Worst PSL', '$e^2(\theta)$'}, ...
            'Interpreter', 'latex', 'FontSize', 12, 'Location', 'NorthOutside', ...
            'NumColumns', 3);
    end

end

% Convergence Curve
figure('Name','Convergence Cruve','NumberTitle','off',...
    'Position', [0 0 floor(ScreenSize(3)/3) floor(ScreenSize(3)/3.5)])
hold on
movegui(PosVec{2})
plot(0:iter_max-1, d_conv_mat.' ./ max(d_conv_mat, [], 2).', 'LineWidth', 1.5)
grid on
xlim([0 5*max(ceil(M_iter / 5))])
xlabel('Iteration (i)', 'Interpreter', 'Latex')
ylabel('Normalized $\Delta \mathbf{d}$','Interpreter', 'Latex')
legend('$M = 16\quad$', '$M = 24\quad$', '$M = 32$', 'Interpreter', 'Latex', 'FontSize', 12, 'NumColumns', 3, 'Location', 'NorthOutside')
box on

% Beampattern Comparison
figure('Name','Beampattern Comparison', 'NumberTitle','off',...
    'Position', [floor(ScreenSize(3)/2) - floor(ScreenSize(3)/6) floor(ScreenSize(4)/12) floor(ScreenSize(3)/3) floor(ScreenSize(3)/6)])
hold on
plot(theta,P_dB_proposed, 'LineWidth', 1.5)
plot(theta,P_dB_oraizi, 'LineWidth', 1.5)
plot(theta,P_dB_aslan, 'LineWidth', 1.5)
plot(theta,P_dB_ula, 'LineWidth', 1.5)
yline(PSL_proposed, 'color', '#0072BD','LineWidth', 1.5, 'LineStyle', '--')
yline(PSL_oraizi, 'color', '#D95319','LineWidth', 1.5, 'LineStyle', '--')
yline(PSL_aslan, 'color', '#EDB120','LineWidth', 1.5, 'LineStyle', '--')
yline(PSL_ula, 'color', '#7E2F8E','LineWidth', 1.5, 'LineStyle', '--')
xlabel('Angle (degree $^\circ$)','Interpreter','latex','FontSize',12')
ylabel('Normalized Beampattern (dB)','Interpreter','latex','FontSize',12')
xlim([min(theta) max(theta)])
xticks(min(theta):30:max(theta))
ylim([-35 0])
grid minor
legend('Proposed', 'ORAIZI', 'ASLAN', 'ULA', 'Interpreter', 'Latex', 'FontSize', 12,'NumColumns', 4, 'Location', 'NorthOutside')
box on

% - Numerical Results
colWidth = 12;
fprintf('\n');
fprintf('============================================================\n');
fprintf('            Worst PSL, ISL, and MSL Values                  \n');
fprintf('============================================================\n');

fprintf('%-20s %*.2f (M = 16)\n', 'Worst PSL (dB):', colWidth, max(PSL_mat(1,:)));
fprintf('%-20s %*.2f (M = 16)\n', 'Worst ISL (dB):', colWidth, max(ISL_mat(1,:)));
fprintf('%-20s %*.2f (M = 16)\n', 'Worst MSL (dB):', colWidth, MSL_proposed);

fprintf('------------------------------------------------------------\n');

fprintf('%-20s %*.2f (M = 32)\n', 'Worst PSL (dB):', colWidth, max(PSL_mat(3,:)));
fprintf('%-20s %*.2f (M = 32)\n', 'Worst ISL (dB):', colWidth, max(ISL_mat(3,:)));
fprintf('%-20s %*.2f (M = 32)\n', 'Worst MSL (dB):', colWidth, MSL_proposed);

fprintf('============================================================\n');
fprintf('                      Comparison Table                      \n');
fprintf('============================================================\n');


headers = {'Criteria', 'Proposed', 'ORAIZI', 'ASLAN', 'ULA'};

rowLabels = {'PSL (dB)', 'ISL (dB)', 'MSL (dB)', 'HBPW (°)'};
dataValues = [PSL_proposed, PSL_oraizi, PSL_aslan, PSL_ula;
              ISL_proposed, ISL_oraizi, ISL_aslan, ISL_ula;
              MSL_proposed, MSL_oraizi, MSL_aslan, MSL_ula;
              HPBW_proposed, HPBW_oraizi, HPBW_aslan, HPBW_ula];

fprintf('%-*s', colWidth, headers{1});
for j = 2:length(headers)
    fprintf('%-*s', colWidth, headers{j});
end
fprintf('\n');

fprintf('%s\n', repmat('-', 1, colWidth * length(headers)));

for i = 1:length(rowLabels)
    fprintf('%-*s', colWidth, rowLabels{i}); % Print row label
    for j = 1:size(dataValues, 2)
        fprintf('%-*.2f', colWidth, dataValues(i, j)); % Left-aligned numeric value
    end
    fprintf('\n');
end

fprintf('%s\n', repmat('=', 1, colWidth * length(headers)));

fprintf('\n');