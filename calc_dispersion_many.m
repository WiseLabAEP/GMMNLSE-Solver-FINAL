%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script uses the propagation constants obtained from the mode
% calculations to approximate the dispersion parameters of each mode
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Set parameters

Nf = 20; % number of frequency/wavelength grid points 
lambda0_build = 1030e-9; % the center wavelength used to solve for the modes, in m
lambda0_disp = 1030e-9; % the center wavelength at which the dispersion parameters will be evaluated, in m
lrange = 180e-9; % wavelength range, in m. Should be the same as the range used to calculate the modes
modes_list=[1:15 28 45]; % modes selected for dispersion calculation (usually 1:num_modes, but can be different)

polynomial_fit_order = 7;
num_disp_orders = 4; % i.e. if this is 3, 4 coefficients will be calculated, including the 0th order

radius = 25; % needed to keep file names consistent
folder_name = 'Fibers/GRIN_1030'; % location to which files are saved

%% Load in the calculated effective n values

% filename and directory formatting based on whether machine is PC or not
if ispc
    sep_char = '/';
else
    sep_char = '\';
end
    
% Set the range in frequency space, which is more objective
c = 2.99792458e-4; % speed of ligth m/ps
if lrange == 0
    error('Cannot calculate dispersion with only one frequency point');
else
    f0 = c/lambda0_build; % center frequency in THz
    frange = c/lambda0_build^2*lrange;
    df = frange/Nf;
    f = (f0 + (-Nf/2:Nf/2-1)*df)'; % frequencies in THz, as a column vector
    l = c./f*10^6; % lambda in um
end


num_modes = length(modes_list);
field = 'scalar';
boundary = '0000';
lambda0_build = lambda0_build*10^6; % um

% Retrieve effective index values for each mode at each given wavelength
n_calc=zeros(Nf, num_modes);
for kk = 1:Nf
    lambda=l(kk);
    for ii=1:num_modes
        fname=[folder_name sep_char 'radius' num2str(round(radius)) 'boundary' boundary 'field' field  'mode' num2str(modes_list(ii)) 'wavelength' num2str(round(lambda*1000))];
        load([fname '.mat'])
        n_calc(kk, ii)=neff;
    end
    fprintf('Loading lambda = %d um\n', round(lambda*1000));
end


%% Calculate the propagation constants

beta_calc = zeros(Nf, num_modes);
w=2*pi*f; % angular frequencies in 1/ps
for midx=1:num_modes
    beta_calc(:, midx) = n_calc(:, midx).*w/c; % beta in 1/m
end


%% Fit the propagation constants to a polynomial and save the appropriate derivatives

dw = 2*pi*df;
w_disp = 2*pi*c/lambda0_disp; % angular frequency at which dispersion is calculated, in 1/ps

b_coefficients = zeros(num_modes, num_disp_orders+1); % The dispersion coefficients
for midx = 1:num_modes
    beta_calc_i = beta_calc(:, midx);
    
    beta_fit_last = polyfit(w, beta_calc_i, polynomial_fit_order); % the fit coefficients
    b_coefficients(midx, 1) = polyval(beta_fit_last, w_disp)/1000; % Save beta_0 in 1/mm
    for disp_order = 1:num_disp_orders
        % The derivatives can be calculated exactly from the coefficients
        beta_fit_last = ((polynomial_fit_order-(disp_order-1)):-1:1).*beta_fit_last(1:(polynomial_fit_order-(disp_order-1)));
        b_coefficients(midx, disp_order+1) = polyval(beta_fit_last, w_disp)*(10^3)^disp_order/1000; % beta_n in fs^n/mm
    end
end

% beta0 and beta1 should be relative to the fundamental mode.
b_coefficients(:, 1) = b_coefficients(:, 1) - ones(num_modes, 1)*b_coefficients(1, 1);
b_coefficients(:, 2) = b_coefficients(:, 2) - ones(num_modes, 1)*b_coefficients(1, 2);

betas = b_coefficients';
save([folder_name sep_char 'betas_many'], 'betas');


%% Display the results

% We need to use cell arrays because the higher orders are calculated from
% finite differences. This means that each order has one less data point
% than the previous one.
w_vectors = cell(num_disp_orders+1, 1); % omegas, in 1/ps
l_vectors = cell(num_disp_orders+1, 1); % lambdas, in um
w_vectors{1} = w;
l_vectors{1} = 2*pi*c./w_vectors{1}*10^6;
for disp_order = 1:num_disp_orders
    w_prev = w_vectors{disp_order};
    w_vectors{disp_order+1} = dw/2 + w_prev(1:length(w_prev)-1); % in 1/ps
    l_vectors{disp_order+1} = 2*pi*c./w_vectors{disp_order+1}*10^6; % in um
end

% beta_full will have all of the orders, for each mode, as a function of
% wavlength
beta_full = cell(num_disp_orders+1, 1);
beta_full{1} = beta_calc/1000;
for disp_order = 1:num_disp_orders
    beta_full{disp_order+1} = zeros(Nf-disp_order, num_modes);
end

% Take the differences to calculate the higher orders
for midx = 1:num_modes
    for disp_order = 1:num_disp_orders
        beta_full{disp_order+1}(:, midx) = diff(beta_full{disp_order}(:, midx))/dw*1000;
    end
end

ggg=figure;
coo=hsv(num_modes);

ylabels = cell(num_disp_orders+1, 1);
ylabels{1} = '1/mm';
ylabels{2} = 'fs/mm';
for disp_order = 2:num_disp_orders
    ylabels{disp_order+1} = ['fs^' num2str(disp_order) '/mm'];
end

% Plot the results
for disp_order = 1:num_disp_orders+1
    subplot(1,num_disp_orders+1,disp_order)
    hold on
    for midx = 1:num_modes
        plot(l_vectors{disp_order}, beta_full{disp_order}(:, midx), 'Color', coo(midx,:))
    end
    hold off
    ylabel(ylabels{disp_order})
    xlabel('\mum')
    axis tight
end