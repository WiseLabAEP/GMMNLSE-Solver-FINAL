% This script tests the effect of group velocity dispersion by propagating
% a Gaussian pulse through a length (2*LD) of SMF
%
% Expected results: temporal broadening, peak power is reduced by a little
% more than a factor 2. For reference, see Figure 3.1 in Nonlinear Fiber
% Optics 3rd Edition, Agrawal

%% Set up Matlab PATH's

sim.cuda_dir_path = '../../cuda';
addpath('../../'); % MATLAB needs to know where the propagate files are located

%% Set up fiber parameters

num_modes = 1; % number of modes (this script is simplified from MM case)
Aeff = 4.6263e-11; % effective area of the SMF, [m^2]

fiber.betas = [0; 0; 24.1616/1000;]; % dispersion coefficients, [ps^n/m]

SR = ones(1, 1, 1, 1); % nonlinear coefficients, simplified from MM case
SR(1, 1, 1, 1) = 1/Aeff;
fiber.SR = SR;

%% Set up simulation parameters

c = 2.99792458e-4; % speed of light, [m/ps]
lambda = 1030e-9; % [m]

sim.f0=c/lambda; % central pulse frequency [THz]
sim.fr = 0; % fractional contribution of Raman nonlinearity:
    % 0.18 is standard for silica fibers, 0 turns off Raman completely
sim.sw = 0; % shock term: 1 includes the shock term, 0 excludes it.
sim.M = 10; % Parallelization extent for MPA. 
    % 1 = no parallelization, 5-20 is recommended; 
    % there are strongly diminishing returns after 5-10
sim.n_tot_max = 20; % Maximum number of iterations for MPA. 
    % This doesn't really matter because if the step size is too large 
    % the algorithm will diverge after a few iterations
sim.n_tot_min = 2; % Minimum number of iterations for MPA
sim.tol = 5*10^-4; % Value of the average NRMSE between consecutive 
    % itertaions in MPA at which the step is considered converged
sim.save_period = 0; % Length between saves, in [m]. 
    % Alternatively, 0 sets this to the fiber length. 
    % This must fit perfectly into the full propagation length
sim.SK_factor = 1; % Constant factor relating SK to SR, 
    % to account for linear or circular polarization 
sim.check_nan = 1; % 1 = check if the field has NaN values each step, 
    % 0 = do not
sim.verbose = 1; % Whether or not to print out information about 
    % the convergence and timing each step

% If defaults are set (for automation), these can be used instead to
    % override the values here    
if ~isfield(sim, 'defaults_set') || sim.defaults_set == 0
    sim.single_yes = 1; % Many GPUs are optimized for single precision, 
        % to the point where one can get a 10-30x speedup just by 
        % switching to single precision
    sim.gpu_yes = 1; % Whether or not to use the GPU. 
        % Using the GPU is HIGHLY recommended, as a speedup of 50-100x 
        % should be possible
    sim.mpa_yes = 1; % Whether to use the MPA algorithm or the 
        % split-step algorithm. If using a GPU the MPA algorithm will 
        % almost always be faster, but if using a CPU the split step 
        % algorithm will probably be faster
end

% The save name contains a specific base plus some information about the
    % propagation
save_name = make_test_save_name('SMF_GVD', sim);

%% Set up initial conditions

N = 2^15; % Number of time/frequency grid points. 
    % 2^9 is fairly sparse and 2^20 is almost always overkill. 
    % A factor of 2 should ALWAYS be used to speed up the FFTs
tfwhm = 0.1; % [ps]
time_window = 20; % Full time extent of the time window.
    % This should be large enough to contain the pulse, 
    % but it will also effect the frequency domain. [ps]
total_energy = 0.000001; % [nJ]

L_D = (tfwhm/1.665)^2/abs(fiber.betas(3)); % dispersion length in [m]
fiber.L0 = 2*L_D; % total length of fiber propagation [m]
sim.deltaZ = fiber.L0/100; % longitudinal step size, in [m].

initial_condition = build_MMgaussian(tfwhm, time_window, total_energy, num_modes, N);

%% Run the propagation

reset(gpuDevice);
prop_output = GMMNLSE_propagate(fiber, initial_condition, sim);
save(save_name, 'prop_output', 'fiber', 'sim');
disp(prop_output.seconds);

%% Plot the results

N = size(prop_output.fields, 1);
start_I = abs(prop_output.fields(:, :, 1)).^2;
end_I = abs(prop_output.fields(:, :, end)).^2;
t = (-N/2:N/2-1)*prop_output.dt; % ps
tlim = 1;

figure();
hold on
plot(t,end_I(:, 1)/max(start_I), 'k'),axis tight, grid on
plot(t,start_I(:, 1)/max(start_I), 'k--')
ylabel('Intensity (W)')
xlabel('Time (ps)')
xlim([-tlim, tlim])
hold off