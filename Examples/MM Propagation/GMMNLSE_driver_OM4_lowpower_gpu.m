% Example propagation in GRIN fiber using GPU, at a relatively low power.
% You can use this as a starting point for more specific simulations

%% Set up Matlab PATH's

sim.cuda_dir_path = '../../cuda'; % GMMNLSE_propagate needs to know where the cuda files are stored
addpath('../../'); % MATLAB needs to know where the propagate files are located

%% Set up fiber parameters

num_modes = 6; % The number of modes to be simulated. 1-~8 will be "very fast", 10-20 will get slower but it still works. More than 20 modes have not been tested

% We need to load in the overlap tenors, which dictate the nonlinear
% interaction between modes, and the beta vectors, which are the dispersion
% coefficients for each mode.
prefix = '../../Fibers/OM4_6';
load([prefix '/S_tensors_' num2str(num_modes) 'modes.mat']); % in m^-2
load([prefix '/betas.mat']); % in fs^n/mm

unit_conversion = 0.001.^(-1:size(betas, 1)-2)'; % The imported values are in fs^n/mm, but the simulation uses ps^n/m
betas = betas.*unit_conversion;

fiber.betas = betas;
fiber.SR = SR;

fiber.L0 = 0.15; % Fiber length in m

%% Set up simulation parameters

c = 2.99792458e-4; %speed of ligth m/ps
lambda = 1030e-9; % m, the center wavelength

sim.f0=c/lambda; % central pulse frequency (THz)
sim.fr = 0.18; % Raman proportial. 0.18 is standard for silica fibers. 0 turns off Raman completely
sim.sw = 1; % 1 includes the shock term, 0 excludes it.
sim.M = 10; % Parallelization extent for MPA. 1 = no parallelization, 5-20 is recommended; there are strongly diminishing returns after 5-10
sim.deltaZ = 25e-6; % The step size in m. This may need to be 1-50 um to account for intermodal beating, even if the nonlinear length is much larger than 50 um
sim.n_tot_max = 20; % Maximum number of iterations for MPA. This doesn't really matter because if the step size is too large the algorithm will diverge after a few iterations
sim.n_tot_min = 2; % Minimum number of iterations for MPA
sim.tol = 5*10^-4; % Value of the average NRMSE between consecutive itertaions in MPA at which the step is considered converged
sim.save_period = 0; % Length between saves, in m. Alternatively, 0 sets this to the fiber length. This must fit perfectly into the full propagation length
sim.SK_factor = 1; % Constant factor relating SK to SR, to account for linear or circular polarization 
sim.check_nan = 1; % 1 = check if the field has NaN values each step, 0 = do not
sim.verbose = 1; % Whether or not to print out information about the convergence and timing each step

% If defaults are set (for automation), these can be used instead to
% override the values here
if ~isfield(sim, 'defaults_set') || sim.defaults_set == 0
    sim.single_yes = 1; % Many GPUs are optimized for single precision, to the point where one can get a 10-30x speedup just by switching to single precision
    sim.gpu_yes = 1; % Whether or not to use the GPU. Using the GPU is HIGHLY recommended, as a speedup of 50-100x should be possible
    sim.mpa_yes = 1; % Whether to use the MPA algorithm or the split-step algorithm. If using a GPU the MPA algorithm will almost always be faster, but if using a CPU the split step algorithm will probably be faster
end

% The save name contains a specific base plus some information about the
% propagation
save_name = make_test_save_name('OM4_lowpower', sim);

%% Set up initial conditions

N = 2^15; % Number of time/frequency grid points. 2^9-2^20 all work, though 2^9 is fairly sparse and 2^20 is almost always overkill. A factor of 2 should ALWAYS be used to speed up the FFTs
tfwhm = 0.05; % ps, FWHM of the initial pulse.
time_window = 8; % ps, Full time extent of the time window. This should be large enough to contain the pulse, but it will also effect the frequency domain
total_energy = 17.04; % nJ, total energy of the initial pulse. By convension this is the total energy in all modes

% This is a helper function to build an evently distributed gaussian
% initial MM pulse
initial_condition = build_MMgaussian(tfwhm, time_window, total_energy, num_modes, N);

%% Run the propagation

reset(gpuDevice); % Resetting the GPU every time is a good idea
prop_output = GMMNLSE_propagate(fiber, initial_condition, sim); % This actually does the propagation

% The output of the propagation is a struct with:
% prop_output.fields = MM fields at each save point and the initial condition. The save points will be determined by sim.save_period, but the initial condition will always be saved as the first page.
% prop_output.dt = dt
% prop_output.seconds = total execution time in the main loop
% prop_output.full_iterations_hist (if using MPA) = a histogram of the
% number of iterations required for convergence

save(save_name, 'prop_output', 'fiber', 'sim'); % Also save the information about the propagation
disp(prop_output.seconds);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot the results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

N = size(prop_output.fields, 1); % Just in case we're loaded from a save

% Plot the histogram of iterations required for convergence
figure();
plot(prop_output.full_iterations_hist);
xlabel('Number of iterations');
ylabel('Counts');

%% Plot the time domain
figure();
I_time = abs(prop_output.fields(:, :, end).^2);
t = (-N/2:N/2-1)*(prop_output.dt);
tlim = 1;

subplot(1, 2, 1);
plot(t, I_time),axis tight, grid on
ylabel('Intensity (W)')
xlabel('Time (ps)')
xlim([-tlim, tlim])

%% Plot the frequency domain
I_freq = abs(ifftshift(ifft(prop_output.fields(:, :, end)))).^2;
f = sim.f0+(-N/2:N/2-1)/(prop_output.dt*N); % ps
flim = 20;

subplot(1, 2, 2);
plot(f, I_freq),axis tight, grid on
ylabel('Intensity (a.u.)')
xlabel('Frequency (THz)')
xlim([sim.f0-flim, sim.f0+flim])

%% Load the spatial modes and plot the full spatial field

% Load the modes
Nx = 800; % The number of spatial grid points that the modes use
mode_profiles = zeros(Nx, Nx, num_modes);
radius = '25'; % Used for loading the file
lambda0 = '1030'; % Used for loading the file
for ii = 1:num_modes
   name = [prefix, '/radius', radius, 'boundary0000fieldscalarmode',int2str(ii),'wavelength', lambda0, '.mat'];
   load(name, 'phi');
   mode_profiles(:, :, ii) = phi; % Save the modes
   disp(['Loaded mode ', int2str(ii)])
end
load(name, 'x');
x = (x-mean(x))*1e-6; % The spatial coordinates along one dimension
dx = x(2)-x(1);

% Downsample in space to reduce memory usage
factor = 4;
dx = dx*factor;
mode_profiles_sampled = zeros(Nx/factor, Nx/factor, num_modes);
for ii = 1:num_modes
    mode_profiles_sampled(:, :, ii) = downsample(downsample(mode_profiles(:, :, ii), factor)', factor)';
end
x = downsample(x, factor);
Nx = Nx/factor;
[X, Y] = meshgrid(x, x);

% Downsample in time to reduce memory usage
downsampled_fields = downsample(prop_output.fields(:, :, end), 4); % The full space-time field requires a huge amount of memory, so we downsample to avoid using too much memory

% Build the field from the modes and the spatial profiles
E_xyt = recompose_into_space(mode_profiles_sampled, dx, downsampled_fields);
A0 = sum(abs(E_xyt).^2, 3)*prop_output.dt/10^12; % Integrate over time to get the average spatial field

% Plot the spatial field
figure();
h = pcolor(X*1e6, Y*1e6, A0);
h.LineStyle = 'none';
colorbar;
axis square;
xlabel('x (um)');
ylabel('y (um)');
xlim([-60, 60]);
ylim([-60, 60]);