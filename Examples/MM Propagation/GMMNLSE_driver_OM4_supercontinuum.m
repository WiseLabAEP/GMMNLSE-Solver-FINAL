% Example propagation in GRIN fiber, at a higher power which leads to
% supercontinuum generation. This example will not be as fully commented as
% the low power version, but it demonstrates the rich dynamics possible in
% multimode fibers

%% Set up Matlab PATH's

sim.cuda_dir_path = '../../cuda';
addpath('../../'); % MATLAB needs to know where the propagate files are located

%% Set up fiber parameters

num_modes = 6;
prefix = '../../Fibers/OM4_6';
load([prefix '/S_tensors_' num2str(num_modes) 'modes.mat']); % in m^-2
load([prefix '/betas.mat']); % in fs^n/mm

unit_conversion = 0.001.^(-1:size(betas, 1)-2)';
betas = betas.*unit_conversion;

fiber.betas = betas;
fiber.SR = SR;

fiber.L0 = 0.05;

%% Set up simulation parameters

c = 2.99792458e-4; %speed of ligth m/ps
lambda = 1030e-9; % m

sim.f0=c/lambda; % central pulse frequency (THz)
sim.fr = 0.18;
sim.sw = 1;
sim.M = 10;
sim.deltaZ = 5e-6;
sim.n_tot_max = 20;
sim.n_tot_min = 2;
sim.tol = 5*10^-4;
sim.save_period = 0; % Just set it to be the fiber length
sim.SK_factor = 1;
sim.check_nan = 1;
sim.verbose = 1;
if ~isfield(sim, 'defaults_set') || sim.defaults_set == 0
    sim.single_yes = 1;
    sim.gpu_yes = 1;
    sim.mpa_yes = 1;
end

save_name = make_test_save_name('OM4_supercontinuum', sim);

%% Set up initial conditions

N = 2^14;
tfwhm = 0.05; % ps
time_window = 20; %ps
total_energy = 410.3; %nJ

initial_condition = build_MMgaussian(tfwhm, time_window, total_energy, num_modes, N);

%% Run the propagation

reset(gpuDevice);
prop_output = GMMNLSE_propagate(fiber, initial_condition, sim);
save(save_name, 'prop_output', 'fiber', 'sim');
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