% Runs a propagation over a range of values of M using the MPA algorithm,
% to demonstrate the effectiveness of the MPA algorithm and the optimal
% choice of M

sim.cuda_dir_path = '../../cuda';
addpath('../../'); % MATLAB needs to know where the propagate files are located

%% Setup fiber parameters
num_modes = 10;
prefix = '../../Fibers/GRIN10';
load([prefix '/S_tensors_' num2str(num_modes) 'modes.mat']); % in m^-2
load([prefix '/betas.mat']); % in fs^n/mm

unit_conversion = 0.001.^(-1:size(betas, 1)-2)';
betas = betas.*unit_conversion;

fiber.betas = betas;
fiber.SR = SR;

base_fiber_length = 0.01; % fiber.L0 will be set later

%% Setup simulation parameters
c = 2.99792458e-4; %speed of ligth m/ps
lambda = 1030e-9; % m

sim.f0=c/lambda; % central pulse frequency (THz)
sim.fr = 0.18;
sim.sw = 1;
sim.deltaZ = 25e-6;
sim.n_tot_max = 20;
sim.n_tot_min = 2;
sim.tol = 5*10^-4;
sim.save_period = 0; % Just set it to be the fiber length
sim.SK_factor = 1;
sim.check_nan = 1;
sim.verbose = 1;
sim.gain_model = 0;
if ~isfield(sim, 'defaults_set') || sim.defaults_set == 0
    sim.single_yes = 1;
    sim.gpu_yes = 1;
    sim.mpa_yes = 1;
end

save_name = make_test_save_name('M_evaluation', sim);

%% Setup initial conditions
N = 2^14;
tfwhm = 0.1; % ps
time_window = 8; %ps
total_energy = 20; %nJ

initial_condition = build_MMgaussian(tfwhm, time_window, total_energy, num_modes, N);

%% Run the propagation at a range of M values (if using MPA)

% If using the CPU, simulate 4 times less so it finishes in a reasonable
% amount of time
if ~sim.gpu_yes
    base_fiber_length = base_fiber_length/4;
end

if sim.mpa_yes
    M_vals = 1:2:31;
else
    M_vals = 1;
end
for ii = 1:length(M_vals)
    sim.M = M_vals(ii);
    
    % We have to be a little careful because M*deltaZ needs to fit into
    % fiber.L0. The solution will be to change the fiber length slightly,
    % and then account for the change at the end
    num_its = round(base_fiber_length/(sim.M*sim.deltaZ));
	fiber.L0 = num_its*sim.M*sim.deltaZ;
    
    reset(gpuDevice); % It's always a good idea to reset the GPU before a simulation
    prop_output = GMMNLSE_propagate(fiber, initial_condition, sim);
    runtime_normalized = prop_output.seconds*base_fiber_length/fiber.L0; % Normalize for the different lengths
    
    if ~sim.gpu_yes
        runtime_normalized = runtime_normalized*4; % We simulated 4 times less, so multiply by 4 to get the corresponding time
    end
    
    save(['M' num2str(sim.M) '_' save_name], 'prop_output', 'fiber', 'sim', 'runtime_normalized');
    disp(runtime_normalized);
end