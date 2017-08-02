function foutput = GMMNLSE_propagate(fiber, initial_condition, sim)
% GMMNLSE_MPA_propagate  Propagate an initial multi-mode pulse through an arbitrary distance of optical fiber
%
% fiber.betas - betas(i, :) = (i-1)th order dispersion coefficient for each mode, in ps^n/m
% fiber.SR - SR tensor, in m^-2
% fiber.L0 - length of fiber, in m
%
% initial_condition.dt - time step
% initial_condition.fields - initial field, in W^1/2, (Nxnum_modes). If the size is (Nxnum_modesxS), then it will take the last S
% 
% sim.f0 - center frequency, in THz
% sim.fr - Raman proportion
% sim.sw - 1 includes self-steepening, 0 does not
% sim.deltaZ - small step size, in m
% sim.M - parallel extent for MPA, 1 is no parallelization
% sim.n_tot_max - maximum number of iterations for MPA
% sim.n_tot_min - minimum number of iterations for MPA
% sim.tol - tolerance for convergence for MPA
% sim.save_period - spatial period between saves, in m. 0 = only save input and output
% sim.singe_yes - 1 = single, 0 = double
% sim.gpu_yes - 1 = GPU, 0 = CPU
% sim.mpa_yes - 1 = MPA, 0 = split step
% sim.SK_factor - SK = SK_factor * fiber.SR
% sim.use_const_mem - 1 = use the GPUs constant memory if possible, 0 = only use global memory
% sim.check_nan  - 1 = Check if the field has NaN components each roundtrip, 0 = do not
% sim.verbose - 1 = print extra information, 0 = stay silent
% sim.cuda_dir_path - path to the cuda directory into which ptx files will be compiled and stored
%
% Output:
% foutput.fields - (N, num_modes, num_save_points) matrix with the multimode field at each save point
% foutput.dt - time grid point spacing, to fully identify the field
% foutput.seconds - time spent in the main loop
% foutput.full_iterations_hist - histogram of the number of iterations, accumulated and saved between each save point

%% Check the validity of input parameters

if sim.save_period == 0
    sim.save_period = fiber.L0;
end

% MPA performs M large steps in parallel
if sim.mpa_yes
    large_step = sim.deltaZ*sim.M;
else
    large_step = sim.deltaZ; % for SS the large step and small step are the same
end
num_large_steps_total = fiber.L0/large_step;
num_large_steps_persave = sim.save_period/large_step;
num_saves_total = fiber.L0/sim.save_period;

if rem(num_large_steps_total+5e-11,1) > 1e-10
    error('Error in GMMNLSE_MPA_propagate: the large step size is %f m and the fiber length is %f m, which are not commensurate', large_step, fiber.L0)
end

if rem(num_large_steps_persave+5e-11,1) > 1e-10
    error('Error in GMMNLSE_MPA_propagate: the large step size is %f m and the save period is %f m, which are not commensurate', large_step, sim.save_period)
end

if rem(num_saves_total+5e-11,1) > 1e-10
    error('Error in GMMNLSE_MPA_propagate: the save period is %f m and the fiber length is %f m, which are not commensurate', sim.save_period, fiber.L0)
end

%% Work out the operlap tensor details
% These will be stored in the mode_info struct

% Get the numerical parameters from the initial condition
N = size(initial_condition.fields, 1);
num_modes = size(initial_condition.fields, 2);


% Find the permutations of SK and SR indices that need to be calculated
mode_info.SR = zeros(1, 1);

if sim.gpu_yes
    mode_info.SR = gpuArray(mode_info.SR);
end
if sim.single_yes
    mode_info.SR = single(mode_info.SR);
    sim.SK_factor = single(sim.SK_factor); % This is needed to exactly match the cuda argument type
end

% The indices of nonzero elements in SR will be stored in nonzero_midx1234s
% If not using the GPU, we also need to calculate the indices that don't
% have all zero coefficients for any given two indices
mode_info.nonzero_midx1234s = zeros(4, 1, 'uint8'); % Once again this is needed to match the argument type. If you want more than 256 modes, you will have to change this to uint16
mode_info.nonzero_midx34s = zeros(2, 1, 'uint8');
if sim.gpu_yes
    mode_info.nonzero_midx1234s = gpuArray(mode_info.nonzero_midx1234s);
    mode_info.nonzero_midx34s = gpuArray(mode_info.nonzero_midx34s);
end

% Fill out which elements are actually nonzero and store them
cntA = 1;
cntB = 1;
for midx1 = 1:num_modes
    for midx2 = 1:num_modes
        for midx3 = 1:num_modes
            for midx4 = 1:num_modes
                if fiber.SR(midx1, midx2, midx3, midx4) == 0
                    continue
                end
                mode_info.nonzero_midx1234s(:, cntA) = [midx1, midx2, midx3, midx4];
                mode_info.SR(cntA) = fiber.SR(midx1, midx2, midx3, midx4);
                cntA = cntA + 1;
            end
        end
        
        % Only avoid a set of two indices if all the other possible
        % combinations have 0 values in SR
        if fiber.SR(:, :, midx1, midx2) == 0
            continue
        end
        mode_info.nonzero_midx34s(:, cntB) = [midx1, midx2];
        cntB = cntB + 1;
    end
end

%% Set up the GPU details
if sim.gpu_yes
    gd = gpuDevice();
    
    if sim.single_yes
        num_size = 8; % 4 bytes * 2 for complex
        single_str = 'single';
    else
        num_size = 16; % 8 bytes * 2 for complex
        single_str = 'double';
    end
    
    specific_filename = ['calculate_sumterm_part_', single_str];
    cudaFilename = [specific_filename, '.cu'];
    ptxFilename = [specific_filename, '.ptx'];
    
    if ispc
        sep_char = '\';
    else
        sep_char = '/';
    end
    
    if exist([sim.cuda_dir_path sep_char ptxFilename], 'file') == 0
        system(['nvcc -ptx "', sim.cuda_dir_path sep_char cudaFilename, '" --output-file "', sim.cuda_dir_path sep_char ptxFilename '"']);
    end
    
    % Setup the kernel from the cu and ptx files
    kernel = parallel.gpu.CUDAKernel([sim.cuda_dir_path sep_char ptxFilename], [sim.cuda_dir_path sep_char cudaFilename]);
    
    % Break up the computation into threads, and group them into blocks
    num_threads_per_block = floor(gd.MaxShmemPerBlock/(num_modes*(2+num_modes)*num_size));
    if num_threads_per_block < 32
        num_threads_per_block = 32;
    end
    if num_threads_per_block > gd.MaxThreadBlockSize(1)
        num_threads_per_block = gd.MaxThreadBlockSize(1);
    end
    
    % The number of blocks is set based on the total number of threads
    if sim.mpa_yes
        tot_num_threads = N*(sim.M+1);
    else
        tot_num_threads = N;
    end
    
    num_blocks = ceil(tot_num_threads/num_threads_per_block);
    
    kernel.ThreadBlockSize = [num_threads_per_block,1,1];
    kernel.GridSize = [num_blocks,1,1];
    
    % Finally save the kernel
    sim.kernel = kernel;
end

%% Calculate the nonlinearity constant
c = 2.99792458e-4; % speed of ligth m/ps
w0 = 2*pi*sim.f0; % angular frequency (THz)
n2 = 2.3*10^-20; % m^2 W^-1
nonlin_const = n2*w0/c; % W^-1 m

%% Pre-calculate the dispersion term

omegas = 2*pi*ifftshift(linspace(-N/2, N/2-1, N))'/(N*initial_condition.dt); % in 1/ps, in the order that the fft gives
if sim.single_yes
    omegas = single(omegas);
end

% The dispersion term in the GMMNLSE, in frequency space
D_op = zeros(N, num_modes);

for mode_i = 1:num_modes
    D_op(:, mode_i) = 1i*(fiber.betas(1, mode_i) - real(fiber.betas(1, 1))) +1i*(fiber.betas(2, mode_i) - real(fiber.betas(2, 1)))*omegas;
    
    for jj = 2:size(fiber.betas, 1)-1
        D_op(:, mode_i) = D_op(:, mode_i) + (1i)*fiber.betas(jj+1, mode_i)/factorial(jj)*omegas.^jj;
    end
end

%% Pre-process as much as possible, depending on the type of simulation
if sim.gpu_yes
    omegas = gpuArray(omegas);
end

if sim.mpa_yes
    % Pre-compute exp(D_op*z) and exp(-D_op*z) for all z
    if sim.gpu_yes
        if sim.single_yes
            D_pos = zeros(N, num_modes, sim.M+1, 'single', 'gpuArray');
            D_neg = zeros(N, num_modes, sim.M+1, 'single', 'gpuArray');
        else
            D_pos = zeros(N, num_modes, sim.M+1, 'gpuArray');
            D_neg = zeros(N, num_modes, sim.M+1, 'gpuArray');
        end
    else
        if sim.single_yes
            D_pos = zeros(N, num_modes, sim.M+1, 'single');
            D_neg = zeros(N, num_modes, sim.M+1, 'single');
        else
            D_pos = zeros(N, num_modes, sim.M+1);
            D_neg = zeros(N, num_modes, sim.M+1);
        end
    end

    for zidx = 0:sim.M
        if sim.single_yes
            D_pos(:, :, zidx+1) = single(exp(D_op*sim.deltaZ*zidx));
            D_neg(:, :, zidx+1) = single(exp(-D_op*sim.deltaZ*zidx));
        else
            D_pos(:, :, zidx+1) = exp(D_op*sim.deltaZ*zidx);
            D_neg(:, :, zidx+1) = exp(-D_op*sim.deltaZ*zidx);
        end
    end
    
    % Take into account the z-step
    D = D_op*sim.deltaZ;
    if sim.gpu_yes
        D = gpuArray(D);
    end
    if sim.single_yes
        D = single(D);
    end
else
    % If we're not using MPA then we don't bother with pre-calculating exp(D*z)
    D = D_op*sim.deltaZ/2; % half here because of the symmetrized SS
    if sim.gpu_yes
        D = gpuArray(D);
    end
    if sim.single_yes
        D = single(D);
    end
end

%% Pre-compute the Raman response in frequency space
t1 = 12.2e-3; % raman parameter t1 [ps]
t2 = 32e-3; % raman parameter t2 [ps]
t_shifted = initial_condition.dt*(0:N-1)'; % time starting at 0
 
hr = ((t1^2+t2^2)/(t1*t2^2)).*exp(-t_shifted/t2).*sin(t_shifted/t1);

hrw = ifft(hr)*N; % The factor of N is needed because of how ifft is defined
if sim.gpu_yes
    hrw = gpuArray(hrw);
end
if sim.single_yes
    hrw = single(hrw);
end

%% Setup the exact save points

% We will always save the initial condition as well
z_points = round(num_large_steps_total)+1;
save_points = round(num_saves_total)+1;
save_freq = round(num_large_steps_total/num_saves_total); % Number of large steps per save

field_out = zeros(N, num_modes, save_points);
if sim.single_yes
    field_out = single(field_out);
end

% Start by saving the initial condition
field_out(:, :, 1) = initial_condition.fields(:, :, end);

% Also setup the last_result in the frequency domain
% This gets passed to the step function, so if using the GPU it also needs
% to live on the GPU
last_result = ifft(initial_condition.fields(:, :, end));
if sim.gpu_yes
    last_result = gpuArray(last_result);
end
if sim.single_yes
    last_result = single(last_result);
end

% Finally, setup a small matrix to track the number of iterations per step
if sim.mpa_yes
    iterations_hist = zeros(sim.n_tot_max, 1);
    full_iterations_hist = zeros(sim.n_tot_max, save_points-1);
end

%% Run the step function over each step
tic
for ii = 2:z_points
    % Print the step number if verbose
    if sim.verbose
        t_in_loop = tic;
        disp(ii)
    end
    
    % Run the correct step function depending on the options chosen
    if sim.mpa_yes
        [num_it, last_result] = GMMNLSE_MPA_step(last_result, initial_condition.dt, sim, nonlin_const, mode_info, omegas, D_pos, D_neg, hrw);
        iterations_hist(num_it) = iterations_hist(num_it)+1;
    else
        last_result = GMMNLSE_ss_step(last_result, initial_condition.dt, sim, nonlin_const, mode_info, omegas, D, hrw);
    end
    
    % Check for any NaN elements, if desired
    if sim.check_nan
        nanres = isnan(last_result);
        if sum(nanres(:)) > 0
            error('NaN field encountered, aborting');
        end
    end
    
    % If it's time to save, get the result from the GPU if needed,
    % transform to the time domain, and save it
    if rem(ii-1, save_freq) == 0
        field_out_ii = fft(last_result);
        if sim.gpu_yes
            field_out(:, :, round((ii-1)/save_freq)+1) = gather(field_out_ii);
        else
            field_out(:, :, round((ii-1)/save_freq)+1) = field_out_ii;
        end
        
        % If using MPA, also save and reset the iteration number histogram
        if sim.mpa_yes
            full_iterations_hist(:, round((ii-1)/save_freq)) = iterations_hist;
            iterations_hist = 0*iterations_hist;
        end
    end
    
    % Also print the time per step if verbose
    if sim.verbose
        toc(t_in_loop)
    end
end

% Just to get an accurate timing, wait before recording the time
if sim.gpu_yes
    wait(gd);
end
fulltime = toc();

%% Save the results in a struct

foutput.fields = field_out;
foutput.dt = initial_condition.dt;
foutput.seconds = fulltime;

if sim.mpa_yes
    foutput.full_iterations_hist = full_iterations_hist;
end

end