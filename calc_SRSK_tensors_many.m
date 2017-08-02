%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script calculates the mode overlap intergrals for the Kerr
% nonlinearity, as in "Multimode Nonlinear Fibre Optics: Theory and
% Applications," P. Horak and F. Poletti
%
% If using a GPU, the CUDA toolkit needs to be installed, which requires
% visual studio as well, and the executables for the C++ compiler and the
% CUDA compiler driver need to be in the path. See the user manual for more
% details
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Set parameters

modes_list=[1:15 28 45];
num_modes = length(modes_list); % number of modes for which the tensors should be calculated
Nx = 800; % number of spatial grid points for each mode
linear_yes = 1; % 1 = linear polarization, 0 = circular polarization
gpu_yes = 1; % 1 = run on GPU, 0 = run on CPU
single_yes = 1; % 1 = single precision, 0 = double precision
dir_prefix = 'Fibers/GRIN_1030'; % folder containing the calculated modes

% File name parameters, as strings:
lambda0 = '1030'; % center wavelength in nm
radius = '25';
boundary = '0000';

%% Load the modes

if ispc
    sep_char = '/';
else
    sep_char = '\';
end

fields = zeros(Nx, Nx, num_modes); % the spatial field of each mode
norms = zeros(num_modes, 1); % the normalization constant of each mode

if single_yes
    fields = single(fields);
    norms = single(norms);
end

% Load each mode, and calculate the normalization constant ahead of tiem
for ii = 1:num_modes
   name = [dir_prefix sep_char 'radius', radius, 'boundary' boundary 'fieldscalarmode',int2str(modes_list(ii)),'wavelength', lambda0, '.mat'];
   load(name, 'phi');
   fields(:, :, ii) = phi;
   norms(ii) = sqrt(sum(sum(abs(phi).^2)));
   disp(['Loaded mode ', int2str(ii)])
end

% Also load the spatial information to calculate Aeff accurately
load(name, 'x');
dx = (x(2)-x(1))*10^-6; % spatial step in m

%% Calculate the overlap integrals

% SR will hold the tensor. We only need to calculate SR, SK is always a
% constant times SR.
if single_yes
    SR = zeros(num_modes^4, 1, 'single');
else
    SR = zeros(num_modes^4, 1);
end

% If using the GPU, we need to do some things differently
if gpu_yes
    gd = gpuDevice();
    reset(gd); % It's always a good idea to reset the GPU before using it
    
    fields = permute(fields, [3 1 2]); % The order needs to go (num_modes, Nx, Nx)
    
    SR = gpuArray(SR);
    fields = gpuArray(fields);
    norms = gpuArray(norms);
    
    if single_yes
        specific_filename = 'cuda/calculate_tensors_single';
    else
        specific_filename = 'cuda/calculate_tensors_double';
    end
    
    % Compile the CUDA code. We don't need to do this every time, but it
    % only takes a few seconds so we might as well
    cudaFilename = [specific_filename, '.cu'];
    ptxFilename = [specific_filename, '.ptx'];
    system(['nvcc -ptx ', cudaFilename, ' --output-file ', ptxFilename]);
    kernel = parallel.gpu.CUDAKernel( ptxFilename, cudaFilename );
    
    % Setup the kernel parameters
    num_threads_per_block = gd.MaxThreadBlockSize(1); % use as many threads per block as possible
    num_blocks = ceil((num_modes^4)/num_threads_per_block); % use as many blocks as needed
    kernel.ThreadBlockSize = [num_threads_per_block,1,1];
    kernel.GridSize = [num_blocks,1,1];
    
    % Run the CUDA code
    SR = feval(kernel, SR, fields, norms, int32(num_modes), int32(Nx));
    SR = gather(SR);
else
    % If we're not using the GPU, then do all the calculations directly in
    % MATLAB
    SR = reshape(SR, [num_modes, num_modes, num_modes, num_modes]);
    
    for midx1 = 1:num_modes
        disp(['Starting midx1 = ', int2str(midx1)])
        for midx2 = 1:num_modes
            disp(['Starting midx2 = ', int2str(midx2)])
            for midx3 = 1:num_modes
                for midx4 = 1:num_modes
                    SR(midx1, midx2, midx3, midx4) = sum(sum(fields(:, :, midx1).*fields(:, :, midx2).*fields(:, :, midx3).*fields(:, :, midx4)))/ ...
                        (norms(midx1)*norms(midx2)*norms(midx3)*norms(midx4));
                end
            end
        end
    end
    
    SR = reshape(SR, [num_modes^4, 1]); % Reshape so it looks the same as it would if the GPU was used
end
% Give SR the correct dimensions
SR = SR/dx^2;

%% Eliminate the zero elements

thresholdzero=SR(1)/100000; % This is fairly arbitrary

cnt = 0;
for midx = 1:num_modes^4
    if abs(SR(midx)) < thresholdzero
        SR(midx) = 0; % Set it to exactly 0
    else
        cnt = cnt + 1;
    end
end
fprintf('Calculated %d nonzero entries in the S_R tensor\n', cnt);

%% Save to disk

% For linear polarization SK=SR, for circular polarization SK=2/3*SR
if linear_yes
    mult_factor = 1;
else
    mult_factor = 2/3;
end

SR = reshape(SR, [num_modes, num_modes, num_modes, num_modes]);
SK = mult_factor*SR;
Aeff = 1/SR(1, 1, 1, 1);

save([dir_prefix sep_char 'S_tensors_' num2str(num_modes) 'modes'], 'SK', 'SR', 'Aeff');