function [num_it_tot, A1_right_end] = GMMNLSE_MPA_step(A0, dt, sim, nonlin_const, mode_info, omegas, D_pos, D_neg, hrw)
% GMMNLSE_MPA_step  Take one step according to the GMMNLSE
% A0 - initial field, (N, num_modes) matrix, in the frequency domain in W^1/2
% dt - time grid point spacing, in ps
%
% sim.f0 - center frequency, in THz
% sim.fr - Raman proportion
% sim.sw - 1 includes self-steepening, 0 does not
% sim.deltaZ - small step size, in m
% sim.M - parallel extent, 1 is no parallelization
% sim.n_tot_max - maximum number of iterations
% sim.n_tot_min - minimum number of iterations
% sim.tol - tolerance for convergence at each iteration
% sim.singe_yes - 1 = single, 0 = double
% sim.gpu_yes - 1 = GPU, 0 = CPU
% sim.SK_factor - SK = SK_factor * fiber.SR
%
% nonlin_const - n2*w0/c, in W^-1 m
% mode_info.SR - SR tensor, in m^-2
% mode_info.nonzero_midx1234s - required indices in total
% mode_info.nonzero_midx34s - required indices for partial Raman term
%
% omegas - angular frequencies in 1/ps, in the fft ordering
% D_pos - exp(Dz) for all modes and all small steps, with size (N, num_modes, M+1)
% D_neg - exp(-Dz) for all modes and all small steps, with size (N, num_modes, M+1)
% hrw - Raman response in the frequency domain
%
% Output:
% num_it_tot - iteration at the end of which convergence was reached
% A1_right_end - (N, num_modes) matrix with the field at the end of the step, for each mode, in the frequency domain

N = size(A0, 1); % number of time/freq points
num_modes = size(A0, 2);

% 1) Set initial values for psi
psi = repmat(A0, 1, 1, sim.M+1); % M copies of psi(w,z) = A(w,z), in the frequency domain

% Iterate to solve for psi
for n_it = 1:sim.n_tot_max
    % 2) Calculate A(w,z) at all z
    A_w = permute(psi.*D_pos, [1 3 2]); % (N, M+1, num_modes)

    % 3) Calculate A(t,z) at all z
    A_t = fft(A_w);

    % 4) Calculate U_p(t,z) = SK term, and V_pl(t,z) = SR term
    % If not using the GPU, we will precompute T_mn before the num_modes^4 sum
    
    % setup the matrices
    if sim.gpu_yes
        if sim.single_yes
            Up = complex(zeros(N, sim.M+1, num_modes, 'single', 'gpuArray'));
            Vpl = complex(zeros(N, sim.M+1, num_modes, num_modes, 'single', 'gpuArray'));
        else
            Up = complex(zeros(N, sim.M+1, num_modes, 'gpuArray'));
            Vpl = complex(zeros(N, sim.M+1, num_modes, num_modes, 'gpuArray'));
        end
    else
        if sim.single_yes
            T_mn = complex(zeros(N, sim.M+1, num_modes, num_modes, 'single'));
            Up = complex(zeros(N, sim.M+1, num_modes, 'single'));
            Vpl = complex(zeros(N, sim.M+1, num_modes, num_modes, 'single'));
        else
            T_mn = complex(zeros(N, sim.M+1, num_modes, num_modes));
            Up = complex(zeros(N, sim.M+1, num_modes));
            Vpl = complex(zeros(N, sim.M+1, num_modes, num_modes));
        end
    end
    
    if sim.gpu_yes
        % If using the GPU, do the computation with fast CUDA code
        [Up, Vpl] = feval(sim.kernel, Up, Vpl, complex(A_t), mode_info.SR, mode_info.nonzero_midx1234s,  N, sim.M+1, sim.SK_factor, size(mode_info.nonzero_midx1234s, 2), num_modes);
        Up = Up*(1-sim.fr);
        Vpl = Vpl*sim.fr;
    else
        % If using the CPU, first precompute T_mn
        for nz_idx = 1:size(mode_info.nonzero_midx34s, 2)
            midx3 = mode_info.nonzero_midx34s(1, nz_idx);
            midx4 = mode_info.nonzero_midx34s(2, nz_idx);
            T_mn(:, :, midx3, midx4) = sim.fr*A_t(:, :, midx3).*conj(A_t(:, :, midx4));
        end
        
        % Then calculate the num_modes^4 sum
        for nz_idx = 1:size(mode_info.nonzero_midx1234s, 2)
            midx1 = mode_info.nonzero_midx1234s(1, nz_idx);
            midx2 = mode_info.nonzero_midx1234s(2, nz_idx);
            midx3 = mode_info.nonzero_midx1234s(3, nz_idx);
            midx4 = mode_info.nonzero_midx1234s(4, nz_idx);

            Up(:, :, midx1) = Up(:, :, midx1) + sim.SK_factor*mode_info.SR(nz_idx)*(1-sim.fr)*A_t(:, :, midx2).*A_t(:, :, midx3).*conj(A_t(:, :, midx4));
            Vpl(:, :, midx1, midx2) = Vpl(:, :, midx1, midx2) + mode_info.SR(nz_idx)*T_mn(:, :, midx3, midx4);
        end
    end
    
    % 5,6) Apply the convolution for each part of the Raman sum
    Vpl = dt*fft(hrw.*ifft(Vpl));
   
    % 7) Finish the sum for the raman term, and add eveything together
    for midx1 = 1:num_modes
        for midx2 = 1:num_modes
            Up(:, :, midx1) = Up(:, :, midx1) + Vpl(:, :, midx1, midx2).*A_t(:, :, midx2);
        end
    end
    
    % 8) Take the fourier transform for each z, p
    Up = permute(ifft(Up), [1 3 2]); % (N, num_modes, M+1)

    % 9) Sum for each z, and save the intermediate results for the next iteration
    
    % Calculate the full integrand in frequency space
    Up = sim.deltaZ*1i*nonlin_const*(1+sim.sw*omegas/(2*pi*sim.f0)).*(Up.*D_neg);
    
    % Save the previous psi at the right end, then compute the new psis
    last_psi = psi(:, :, sim.M+1);
    last_zq = Up(:, :, 1)/2; % this is the R_0 term at first
    for q = 2:sim.M+1
        psi(:, :, q) = psi(:, :, 1) + last_zq + Up(:, :, q)/2; % The 1/2 is for the correct trap rule
        last_zq = last_zq + Up(:, :, q); % Then also add to the total
    end
    
    % Calculate the average NRMSE = take the RMSE between the previous psi
    % and the current psi at the right edge, normalize by the absolute max,
    % and average over all modes
    avg_NRMSE = 0;
    for midx = 1:num_modes
        diff_sq = abs(psi(:, midx, sim.M+1) - last_psi(:, midx)).^2;
        NRMSE_p = sqrt(sum(diff_sq(:))/N)/max(abs(psi(:, midx, sim.M+1)));
        if ~isnan(NRMSE_p)
            avg_NRMSE = avg_NRMSE + NRMSE_p;
        end
    end
    avg_NRMSE = avg_NRMSE/num_modes;
    
    if sim.verbose
        fprintf('iteration %d, avg NRMSE: %f\n', n_it, avg_NRMSE)
    end
    
    % If it has converged to within tol, then quit
    if (avg_NRMSE < sim.tol && n_it >= sim.n_tot_min)
        num_it_tot = n_it; % Save the number of iterations it took
        break
    end
    
    if n_it == sim.n_tot_max
        error('Error in GMMNLSE_MPA_step: The step did not converge after %d iterations, aborting.', sim.n_tot_max);
    end
    
    % 10) Psi has been updated at all z_j, so now repeat n_tot times
end

% 11) Get back to A from psi at the right edge
A1_w = psi.*D_pos;
A1_right_end = A1_w(:, :, sim.M+1);

end