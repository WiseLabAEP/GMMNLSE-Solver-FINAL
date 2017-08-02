function A1 = GMMNLSE_ss_step(A0, dt, sim, nonlin_const, mode_info, omegas, D, hrw)
% GMMNLSE_ss_step  Take one step according to the GMMNLSE, using the split step method
% A0 - initial field, (N, num_modes) matrix, in the frequency domain in W^1/2
% dt - time grid point spacing, in ps
%
% sim.f0 - center frequency, in THz
% sim.fr - Raman proportion
% sim.sw - 1 includes self-steepening, 0 does not
% sim.deltaZ - step size, in m
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
% D - dispersion term for all modes in m^-1, with size (N, num_modes)
% hrw - Raman response in the frequency domain
%
% Output:
% A1 - (N, num_modes) matrix with the field after the step, for each mode, in the frequency domain

% 1) Propagate through the first dispersion section
A_t = fft(A0.*exp(D));

% 2) Propagate through the nonlinearity with RK4
prefactor = 1i*nonlin_const*(1+sim.sw*omegas/(2*pi*sim.f0));

k1 = sim.deltaZ*get_dA_dz(A_t, dt, sim, mode_info, hrw, prefactor);
k2 = sim.deltaZ*get_dA_dz(A_t+k1/2, dt, sim, mode_info, hrw, prefactor);
k3 = sim.deltaZ*get_dA_dz(A_t+k2/2, dt, sim, mode_info, hrw, prefactor);
k4 = sim.deltaZ*get_dA_dz(A_t+k3, dt, sim, mode_info, hrw, prefactor);

A_w_afternonlin = ifft(A_t + 1/6*(k1+2*k2+2*k3+k4));

% 3) propagate through the second dispersion section
A1 = A_w_afternonlin.*exp(D);

end

function dA_dz = get_dA_dz(A_t, dt, sim, mode_info, hrw, prefactor)
% get_dA_dz  Calculate dA_dz
% prefactor - prefactor for the nonlinearity

    N = size(A_t, 1); % number of time/freq points
    num_modes = size(A_t, 2);
    
    % Setup the matrices
    if sim.gpu_yes
        if sim.single_yes
            Up = complex(zeros(N, num_modes, 'single', 'gpuArray'));
            Vpl = complex(zeros(N, num_modes, num_modes, 'single', 'gpuArray'));
        else
            Up = complex(zeros(N, num_modes, 'gpuArray'));
            Vpl = complex(zeros(N, num_modes, num_modes, 'gpuArray'));
        end
    else
        if sim.single_yes
            T_mn = zeros(N, num_modes, num_modes, 'single');
            Up = zeros(N, num_modes, 'single');
            Vpl = zeros(N, num_modes, num_modes, 'single');
        else
            T_mn = zeros(N, num_modes, num_modes);
            Up = zeros(N, num_modes);
            Vpl = zeros(N, num_modes, num_modes);
        end
    end
    
    % Calculate the large num_modes^4 sum term
    if sim.gpu_yes
        % If using the GPU, do the computation with fast CUDA code
        [Up, Vpl] = feval(sim.kernel, Up, Vpl, complex(A_t), mode_info.SR, mode_info.nonzero_midx1234s,  N, 1, sim.SK_factor, size(mode_info.nonzero_midx1234s, 2), num_modes);
        Up = Up*(1-sim.fr);
        Vpl = Vpl*sim.fr;
    else
        % If using the CPU, first precompute T_mn
        for nz_idx = 1:size(mode_info.nonzero_midx34s, 2)
            midx3 = mode_info.nonzero_midx34s(1, nz_idx);
            midx4 = mode_info.nonzero_midx34s(2, nz_idx);
            T_mn(:, midx3, midx4) = sim.fr*A_t(:, midx3).*conj(A_t(:, midx4));
        end

        % Then calculate the num_modes^4 sum
        for nz_idx = 1:size(mode_info.nonzero_midx1234s, 2)
            midx1 = mode_info.nonzero_midx1234s(1, nz_idx);
            midx2 = mode_info.nonzero_midx1234s(2, nz_idx);
            midx3 = mode_info.nonzero_midx1234s(3, nz_idx);
            midx4 = mode_info.nonzero_midx1234s(4, nz_idx);

            Up(:, midx1) = Up(:, midx1) + sim.SK_factor*mode_info.SR(nz_idx)*(1-sim.fr)*A_t(:, midx2).*A_t(:, midx3).*conj(A_t(:, midx4));
            Vpl(:, midx1, midx2) = Vpl(:, midx1, midx2) + mode_info.SR(nz_idx)*T_mn(:, midx3, midx4);
        end
    end

    % Calculate h*Vpl as F-1(h F(Vpl))
    Vpl = dt*fft(hrw.*ifft(Vpl));

    % Finish the sum for the Raman term, and add eveything together
    for midx1 = 1:num_modes
        for midx2 = 1:num_modes
            Up(:, midx1) = Up(:, midx1) + Vpl(:, midx1, midx2).*A_t(:, midx2);
        end
    end
    
    % Now eveyerthing has been summed into Up, so transform into the
    % frequency domain for the prefactor, then back into the time domain
    dA_dz = fft(prefactor.*ifft(Up));
end
