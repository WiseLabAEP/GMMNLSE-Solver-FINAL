function E_xyt = recompose_into_space(mode_space_profiles, dx, mode_time_profiles)
% recompse_into_space  Combine a set of mode time profiles with their
% corresponding space profiles to get the full 3D spatio-temperal field
% mode_space_profile - a (Nx, Nx, num_modes) matrix with each mode's profile in space. The units do not matter
% dx - spatial grid size, in m
% mode_time_profiles - a (Nt, num_modes) matrix with each mode's time profile in W^1/2.

num_modes = size(mode_space_profiles, 3);
Nx = size(mode_space_profiles, 1);
Nt = size(mode_time_profiles, 1);

E_xyt = zeros(Nx, Nx, Nt); % The full field to output

for ii = 1:num_modes
    mode_phi = mode_space_profiles(:, :, ii);
    N_ii = sqrt(sum(abs(mode_phi(:)).^2))*dx; % Normalization constant
    
    % Combine the time and space profiles at each time point to get the
    % full field
    for jj = 1:Nt
        E_xyt(:, :, jj) = E_xyt(:, :, jj) + mode_phi/N_ii*mode_time_profiles(jj, ii);
    end
end

end