function output = build_MMgaussian(tfwhm, time_window, total_energy, num_modes, N,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% build_MMgaussian - function that builds a multimode supergaussian
% temporal pulse using the following parameters:
%
% tfwhm - full width at half maximum of pulse, in ps
% time_window - width of entire time window, in ps
% total_energy - total energy of the pulse in all modes, in nJ
% num_modes - number of modes
% N - number of time grid points
% Optional:
% coeffs - the normalized complex amplitude coefficients of the different
% modes (default is equal across all modes)
% center - temporal position of the pulse in the time window (default is 0)
% gaussexpo - supergaussian exponent (~exp(-t^(2*gaussexpo))) (default is
% 1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Accept only 3 optional inputs at most
numvarargs = length(varargin);
if numvarargs > 3
    error('build_MMgaussian:TooManyInputs', ...
        'requires at most 3 optional inputs');
end

% Set defaults for optional inputs
coeffs=ones(num_modes,1);
coeffs=coeffs./sqrt(sum(abs(coeffs).^2));
optargs = {coeffs 0 1};

% Now put these defaults into the valuesToUse cell array, 
% and overwrite the ones specified in varargin.
optargs(1:numvarargs) = varargin;

% Place optional args in memorable variable names
[coeffs, center, gaussexpo] = optargs{:};

t0 = tfwhm/1.665; % ps
dt = time_window/N; % ps
t = (-N/2:N/2-1)*dt; % ps

gexpo=2*gaussexpo;

% Construct a single gaussian electric field envelope, in W^0.5
time_profile = sqrt(total_energy/(t0*sqrt(pi))*10^3)...
    *exp(-(t-center).^gexpo/(2*t0^gexpo));

% Apply this time profile to each mode using the coefficients
field = zeros(N, num_modes);
for idx = 1:num_modes
    field(:, idx) = coeffs(idx)*time_profile;
end

% Output as a struct
output.fields = field;
output.dt = dt;

end