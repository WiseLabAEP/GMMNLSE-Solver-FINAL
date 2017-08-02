function [epsilon, x, dx] = build_step(lambda, Nx, spatial_window, radius, extra_params)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% build_step - function that builds the refractive index profile of a step
% index fiber with the following input parameters:
%
% lambda - wavelength, in um
% Nx - total number of grid points in each spatial dimension (x and y)
% spatial_window - total length of grid in each dimension (x and y), in um
% radius - radius of the GRIN fiber, in um
%
% extra_params.ncore_diff - amount to add to the Sellmeier result to get n
% (index) at the center of the core
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Use the Sellmeier equation with the following coefficients to generate
% n(lambda) of silicon (nsi) at the chosen wavelength
a1=0.6961663;
a2=0.4079426;
a3=0.8974794;
b1= 0.0684043;
b2=0.1162414;
b3=9.896161;

nsi=(1+a1*(lambda.^2)./(lambda.^2 - b1^2)+a2*(lambda.^2)./(lambda.^2 - b2^2)+a3*(lambda.^2)./(lambda.^2 - b3^2)).^(0.5);

% The cladding (ncl) is taken as undoped silica, and the core is modified (nco)
nco = nsi + extra_params.ncore_diff; % core index
ncl = nsi; % cladding index

% Generate spatial grid
dx = spatial_window/Nx; % um
x = (-Nx/2:Nx/2-1)*dx;
[X, Y] = meshgrid(x, x);

% Step index profile
epsilon = ncl^2*ones(Nx, Nx);
epsilon(X.^2+Y.^2 < radius^2) = nco^2;

end