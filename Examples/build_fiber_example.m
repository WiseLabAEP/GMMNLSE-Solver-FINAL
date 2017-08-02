% This is an example demonstrating the steps that need to be taken to build
% a fiber from the parameters of an index profile. This file will just call
% the three scripts that actually do the calculations, and in general you
% should modify and run each of the three scripts for different fibers.
% The purpose of this script is simply to demonstrate the steps that need
% to be taken

% The fiber that gets built by default is a GRIN fiber that supports 6
% modes at 1030 nm. It is a good fiber to test propagation with as there
% are only a few modes.

cd ../; % The files exist in the root directory

%% 1. Define the fiber and solve for the modes
% You solve for the modes with solve_for_modes, after defining the fiber
% type and fiber parameters at the top of the script. The fiber type is
% determined by the function set to 'profile_function', which should take
% the wavelength, number of grid points, spatial window size, fiber radius,
% and any extra parameters needed for a given fiber type, and output the
% relative permittivity = n^2 as a function of position. The script solves
% for the modes over a range of wavelength, so the dispersion coefficients
% can be calculated later
solve_for_modes;

%% 2. Calculate the dispersion coefficients
% Once the modes and propagation constants have been solved for over a
% range of wavelength, the dispersion coefficients can be calculated. This
% happens in calc_dispersion. Similar parameters as solve_for_modes are set
% at the top of the file, and then the propagation constants are loaded in
% and the dispersion coefficients are calculate. A plot of the coefficients
% as a function of wavelength is also shown for each mode. The dispersion
% coefficients are saved in 'betas.mat' in the same folder that holds the
% modes.
calc_dispersion;

%% 3. Calculte the overlap tenors
% Finally, the intermode coupling through the nonlinearity is determined by
% a pair of overlap tenors, SK and SR, which are calculated in
% calc_SRSK_tenors. The code here supports linear and circular
% polarization, for which SR is the same and SK is a constant times SR. As
% this computation can require a significant amount of computation power it
% has been optimized to run on a GPU as well. Again these tensors are
% stored in 'S_tensors_6modes.mat' in the same folder that holds the modes.
calc_SRSK_tensors;

%% Use the new fiber!
% Now you can use the newly created fiber in simulations by loading the
% dispersion coefficients and the overlap tensors as is done in the
% examples.