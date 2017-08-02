% This script runs all the SMF simulations and two MM simulations in series

sim.single_yes = 1;
sim.gpu_yes = 1;
sim.mpa_yes = 1;
sim.defaults_set = 1;

cd 'SMF';
GMMNLSE_driver_SMF_GVD
GMMNLSE_driver_SMF_NL
GMMNLSE_driver_SMF_NLTOD
cd '../';

cd 'MM Propagation';
GMMNLSE_driver_OM4_lowpower_gpu
GMMNLSE_driver_OM4_supercontinuum
cd '../';