% This script compares the runtimes of the GMMNLSE_driver.m code using the
% MPA algorithm and the split-step algorithm, with GPU and with CPU

%% Run the propagation with MPA and ss, with GPU and CPU
% You can skip this and just run the next section if these have already run

sim.defaults_set = 1;
sim.single_yes = 1;

% First run with MPA and GPU
sim.gpu_yes = 1;
sim.mpa_yes = 1;
GMMNLSE_driver;

% Then run with MPA and CPU
sim.gpu_yes = 0;
sim.mpa_yes = 1;
GMMNLSE_driver;

% Then run with SS and GPU
sim.gpu_yes = 1;
sim.mpa_yes = 0;
GMMNLSE_driver;

% Then run with SS and CPU
sim.gpu_yes = 0;
sim.mpa_yes = 0;
GMMNLSE_driver;

%% Plot the speed comparison
% The execution time is plotted for MPA and SS methods, run on the GPU and
% CPU. The two MPA curves, for CPU and GPU, are shown as solid lines as a
% function of M (the MPA parallelization parameter. The two SS curves are
% shown as dashed lines as there is no M parameter in the split-step
% method. Note the difference in the axes as the GPU execution should be
% 50-100x faster than the CPU execution.

load('M1_M_evaluation_single_gpu_ss.mat', 'runtime_normalized');
ss_gpu_time = runtime_normalized;

load('M1_M_evaluation_single_cpu_ss.mat', 'runtime_normalized');
ss_cpu_time = runtime_normalized;

M_vals = 1:2:31;
mpa_gpu_times = zeros(size(M_vals));
mpa_cpu_times = zeros(size(M_vals));

for ii = 1:length(M_vals)
    load(['M' num2str(M_vals(ii)) '_M_evaluation_single_gpu_mpa.mat'], 'runtime_normalized');
    mpa_gpu_times(ii) = runtime_normalized;
    
    load(['M' num2str(M_vals(ii)) '_M_evaluation_single_cpu_mpa.mat'], 'runtime_normalized');
    mpa_cpu_times(ii) = runtime_normalized;
end

fs = 20;

figure('Position', [200, 200, 800, 600]);
box on
hold on;

yyaxis left
h = plot(M_vals, mpa_gpu_times, 'b');
set(h, 'LineWidth', 2);
h = plot([M_vals(1), M_vals(end)], [ss_gpu_time, ss_gpu_time], 'b--');
set(h, 'LineWidth', 2);
ylabel('time on GPU (s)');
ylim([0, ss_gpu_time*3/2])
ax = gca;
ax.YColor = 'blue';

yyaxis right
h = plot(M_vals, mpa_cpu_times, 'r');
set(h, 'LineWidth', 2);
h = plot([M_vals(1), M_vals(end)], [ss_cpu_time, ss_cpu_time], 'r--');
set(h, 'LineWidth', 2);
xlabel('M');
ylabel('time on CPU (s)');
ylim([0, ss_gpu_time*3/2*50])
ax = gca;
ax.YColor = 'red';

legend('MPA GPU', 'SS GPU', 'MPA CPU', 'SS CPU');
set(gca, 'FontSize', fs, 'LineWidth', 2);
title({'MPA acceleration vs M', '(Note different axes)'});
xlim([1, 31])