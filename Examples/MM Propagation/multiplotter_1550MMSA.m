%This is an example of some more interesting plots 


for run=1:2

sim.single_yes = 1;
sim.gpu_yes = 1;
sim.mpa_yes = 1;

load_name = make_test_save_name([pwd '\GRIN_1550_MMS' num2str(run)], sim);
savepath = make_test_save_name([pwd '\plots\GRIN_1550_MMS' num2str(run)], sim);
prefix = '../../Fibers/GRIN_1550';

zplot=100; %This is the simulation output we want to plot


load(load_name)

%% Plot the results
N = size(prop_output.fields, 1);
Zp = size(prop_output.fields, 3);
num_modes=size(prop_output.fields, 2);

I_time = abs(prop_output.fields(:, :, zplot).^2);
energies(:,zi)=sum(I_time,1)*prop_output.dt*(1E-12);
I_freq = abs(ifftshift(ifft(prop_output.fields(:, :, zplot)),1)).^2;
t = (-N/2:N/2-1)*(prop_output.dt);
f = sim.f0+(-N/2:N/2-1)/(prop_output.dt*N); % THz
lambdas=(3e5)./f; %nm
fmin=0;%sim.f0-200;
fmax=max(f);%sim.f0+200;

%Wavelength range to plot
lmin=1300;
lmax=1900;

tmin=15;
tmax=20;
%%
%Spatiotemporal plots

% Load the spatial modes and plot the full spatial field

% Load the modes
Nx = 800; % The number of spatial grid points that the modes use
mode_profiles = zeros(Nx, Nx, num_modes);
radius = '25'; % Used for loading the file
lambda0 = '1550'; % Used for loading the file
mode_list=[1 2 3 4 5 6 15 28];
for ii = 1:num_modes
   name = [prefix, '/radius', radius, 'boundary0000fieldscalarmode',int2str(mode_list(ii)),'wavelength', lambda0, '.mat'];
   load(name, 'phi');
   mode_profiles(:, :, ii) = phi; % Save the modes
   disp(['Loaded mode ', int2str(ii)])
end
load(name, 'x');
x = (x-mean(x))*1e-6; % The spatial coordinates along one dimension
dx = x(2)-x(1);

% Downsample in space to reduce memory usage
factor = 4;
dx = dx*factor;
mode_profiles_sampled = zeros(Nx/factor, Nx/factor, num_modes);
for ii = 1:num_modes
    mode_profiles_sampled(:, :, ii) = downsample(downsample(mode_profiles(:, :, ii), factor)', factor)';
end
x = downsample(x, factor);
Nx = Nx/factor;
[X, Y] = meshgrid(x, x);

% Downsample in time to reduce memory usage
downsampled_fields = downsample(prop_output.fields(:, :, end), 4); % The full space-time field requires a huge amount of memory, so we downsample to avoid using too much memory

% Build the field from the modes and the spatial profiles
E_xyt = recompose_into_space(mode_profiles_sampled, dx, downsampled_fields);
A0 = sum(abs(E_xyt).^2, 3)*prop_output.dt/10^12; % Integrate over time to get the average spatial field

% Plot the spatial field
xy=figure();
h = pcolor(X*1e6, Y*1e6, A0);
h.LineStyle = 'none';
colorbar;
axis square;
xlabel('x (\mum)');
ylabel('y (\mum)');
xlim([-60, 60]);
ylim([-60, 60]);

saveas(xy,[savepath '_xy_plot.fig'],'fig')
print(xy,[savepath '_xy_plot.png'],'-dpng')
close(xy)

% Plot the spatial field
A0 = sum(abs(E_xyt).^2, 1)*dx; % Integrate over x
yt=figure();
tds=interp1(1:length(t),t,linspace(1,length(t),length(t)/factor));
h = pcolor(tds,x*1e6, squeeze(A0));
h.LineStyle = 'none';
colorbar;
axis square;
ylabel('y (\mum)');
xlabel('t (ps');
ylim([-60, 60]);
xlim([tmin, tmax]);

saveas(yt,[savepath '_yt_plot.fig'],'fig')
print(yt,[savepath '_yt_plot.png'],'-dpng')
close(yt)

A0 = sum(abs(E_xyt).^2, 2)*dx; % Integrate over y

% Plot the spatial field
tx=figure();
h = pcolor(tds,x*1e6, squeeze(A0));
h.LineStyle = 'none';
colorbar;
axis square;
ylabel('x (\mum)');
xlabel('t (ps');
ylim([-60, 60]);
xlim([tmin, tmax]);

saveas(tx,[savepath '_xt_plot.fig'],'fig')
print(tx,[savepath '_xt_plot.png'],'-dpng')
close(tx)

end