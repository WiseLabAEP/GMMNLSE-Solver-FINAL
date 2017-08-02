zplot=36; %This is the simulation output we want to plot

sim.single_yes = 1;
sim.gpu_yes = 1;
sim.mpa_yes = 1;

load_name = make_test_save_name([pwd '\GRIN_1030_SPMGRINA'], sim);
savepath = [pwd '\plots\'];
load(load_name)

%% Plot the results
N = size(prop_output.fields, 1);
Zp = size(prop_output.fields, 3);
num_modes=size(prop_output.fields, 2);

I_time = abs(prop_output.fields(:, :, zplot).^2);
energies=sum(I_time,1)*prop_output.dt*(1E-12);
I_freq = abs(ifftshift(ifft(prop_output.fields(:, :, zplot)),1)).^2;
t = (-N/2:N/2-1)*(prop_output.dt);
f = sim.f0+(-N/2:N/2-1)/(prop_output.dt*N); % THz
lambdas=(3e5)./f; %nm
fmin=0;%sim.f0-200;
fmax=max(f);%sim.f0+200;

%Wavelength range to plot
lmin=800;
lmax=1500;

tmin=-0.5;
tmax= 1;

filterc=(3e8)/(1325E-9)*(1E-12); %THz
filterbw= 15; %THz
GaussFilter=exp(-(f-filterc).^2/2/(filterbw).^2);
E_freqF=I_freq*0;
for idx=1:num_modes
E_freqF(:,idx) = GaussFilter'.*squeeze(ifftshift(ifft(prop_output.fields(:, idx, zplot)),1));
end
I_freqF = abs(E_freqF).^2;
I_timeF = abs(fft(E_freqF)).^2;
energiesF=sum(I_timeF,1)*prop_output.dt*(1E-12);

f1=figure('Position',[1 1 360 660]);
subplot(3,2,1)
col=lines(num_modes);
hold on
for idx=1:num_modes
plot(t, I_time(:,idx)/1000,'Color',col(idx,:))
end
hold off
axis tight
box on
xlim([tmin tmax])
ylabel('Intensity (kW)')
xlabel('Time (ps)')


subplot(3,2,2)
hold on
for idx=1:num_modes
plot(lambdas, I_freq(:,idx)/max(I_freq(:,idx))+(idx-1),'Color',col(idx,:))
text(1400, idx-0.5, [num2str(round(energies(idx)*1E9)) ' nJ'],'Color',col(idx,:))
end
hold off
axis tight
ylabel('Intensity (a.u.)')
xlabel('Wavelength (nm)')
xlim([lmin lmax])


zplot=28; %This is the simulation output we want to plot

sim.single_yes = 1;
sim.gpu_yes = 1;
sim.mpa_yes = 1;

load_name = make_test_save_name([pwd '\GRIN_1030_SPMSTEPSA'], sim);
load(load_name)

N = size(prop_output.fields, 1);
Zp = size(prop_output.fields, 3);
num_modes=size(prop_output.fields, 2);

I_time2 = abs(prop_output.fields(:, :, zplot).^2);
energies2=sum(I_time2,1)*prop_output.dt*(1E-12);
I_freq2 = abs(ifftshift(ifft(prop_output.fields(:, :, zplot)),1)).^2;
t2 = (-N/2:N/2-1)*(prop_output.dt);
f2 = sim.f0+(-N/2:N/2-1)/(prop_output.dt*N); % THz
lambdas2=(3e5)./f; %nm


%Wavelength range to plot
lmin=800;
lmax=1500;

tmin=-0.5;
tmax= 1;

filterc=(3e8)/(1305E-9)*(1E-12); %THz
filterbw= 10; %THz
GaussFilter=exp(-(f-filterc).^2/2/(filterbw).^2);
E_freqF=I_freq*0;
for idx=1:num_modes
E_freqF(:,idx) = GaussFilter'.*squeeze(ifftshift(ifft(prop_output.fields(:, idx, zplot)),1));
end
I_freqF2 = abs(E_freqF).^2;
I_timeF2 = abs(fft(E_freqF)).^2;
energiesF2=sum(I_timeF2,1)*prop_output.dt*(1E-12);

subplot(3,2,3)
col=lines(num_modes);
hold on
for idx=1:num_modes
plot(t2, I_time2(:,idx)/1000,'Color',col(idx,:))
end
hold off
axis tight
box on
xlim([tmin tmax])
ylabel('Intensity (kW)')
xlabel('Time (ps)')


subplot(3,2,4)
hold on
for idx=1:num_modes
plot(lambdas2, I_freq2(:,idx)/max(I_freq2(:,idx))+(idx-1),'Color',col(idx,:))
text(1400, idx-0.5, [num2str(round(energies2(idx)*1E9)) ' nJ'],'Color',col(idx,:))
end
hold off
axis tight
ylabel('Intensity (a.u.)')
xlabel('Wavelength (nm)')
xlim([lmin lmax])


num_modes2=10;
num_modes=8;

subplot(3,2,5)
hold on
col=lines(num_modes);

for idx=1:num_modes
plot(t, I_timeF(:,idx)/1000,'Color',col(idx,:))
end
col=lines(num_modes2);
for idx=1:num_modes2
plot(t2+0.3, I_timeF2(:,idx)/1000,'Color',col(idx,:),'LineStyle','--')
end
hold off

axis tight
xlim([-0.5 0.5])
box on
ylabel('Intensity (kW)')
xlabel('Time (ps)')

subplot(3,2,6)
hold on
plot(lambdas, I_freqF(:,1)/max(I_freqF(:,1))+(0),'Color','k')
text(1350, 1-0.5, ['GRIN:' num2str(round(energiesF(1)*1E9)) ' nJ'],'Color','k')

plot(lambdas2, I_freqF2(:,1)/max(I_freqF2(:,1))+(1),'Color','k','LineStyle','--')
text(1350, 2-0.5, ['STEP:' num2str(round(energiesF2(1)*1E9)) ' nJ'],'Color','k')

hold off
axis tight
ylabel('Intensity (a.u.)')
xlabel('Wavelength (nm)')
xlim([1150 1480])

saveas(f1,[savepath 'SPMGRINSTEP.fig'],'fig')
print(f1,[savepath 'SPMGRINSTEP.png'],'-dpng')
%close(f1)