sim.single_yes = 1;
sim.gpu_yes = 1;
sim.mpa_yes = 1;

load_name = make_test_save_name([pwd '\GRIN_1550_MMS_XPMNSS1'], sim);
savepath = make_test_save_name([pwd '\plots\GRIN_1550_MMS_XPMNSS'], sim)
load(load_name)
N = size(prop_output.fields, 1);
Zp = size(prop_output.fields, 3);
num_modes=size(prop_output.fields, 2);

for zi=1:Zp




load_name = make_test_save_name([pwd '\GRIN_1550_MMS_XPMNSS1'], sim);
load(load_name)
I_time1 = abs(prop_output.fields(:, :, zi).^2);
energies1=sum(I_time1,1)*prop_output.dt*(1E-12);
I_freq1 = abs(ifftshift(ifft(prop_output.fields(:, :, zi)),1)).^2;
t = (-N/2:N/2-1)*(prop_output.dt);
f = sim.f0+(-N/2:N/2-1)/(prop_output.dt*N); % THz
lambdas=(3e5)./f; %nm
fmin=0;%sim.f0-200;
fmax=max(f);%sim.f0+200;

%Wavelength range to plot
lmin=1400;
lmax=1800;

load_name = make_test_save_name([pwd '\GRIN_1550_MMS_XPMNSS2'], sim);
load(load_name)
I_time2 = abs(prop_output.fields(:, :, zi).^2);
energies2=sum(I_time2,1)*prop_output.dt*(1E-12);
I_freq2 = abs(ifftshift(ifft(prop_output.fields(:, :, zi)),1)).^2;


trange=2;


f1=figure()
subplot(2,2,1)
col=lines(num_modes);
hold on
for idx=1:num_modes
plot(t, I_time1(:,idx)/1000,'Color',col(idx,:))
end
hold off
[maxt,maxti]=max(I_time1(:,1));

axis tight, grid on
xlim([t(maxti)-trange/2 t(maxti)+trange/2])
ylabel('Intensity (kW)')
xlabel('Time (ps)')


subplot(2,2,2)
hold on
%We will calculate the spectral center of mass for each mode
CMlambda=zeros(num_modes,1);
for idx=1:num_modes
CMlambda(idx)=sum(I_freq1(:,idx).*lambdas',1)/sum(I_freq1(:,idx),1);
plot(lambdas, I_freq1(:,idx)/max(I_freq1(:,idx))+(idx-1),'Color',col(idx,:))
text(1700, idx-0.5, [num2str(round(energies1(idx)*1E9)) ' nJ'],'Color',col(idx,:))
end
plot(CMlambda,0.5:1:(num_modes-0.5),'k.-')
hold off
axis tight, grid on
ylabel('Intensity (a.u.)')
xlabel('Wavelength (nm)')
xlim([lmin lmax])


trange=8;

subplot(2,2,3)
col=lines(num_modes);
hold on
for idx=1:num_modes
plot(t, I_time2(:,idx)/1000,'Color',col(idx,:))
end
hold off
[maxt,maxti]=max(I_time2(:,1));

axis tight, grid on
xlim([t(maxti)-trange/2 t(maxti)+trange/2])
ylabel('Intensity (kW)')
xlabel('Time (ps)')
subplot(2,2,4)
hold on
%We will calculate the spectral center of mass for each mode
CMlambda=zeros(num_modes,1);
for idx=1:num_modes
CMlambda(idx)=sum(I_freq2(:,idx).*lambdas',1)/sum(I_freq2(:,idx),1);
plot(lambdas, I_freq2(:,idx)/max(I_freq2(:,idx))+(idx-1),'Color',col(idx,:))
text(1700, idx-0.5, [num2str(round(energies2(idx)*1E9)) ' nJ'],'Color',col(idx,:))
end
plot(CMlambda,0.5:1:(num_modes-0.5),'k.-')
hold off
axis tight, grid on
ylabel('Intensity (a.u.)')
xlabel('Wavelength (nm)')
xlim([lmin lmax])

saveas(f1,[savepath '_CM_plot_' num2str(zi) '.fig'],'fig')
print(f1,[savepath '_CM_plot_' num2str(zi) '.png'],'-dpng')
close(f1)
end