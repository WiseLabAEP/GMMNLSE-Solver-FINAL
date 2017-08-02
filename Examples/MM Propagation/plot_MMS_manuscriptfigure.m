sim.single_yes = 1;
sim.gpu_yes = 1;
sim.mpa_yes = 1;

load_name = make_test_save_name([pwd '\GRIN_1550_MMS1'], sim);
savepath = make_test_save_name([pwd '\plots\GRIN_1550_MMS'], sim)
load(load_name)
N = size(prop_output.fields, 1);
Zp = size(prop_output.fields, 3);
num_modes=size(prop_output.fields, 2);
t = (-N/2:N/2-1)*(prop_output.dt);
f = sim.f0+(-N/2:N/2-1)/(prop_output.dt*N); % THz
lambdas=(3e5)./f; %nm

zi=Zp;


filterc=(3e8)/(1690E-9)*(1E-12); %THz
filterbw= 8; %THz
GaussFilter=exp(-(f-filterc).^8/2/(filterbw).^8);
E_freqF=prop_output.fields(:, :, zi)*0;
for idx=1:num_modes
E_freqF(:,idx) = GaussFilter'.*squeeze(ifftshift(ifft(prop_output.fields(:, idx, zi)),1));
end
I_freq1F = abs(E_freqF).^2;
I_time1F = abs(fft(E_freqF)).^2;
energies1F=sum(I_time1F,1)*prop_output.dt*(1E-12);




load_name = make_test_save_name([pwd '\GRIN_1550_MMS_XPM1'], sim);
load(load_name)
N2 = size(prop_output.fields, 1);
Zp2 = size(prop_output.fields, 3);
num_modes2=size(prop_output.fields, 2);

zi=Zp2;

I_time2 = abs(prop_output.fields(:, :, zi).^2);
energies2=sum(I_time2,1)*prop_output.dt*(1E-12);
I_freq2 = abs(ifftshift(ifft(prop_output.fields(:, :, zi)),1)).^2;
t2 = (-N/2:N/2-1)*(prop_output.dt);
f2 = sim.f0+(-N/2:N/2-1)/(prop_output.dt*N); % THz
lambdas2=(3e5)./f; %nm


load_name = make_test_save_name([pwd '\GRIN_1550_MMS2'], sim);
load(load_name)
N = size(prop_output.fields, 1);
Zp = size(prop_output.fields, 3);
num_modes=size(prop_output.fields, 2);

zi=Zp;

I_time3 = abs(prop_output.fields(:, :, zi).^2);
energies3=sum(I_time3,1)*prop_output.dt*(1E-12);
I_freq3 = abs(ifftshift(ifft(prop_output.fields(:, :, zi)),1)).^2;




f1=figure('Position',[1 1 360 660])
subplot(3,2,1)
col=lines(num_modes);
trange = 1;
hold on
for idx=1:num_modes
plot(t, I_time1F(:,idx)/1000,'Color',col(idx,:))
end
hold off
[maxt,maxti]=max(I_time1F(:,1));

axis tight
xlim([t(maxti)-trange/2 t(maxti)+trange/2])
ylabel('Intensity (kW)')
xlabel('Time (ps)')
box on

subplot(3,2,2)
hold on
plot(lambdas,GaussFilter*num_modes,'k--')
%We will calculate the spectral center of mass for each mode
CMlambda=zeros(num_modes,1);
for idx=1:num_modes
CMlambda(idx)=sum(I_freq1F(:,idx).*lambdas',1)/sum(I_freq1F(:,idx),1);
plot(lambdas, I_freq1F(:,idx)/max(I_freq1F(:,idx))+(idx-1),'Color',col(idx,:))
text(1770, idx-0.5, [num2str(round(energies1F(idx)*1E9*100)/100) ' nJ'],'Color',col(idx,:))
end
plot(CMlambda,0.5:1:(num_modes-0.5),'k.-')

hold off
axis tight
ylabel('Intensity (a.u.)')
xlabel('Wavelength (nm)')
xlim([1600 1800])




subplot(3,2,3)
col=lines(num_modes);
trange=1;
hold on
for idx=1:num_modes
plot(t, I_time2(:,idx)/1000,'Color',col(idx,:))
end
hold off
[maxt,maxti]=max(I_time2(:,1));

axis tight
xlim([t(maxti)-trange/2 t(maxti)+trange/2])
ylabel('Intensity (kW)')
xlabel('Time (ps)')
box on
subplot(3,2,4)
hold on
%We will calculate the spectral center of mass for each mode
CMlambda=zeros(num_modes,1);
for idx=1:num_modes
CMlambda(idx)=sum(I_freq2(:,idx).*lambdas',1)/sum(I_freq2(:,idx),1);
plot(lambdas, I_freq2(:,idx)/max(I_freq2(:,idx))+(idx-1),'Color',col(idx,:))
text(1620, idx-0.5, [num2str(round(energies2(idx)*1E9*100)/100) ' nJ'],'Color',col(idx,:))
end
plot(CMlambda,0.5:1:(num_modes-0.5),'k.-')
hold off
axis tight
ylabel('Intensity (a.u.)')
xlabel('Wavelength (nm)')
xlim([1475 1650])



subplot(3,2,5)
col=lines(num_modes);
trange=20;
hold on
for idx=1:num_modes
plot(t, I_time3(:,idx)/1000,'Color',col(idx,:))
end
hold off
[maxt,maxti]=max(I_time3(:,1));

axis tight
xlim([t(maxti)-trange/2 t(maxti)+trange/2])
ylabel('Intensity (kW)')
xlabel('Time (ps)')
box on
subplot(3,2,6)
hold on
%We will calculate the spectral center of mass for each mode
CMlambda=zeros(num_modes,1);
for idx=1:num_modes
CMlambda(idx)=sum(I_freq3(:,idx).*lambdas',1)/sum(I_freq3(:,idx),1);
plot(lambdas, I_freq3(:,idx)/max(I_freq3(:,idx))+(idx-1),'Color',col(idx,:))
text(1585, idx-0.5, [num2str(round(energies3(idx)*1E9*100)/100) ' nJ'],'Color',col(idx,:))
end
plot(CMlambda,0.5:1:(num_modes-0.5),'k.-')
hold off
axis tight
ylabel('Intensity (a.u.)')
xlabel('Wavelength (nm)')
xlim([1525 1600])



saveas(f1,[savepath '_MSplot.fig'],'fig')
print(f1,[savepath '_MSplot.png'],'-dpng')
%close(f1)

