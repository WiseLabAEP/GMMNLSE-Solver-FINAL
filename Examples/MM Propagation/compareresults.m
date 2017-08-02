sim.single_yes = 1;
sim.gpu_yes = 1;
sim.mpa_yes = 1;
run=1'

load_name = make_test_save_name([pwd '\GRIN_1550_MMS_XPM' num2str(run)], sim);
load(load_name)
N = size(prop_output.fields, 1);
Zp = size(prop_output.fields, 3);
num_modes=size(prop_output.fields, 2);


zi=Zp;
     
I_time = abs(prop_output.fields(:, :, zi).^2);

I_freq = abs(ifftshift(ifft(prop_output.fields(:, :, zi)),1)).^2;
t = (-N/2:N/2-1)*(prop_output.dt);
f = sim.f0+(-N/2:N/2-1)/(prop_output.dt*N); % THz
lambdas=(3e5)./f; %nm

load_name = make_test_save_name([pwd '\GRIN_1550_MMS_XPMHR' num2str(run)], sim);
savepath = make_test_save_name([pwd '\plots\GRIN_1550_MMS_XPMHR' num2str(run)], sim);
load(load_name)
N2 = size(prop_output.fields, 1);
Zp2 = size(prop_output.fields, 3);
num_modes=size(prop_output.fields, 2);

zi=Zp2;
     
I_time2 = abs(prop_output.fields(:, :, zi).^2);
I_freq2 = abs(ifftshift(ifft(prop_output.fields(:, :, zi)),1)).^2;
t2 = (-N2/2:N2/2-1)*(prop_output.dt);
f2 = sim.f0+(-N2/2:N2/2-1)/(prop_output.dt*N2); % THz
lambdas2=(3e5)./f; %nm
fmin=0;%sim.f0-200;
fmax=max(f);%sim.f0+200;

%Wavelength range to plot
lmin=1300;
lmax=1900;








f1=figure('Position',[1 1 1260 700]);
col=lines(num_modes);
semilogy(t, I_time(:,1),'Color',col(1,:)),hold on
semilogy(t2, I_time2(:,1),'Color',col(2,:))
hold off

axis tight, grid on

ylabel('Intensity (W)')
xlabel('Time (ps)')
legend('NSS','NSSHR')



saveas(f1,[savepath '_compare.fig'],'fig')
print(f1,[savepath  '_compare.png'],'-dpng')
%close(f1)

f1=figure('Position',[1 1 1260 700]);
col=lines(num_modes);
semilogy(f, I_freq(:,1),'Color',col(1,:)),hold on
semilogy(f2, I_freq2(:,1),'Color',col(2,:))
hold off

axis tight, grid on

ylabel('Intensity (a. u.)')
xlabel('Frequency (THz)')
legend('NSS','NSSHR')



saveas(f1,[savepath '_compareF.fig'],'fig')
print(f1,[savepath  '_compareF.png'],'-dpng')
%close(f1)