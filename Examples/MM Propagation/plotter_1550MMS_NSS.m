

for run=1:1



sim.single_yes = 1;
sim.gpu_yes = 1;
sim.mpa_yes = 1;

load_name = make_test_save_name([pwd '\GRIN_1550_MMS_NSS' num2str(run)], sim);
savepath = make_test_save_name([pwd '\plots\GRIN_1550_MMS_NSS' num2str(run)], sim);



load(load_name)
%% Plot the results
N = size(prop_output.fields, 1);
Zp = size(prop_output.fields, 3);
num_modes=size(prop_output.fields, 2);

maxt=max(max(max(abs(prop_output.fields(:, :, :).^2))));
maxf=max(max(max(abs(ifftshift(ifft(prop_output.fields(:, :, :)))).^2)));

energies=zeros(num_modes,Zp);


for zi=1:1:Zp
    
    
   
    zplot=(zi-1)*fiber.L0/(Zp-1);

I_time = abs(prop_output.fields(:, :, zi).^2);
energies(:,zi)=sum(I_time,1)*prop_output.dt*(1E-12);
I_freq = abs(ifftshift(ifft(prop_output.fields(:, :, zi)),1)).^2;
t = (-N/2:N/2-1)*(prop_output.dt);
f = sim.f0+(-N/2:N/2-1)/(prop_output.dt*N); % THz
lambdas=(3e5)./f; %nm
fmin=0;%sim.f0-200;
fmax=max(f);%sim.f0+200;

%Wavelength range to plot
lmin=1300;
lmax=1900;


%We will create a moving time window to help follow the pulse
trange=5;


%longitudinal positions
z=linspace(0,fiber.L0,Zp);

f1=figure('Position',[1 1 1260 700]);
subplot(2,2,1)
col=lines(num_modes);
hold on
for idx=1:num_modes
plot(t, I_time(:,idx),'Color',col(idx,:))
end
hold off
[maxt,maxti]=max(I_time(:,1));

axis tight, grid on
xlim([t(maxti)-trange/2 t(maxti)+trange/2])
ylabel('Intensity (W)')
xlabel('Time (ps)')
legend('1', '2', '3', '4', '5', '6', '7', '8')

subplot(2,2,2)
hold on
for idx=1:num_modes
plot(lambdas, I_freq(:,idx)/max(I_freq(:,idx))+(idx-1),'Color',col(idx,:))
text(1800, idx-0.5, [num2str(round(energies(idx,zi)*1E9)) ' nJ'],'Color',col(idx,:))
end
hold off
axis tight, grid on
ylabel('Intensity (a.u.)')
xlabel('Wavelength (nm)')
xlim([lmin lmax])

subplot(2,2,3)
semilogy(t, I_time(:,1),'Color',col(1,:)),hold on
for idx=2:num_modes
semilogy(t, I_time(:,idx),'Color',col(idx,:))
end
hold off
axis tight, grid on
ylabel('Intensity (W)')
xlabel('Time (ps)')

subplot(2,2,4)
semilogy(f, I_freq(:,1),'Color',col(1,:)),hold on
for idx=2:num_modes
semilogy(f, I_freq(:,idx),'Color',col(idx,:))
end
hold off
axis tight, grid on
ylabel('Intensity (a.u.)')
xlabel('Frequency (THz)')
%ylim([maxf*1E-6 maxf])
%xlim([fmin fmax])

saveas(f1,[savepath '_z_' int2str(zi) '_all_plot.fig'],'fig')
print(f1,[savepath  '_z_' int2str(zi) '_all_plot.png'],'-dpng')
close(f1)

end

f2=figure()
plot(z,energies*(1E9))
legend('1','2','3','4','5','6','7','8')
xlabel('z (m)')
ylabel('Energy (nJ)')
saveas(f2,[savepath '_energy_plot.fig'],'fig')
print(f2,[savepath '_energy_plot.png'],'-dpng')
close(f2)

end