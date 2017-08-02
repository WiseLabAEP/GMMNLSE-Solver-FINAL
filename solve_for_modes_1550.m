%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This script builds the fiber index profile and calls the svmodes function
% to solve for the lowest m modes over a range of frequencies.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Set parameters

Nf = 20; % number of frequency points at which the modes will be calculated
lambda0 = 1550e-9; % center wavelength, in m
lrange = 180e-9; % wavelength range, in m. If 0 only the center wavelength will be used
num_modes = 30; % number of modes to compute
radius = 25; % outer radius of fiber, in um
folder_name = 'Fibers/GRIN_1550'; % folder where the output will be stored

Nx = 800; % number of spatial grid points
spatial_window = 200; % full spatial window size, in um
profile_function = @build_GRIN; % function that builds the fiber
extra_params.ncore_diff = 0.0137; % difference between the index at the center of the core, and the cladding
extra_params.alpha = 2.08; % Shape parameter

%% Calculate the modes

mkdir(folder_name);
if ispc
    sep_char = '/';
else
    sep_char = '\';
end
    
% Set the range in frequency space, which is more objective
c = 2.99792458e-4; % speed of ligth m/ps
if lrange == 0
    l = lambda0*10^6;
else
    f0 = c/lambda0; % center frequency in THz
    frange = c/lambda0^2*lrange;
    df = frange/Nf;
    f = f0 + (-Nf/2:Nf/2-1)*df
    l = c./f*10^6; % um
end

% At each wavelength, calculate the modes
for kk = 1:length(l)
    lambda = l(kk); % wavelength
    
    % Build the index profile. The funcation can be arbitrary, and can take
    % any extra parameters
    [epsilon, x, dx] = profile_function(lambda, Nx, spatial_window, radius, extra_params);
    guess = sqrt(epsilon(Nx/2, Nx/2));

    % Quickly show the index profile to make sure everything's working
    % correctly
    gg=figure;
    subplot(2,1,1)

    pcolor(x,x,epsilon.^0.5)

    colormap(gray)
    colormap(flipud(colormap))
    shading interp
    axis square

    subplot(2,1,2)
    plot(x,epsilon(:,Nx/2).^0.5)

    saveas(gg,[folder_name sep_char 'fiberprofile'],'fig');
    print(gg,[folder_name sep_char 'fiberprofile'],'-dpng');
    close (gg)

    % Actually do the calculation
    field = 'scalar'; % See svmodes for details
    boundary = '0000'; % See svmodes for details
    t_justsolve = tic();
    [phi1,neff1]=svmodes(lambda,guess,num_modes,dx,dx,epsilon,boundary,field);
    toc(t_justsolve);

    % Save each mode in a separate file
    for ii=1:num_modes
        gg=figure('Position',[1 1 1200 800]);

        phi = phi1(:,:,ii);
        neff = neff1(ii);

        pcolor(x,x,phi)
        shading interp
        axis square
        title(['n_{eff} = ' num2str(neff)]) 
        
        % Save the file with identifying information
        fname=[folder_name sep_char 'radius'  num2str(radius) 'boundary' boundary 'field' field  'mode' num2str(ii) 'wavelength' num2str(round(lambda*1000))];
        print(gg,fname,'-dpng')
        save(fname,'x','phi','epsilon','neff')
        close(gg)
    end
end