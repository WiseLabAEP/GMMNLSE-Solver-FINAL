function save_name = make_test_save_name(base, sim)

save_name = base;

if sim.single_yes
    save_name = [save_name '_single'];
else
    save_name = [save_name '_double'];
end

if sim.gpu_yes
    save_name = [save_name '_gpu'];
else
    save_name = [save_name '_cpu'];
end

if sim.mpa_yes
    save_name = [save_name '_mpa'];
else
    save_name = [save_name '_ss'];
end

save_name = [save_name '.mat'];

end