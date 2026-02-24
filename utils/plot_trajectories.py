import matplotlib.pyplot as plt

def _trajectory_plot_all_dims(trajectories, times, no_traj_to_plot = 100):
    times_cpu = times.detach().cpu().numpy()
    trajectories_cpu = trajectories.detach().cpu().numpy()
    
    dims = trajectories_cpu.shape[-1]
    fig, axes = plt.subplots(1, dims, figsize=(8*dims, 5))
    axes = [axes] if dims == 1 else axes
    for i in range(no_traj_to_plot):
        for j in range(dims):
            axes[j].plot(times_cpu[i,:], trajectories_cpu[i,:,j])

    return fig