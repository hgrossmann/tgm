from geomloss import SamplesLoss


def mmd_metric(traj_samples, traj_data):
    """
    Assume that traj_samples and traj_data share a common time grid,
    i.e. both of shape [*, T, D]. The first dimension may differ...
    
    Note that SamplesLoss expects point clouds of shape [N, D] 
    where N = number of data points, D = dimensionality of each point.
    """
    mmd = SamplesLoss("energy")
    N = traj_samples.shape[0]
    return mmd(traj_samples.reshape(N,-1), traj_data.reshape(N,-1)).item()

def sinkhorn_dist(traj_samples, traj_data):
    sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)
    N = traj_samples.shape[0]
    return sinkhorn(traj_samples.reshape(N,-1), traj_data.reshape(N,-1)).item()