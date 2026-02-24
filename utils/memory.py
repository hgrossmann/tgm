import torch

def get_memory(data, times, idx_last_observation, memory_length, device = None):
    """
    Extracts memory slices ending at idx_last_observation (inclusive),
    starting from `idx_last_observation - memory_length + 1`, 
    with left-side padding using the first available value and 0s for time.
    Inputs
    data of shape [batchsize, seq_len, data_dim]
    times of shape [batchsize, seq_len]
    idx_last_observation [batchsize]
    Returns 
    x_mem of shape [batchsize, memory_length, data_dim]
    t_mem of shape [batchsize, memory_length, 1]
    """
    if device is None: device = data.device.type 
    batch_size, seq_len, _ = data.shape
    mem_start = torch.clamp(idx_last_observation - memory_length + 1, min=0)
    lengths = idx_last_observation - mem_start + 1  # actual available length for each row

    # Create index range for right-aligned filling
    index_range = torch.arange(memory_length, device=device).unsqueeze(0).expand(batch_size, -1)  # (B, M)

    # Compute padding offset: how many leftmost positions to fill with padding
    pad_left = memory_length - lengths  # (B,)
    gather_indices = mem_start.unsqueeze(1) + index_range - pad_left.unsqueeze(1)  # shift for right-align

    # Clamp indices to start index to use first available value for padding
    gather_indices = torch.maximum(gather_indices, mem_start.unsqueeze(1))  # no index before mem_start

    # Batch indexing
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(gather_indices)

    # Gather memory and times
    data_mem = data[batch_indices, gather_indices]
    time_mem = times[batch_indices, gather_indices]

    # Padding for time: zero for left-pad positions
    time_pad_mask = gather_indices == mem_start.unsqueeze(1)
    time_mem = torch.where(time_pad_mask & (index_range < pad_left.unsqueeze(1)), torch.zeros_like(time_mem), time_mem)

    return data_mem, time_mem.unsqueeze(2)


def get_memory_unif_sampling_deprecated(data, t_current, bridge_times, memory_length):
    """
    Extract memory during autoregressive trajectory generation, assuming uniform
    generation, i.e. identical timeframe, identical alignment of markov-bridges
    for all trajectories:
    data : [no_trajectories, no_timepoints, state_dim]
    t_current : [1]
    bridge_times : [no_markov_bridges + 1]
    memory_length : int

    Returns
    -------
    x_mem of shape [no_trajectories, memory_length, data_dim]
    t_mem of shape [no_trajectories, memory_length, 1]

    """
    mem_times = (bridge_times[bridge_times <= t_current])[-memory_length:]  
    mem_data = data[:,mem_times]
    # padd if necessary
    if mem_data.shape[1] < memory_length:
        pad_length = memory_length - mem_data.shape[1]
        padding = torch.ones(data.shape[0], pad_length, 1, device=data.device) * mem_data[:,:1,:]
        mem_data = torch.cat((padding, mem_data), dim=1)
        padding = torch.zeros(pad_length, device=data.device)
        mem_times = torch.cat((padding, mem_times),dim=0)
    # expand and reshape mem_times as required
    mem_times = mem_times.repeat(data.shape[0],1).unsqueeze(2)
    return mem_data, mem_times