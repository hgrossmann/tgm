from datetime import datetime
import os

def _output_path(cfg):
    save_dir = cfg.train.save_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = f"{cfg.data.dataset}_{cfg.model.model_name}_sig{cfg.model.sigma}_rho{cfg.model.rho}_lr{cfg.train.lr}_sub{cfg.data.T_sub}_{cfg.train.manual_seed}_{timestamp}"
    output_path = os.path.join(save_dir, short_id)
    os.makedirs(output_path, exist_ok=True)
    return output_path