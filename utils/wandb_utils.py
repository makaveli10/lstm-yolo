import wandb


class Wandb:
    def __init__(self, opt, run_id=None, job_type='Training'):
        self.job_type = job_type
        self.wandb = wandb
        self.wandb_run = wandb.init(config=opt,
                                    resume="allow",
                                    project='LSTM-YOLO',
                                    entity=opt.entity,
                                    job_type=job_type,
                                    id=run_id,
                                    allow_val_change=True)
    
    def log(self, log_dict, step=None):
        self.wandb.log(log_dict, step=step)
