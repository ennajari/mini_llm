# helper functions for running training/inference from command line
def print_startup(cfg):
    print("Starting run with config:")
    for k,v in cfg.__dict__.items():
        print(f"  {k}: {v}")
