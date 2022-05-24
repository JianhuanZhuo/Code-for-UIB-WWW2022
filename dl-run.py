import socket

from tools.config import load_specific_config
from trainer import wrap

if __name__ == '__main__':
    assert socket.gethostname() == 'dell-PowerEdge-T640'
    config_file = "configs/AIV/BPR.yaml"
    wrap(load_specific_config(config_file))
