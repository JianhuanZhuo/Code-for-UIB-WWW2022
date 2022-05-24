from search import grid_search
import socket

if __name__ == '__main__':
    assert socket.gethostname() == 'dell-PowerEdge-T640'
    gpus = [0, 1, 3, 2] * 2 + [0, 3] * 3
    config_file = "configs/AIV/UmBPR.yaml"
    grid_search(gpus, config_file)
