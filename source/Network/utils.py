import torch
from time import time
import torch.nn as nn
from typing import Tuple
from source.Network.base_network import BaseNetwork


def get_network_parameter_number(model: nn.Module) -> int:
    r"""
    Calculates number of trainable parameter of a model.

    Args:

    * `model (nn.Module)`: 
        * The model for parameter number calculation.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def do_network_time_measurement_CPU(
    model: BaseNetwork, 
    input: Tuple[torch.Tensor]) -> None:
    r"""
    Performs time measurement for one module on the CPU.

    Args:

    * `model (BaseNetwork | nn.Module)`:
        * Model for which the time measurement is performed.
    * `input (Tuple(torch.Tensor))`:
        * Dummy input for the model.
    """
    iterations=None
    input = input.cpu()
    _ = model(input).cpu()
    with torch.no_grad():
        for _ in range(10):
            model(input)
    
        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)
    
        print('=========Speed Testing=========')
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    FPS = 1000 / latency
    print("{} iterations".format(iterations))
    print("{} fps".format(FPS))


def do_network_time_measurement_GPU(
    model: BaseNetwork, 
    input: Tuple[torch.Tensor]) -> None:
    r"""
    Performs time measurement for one module on the GPU.

    Args:

    * `model (BaseNetwork | nn.Module)`:
        * Model for which the time measurement is performed.
    * `input (Tuple(torch.Tensor))`:
        * Dummy input for the model.
    """
    iterations=None
    input = input.cuda()
    _ = model(input).cuda()
    with torch.no_grad():
        for _ in range(10):
            model(input)
    
        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)
    
        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print("{} iterations".format(iterations))
    print("{} fps".format(FPS))
