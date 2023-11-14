import torch.nn as nn

def kaiming_init(
    module          :   nn.Module,
    a               :   float = 0.0,
    mode            :   str = 'fan_out',
    nonlinearity    :   str = 'relu',
    bias            :   float = 0.0,
    distribution    :   str = 'normal') -> None:
    r"""
    # Kaiming Weight Initializer

    Initializes modules parameters with the values according to the method descriped in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification - He, K. et al. (2015).

    Args:

    * `module (nn.Module)`:
        * Torch module to initialize.
    * `a (float)`:
        * Negative slope of the rectifier used after this layer. 
        * Default value is zero.
    * `mode (str)`:
        * Possible modes are **fan_in** and **fan_out**. Default is **fan_out**.
        * **fan_in** preserves the magnitude of the variance of the weights in the forward pass.
        * **fan_out** preserves the magnitudes in the backwards pass.
    * `nonlinearity (str)`:
        * Name of non-linear function (`nn.functional` name)
        * Default value is `relu`.
    * `bias (float)`:
        * Value to fill the bias. Default value is 0.
    * `distribution (float)`:
        * Distribution can be either **normal** or **uniform**.
        * Default value is **normal**.
    """
    assert distribution in ["uniform", "normal"]
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == "uniform":
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity
            )        
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity
            )
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(
    module       : nn.Module, 
    gain         : float=1., 
    bias         : float=0., 
    distribution : str="normal") -> None:
    r"""
    # Xavier Weight Initializer

    Initializes modules parameters with the values according to the method descriped in `Understanding the difficulty of training deep feedforward
    neural networks - Glorot, X. & Bengio, Y. (2010).

    Args:

    * `module (nn.Module)`:
        * Torch module to initialize.
    * `gain (float)`:
        * Scaling factor. Default value is 1.
    * `bias (float)`:
        * Value to fill the bias. Default value is 0.
    * `distribution (float)`:
        * Distribution can be either **normal** or **uniform**.
        * Default value is **normal**.
    """
    assert distribution in ["uniform", "normal"]
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == "uniform":
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def normal_init(
    module  : nn.Module, 
    mean    : float=0., 
    std     : float=1., 
    bias    : float=0.) -> None:
    r"""
    # Normal Weight Initializer

    Initializes modules parameters with the values from the normal distribution:
    $$\mathcal{N}(\text{mean}, \text{std}^2)$$.

    Args:

    * `mean (float)`:
        * Mean of the normal distribution. Default value is zero.
    * `std (float)`:
        * Standard deviation of the normal distribution. Default value is 1.
    * `bias (float)`:
        * Value to fill the bias. Default value is 0.
    """
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    r"""
    # Constant Weight Initializer

    Initializes modules parameters with constant values.

    Args:

    * `val (float)`:
        * Value to fill the weights in the module with.
    * `bias (float)`:
        * Value to fill the bias. Default value is 0.
    """
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)