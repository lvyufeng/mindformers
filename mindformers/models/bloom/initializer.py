from __future__ import absolute_import
import math

from mindspore._checkparam import Validator
from mindspore.common.initializer import _register, Initializer, _assignment, _init_random_normal
from mindspore.nn.transformer.layers import _args_type_validator_check



@_register("small_init")
class SmallNormal(Initializer):
    """Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2010), using a normal distribution.

        .. math::
        \sigma = \sqrt 5*dim
    """

    @_args_type_validator_check(hidden_size=Validator.check_positive_int)
    def __init__(self, hidden_size, sigma=0.02, mean=0.0):
        super(SmallNormal, self).__init__(sigma=sigma, mean=mean)
        self.sigma = math.sqrt(2 / (5 * hidden_size))
        self.mean = mean

    def _initialize(self, arr):
        data = _init_random_normal(self.mean, self.sigma, arr.shape)
        _assignment(arr, data)


@_register("wang_init")
class WangNormal(Initializer):
    """FFN Init from Ben Wang. 2021. Mesh-Transformer-JAX: Modelparallel implementation of transformer language model with JAX.

    """

    @_args_type_validator_check(hidden_size=Validator.check_positive_int,
                                n_layers=Validator.check_positive_int)
    def __init__(self, hidden_size, n_layers, sigma=0.02, mean=0.0):
        super(WangNormal, self).__init__(sigma=sigma, mean=mean)
        self.sigma = 2 / n_layers / math.sqrt(hidden_size)
        self.mean = mean

    def _initialize(self, arr):
        data = _init_random_normal(self.mean, self.sigma, arr.shape)
        _assignment(arr, data)
