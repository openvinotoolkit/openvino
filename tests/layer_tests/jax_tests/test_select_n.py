import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jax_layer_test_class import JaxLayerTest

rng = np.random.default_rng(5402)


class TestSelectN(JaxLayerTest):
    def _prepare_input(self):
        cases = []
        which = rng.uniform(0,self.input_shape, self.input_shape).astype(self.input_type)
        which = np.array(which)
        
        cases = []
        for i in range(self.input_shape):
            cases.append(jnp.array(rng.uniform(i*10, (i+1)*10, self.input_shape).astype(self.input_type)))
        cases = np.array(cases)
        return (which, cases)

    def create_model(self, input_shape, input_type):
        self.input_shape = input_shape
        self.input_type = input_type

        def jax_select_n(which, cases):
            return jax.lax.select_n(which, *cases)
        
        return jax_select_n, None, 'select_n'
        

    @pytest.mark.parametrize("input_shape", [1,2,3,4,5,6,7,8,9,10])
    @pytest.mark.parametrize("input_type", [np.int32, np.int64])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_jax_fe
    def test_select_n(self, ie_device, precision, ir_version, input_shape, input_type):
        self._test(*self.create_model(input_shape, input_type),
                   ie_device, precision,
                   ir_version)
