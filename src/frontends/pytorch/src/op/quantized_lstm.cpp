// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

/*
    aten::quantized_lstm.input(
        tensor input,
        tensor hx [],
        __torch__.torch.classes.rnn.CellParamsBase[] params,
        bool has_biases,
        int num_layers,
        float dropout,
        bool train,
        bool bidirectional,
        bool batch_first,
        *,
        ScalarType dtype=None,
        bool use_dynamic=False
    ) -> (tensor, tensor, tensor)
    above information was found according to pytorch export + unpacking done in testing phase
    summary:
    1. no int8 tensor payload is passed
       The operator doesn't pass quantized weight tensors.
       instead it passes CellParamsBase objects which are opaque
       runtime-packed containers.

    2. CellParamsBase cannot be unpacked
       these objects aren't simple tensor lists and cannot be extracted
       using current provided methods.

    3. generic_rnn fallback isn't feasible
       generic_rnn expects a list of real weight tensors, but since quantized_lstm 
       provides only runtime-packed containers,it is impossible to reuse the existing 
       LSTM conversion path.

    dynamic quantization is runtime-only by design
    PyTorch dynamic quantization intentionally defers packing
    and kernel selection until runtime execution.
    this makes graph export impossible without replicating
    PyTorch's backend packing logic.
    users should export quantized LSTM models using FX/PT2E static
    quantization which produces real quantized tensors that can be
    converted correctly.
*/

OutputVector translate_quantized_lstm(const NodeContext& context) {

    // quantized_lstm always has 11 inputs (including dtype + use_dynamic)
    num_inputs_check(context, 11, 11);

    // check runtime flag(found from pytorch eport variables)

    const auto use_dynamic = context.const_input<bool>(10);

    PYTORCH_OP_CONVERSION_CHECK(!use_dynamic,
        "Quantized LSTM is not supported. "
        "Use FX/PT2E static quantization export.");
    /*
    in some case if use dynamic has different value, then also we get runtime value 
    only so pass an early conversion failure
    */

    FRONT_END_OP_CONVERSION_CHECK(false,
        "Quantized LSTM is not supported. "
        "Use FX/PT2E static quantization export.");
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
