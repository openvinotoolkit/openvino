// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "builder/blocks/common.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace blocks {

std::string rms_norm(GraphEmitter& e,
                     const std::string& in,
                     const std::string& weight,
                     const std::string& out_prefix,
                     float eps) {
    e.add_weight(weight);
    auto norm = e.add_op("GGML_OP_RMS_NORM",
                         out_prefix + ".rms",
                         {in},
                         e.shape_of_tensor(in),
                         ov::element::f32,
                         0,
                         {{"eps", eps}});
    return e.add_op("GGML_OP_MUL", out_prefix, {norm, weight}, e.shape_of_tensor(in), ov::element::f32);
}

std::string scale(GraphEmitter& e, const std::string& x, float factor, const std::string& name) {
    return e.add_op("GGML_OP_SCALE",
                    name,
                    {x},
                    e.shape_of_tensor(x),
                    ov::element::f32,
                    0,
                    {{"scale", factor}, {"bias", 0.0f}});
}

std::string add_bias(GraphEmitter& e, const std::string& x, const std::string& bias_weight, const std::string& name) {
    e.add_named_weight(bias_weight);
    return e.add_op("GGML_OP_ADD", name, {x, bias_weight}, e.shape_of_tensor(x), ov::element::f32);
}

}  // namespace blocks
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
