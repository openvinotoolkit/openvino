// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>

#include "op_translation_utils.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/frontend/tensorflow_lite/decoder.hpp"
#include "openvino/frontend/tensorflow_lite/node_context.hpp"
#include "ov_ops/rms.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

/// \brief Translate odml.rms_norm composite op.
/// The composite encodes RMS normalization with learnable gamma (scale).
/// Expected inputs: [data, gamma]
/// Attributes: epsilon value
OutputVector translate_odml_rms_norm(const ov::frontend::tensorflow_lite::NodeContext& node) {
    auto inputs = node.get_inputs();
    FRONT_END_GENERAL_CHECK(inputs.size() == 2,
                            "STABLEHLO_COMPOSITE odml.rms_norm expects 2 inputs (data, gamma), got ",
                            inputs.size());

    const auto& data = inputs[0];
    const auto& gamma = inputs[1];

    double epsilon = static_cast<double>(node.get_attribute<float>("epsilon", 1e-6f));

    auto rms = std::make_shared<ov::op::internal::RMS>(data, gamma, epsilon);
    return ov::OutputVector{rms->output(0)};
}

/// \brief Dispatcher for STABLEHLO_COMPOSITE operations.
/// Routes to specific composite translators based on the composite name.
OutputVector stablehlo_composite(const ov::frontend::tensorflow_lite::NodeContext& node) {
    const auto decoder = node.get_decoder();
    FRONT_END_GENERAL_CHECK(decoder != nullptr, "Decoder is required for STABLEHLO_COMPOSITE translation");

    auto composite_name = decoder->get_attribute("composite_name").as<std::string>();

    // Dispatch to appropriate translator based on composite name
    if (composite_name == "odml.rms_norm") {
        return translate_odml_rms_norm(node);
    }

    // Unsupported composite operation
    FRONT_END_OP_CONVERSION_CHECK(false,
                                  "Unsupported STABLEHLO_COMPOSITE operation: ",
                                  composite_name,
                                  ". Currently only odml.rms_norm is supported.");
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
