// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>

#include "op_translation_utils.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/decompositions/rms_norm.hpp"
#include "openvino/frontend/tensorflow_lite/decoder.hpp"
#include "openvino/frontend/tensorflow_lite/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/pass/node_registry.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

/// \brief Translate odml.rms_norm composite op.
/// Builds the canonical RMSNorm decomposition via ov::decomposition::rms_norm,
/// matched by ov::pass::RMSFusion during plugin compilation.
/// Expected inputs: [data, gamma]
/// Attributes: epsilon (float, default 1e-6)
OutputVector translate_odml_rms_norm(const ov::frontend::tensorflow_lite::NodeContext& node) {
    auto inputs = node.get_inputs();
    FRONT_END_GENERAL_CHECK(inputs.size() == 2,
                            "STABLEHLO_COMPOSITE odml.rms_norm expects 2 inputs (data, gamma), got ",
                            inputs.size());

    const auto& data = inputs[0];
    const auto& gamma = inputs[1];

    auto epsilon = node.get_attribute<float>("epsilon", 1e-6f);
    auto eps_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {epsilon});
    ov::Output<ov::Node> eps = eps_const;
    if (eps.get_element_type() != data.get_element_type())
        eps = std::make_shared<ov::op::v1::ConvertLike>(eps, data);

    // odml.rms_norm normalizes over the last dimension. RMSFusion fuses single
    // last-axis reductions only — see ov::decomposition::rms_norm docstring.
    auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {-1});

    ov::pass::NodeRegistry reg;
    auto out = ov::decomposition::rms_norm(reg, data, axes, eps, gamma);

    out.get_node()->set_friendly_name(node.get_decoder()->get_op_name());
    return ov::OutputVector{out};
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
    return {};  // unreachable; FRONT_END_OP_CONVERSION_CHECK throws
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
