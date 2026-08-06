// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>

#include "op_translation_utils.hpp"
#include "openvino/core/model.hpp"
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

/// \brief Fallback that inlines the decomposition_subgraph_index subgraph.
/// Every STABLEHLO_COMPOSITE op carries a TFLite subgraph implementing the
/// composite's math equivalent. When no hand-written translator exists for the
/// composite_name, we materialize that subgraph as an ov::Model and splice it
/// into the parent graph: each Parameter is replaced by the corresponding
/// parent input, and each Result becomes one of this op's outputs.
/// Splicing pattern mirrors the pre-condition inlining in op/while.cpp.
OutputVector translate_decomposition_fallback(const ov::frontend::tensorflow_lite::NodeContext& node,
                                              const std::string& composite_name) {
    const auto decoder = node.get_decoder();
    auto idx_any = decoder->get_attribute("decomposition_subgraph_index");
    FRONT_END_OP_CONVERSION_CHECK(!idx_any.empty(),
                                  "STABLEHLO_COMPOSITE '",
                                  composite_name,
                                  "' has no decomposition_subgraph_index and no hand-written translator");
    auto decomp_idx = idx_any.as<int32_t>();

    auto decomp_model = node.get_subgraph(decomp_idx);
    FRONT_END_GENERAL_CHECK(decomp_model != nullptr,
                            "Failed to materialize decomposition subgraph for STABLEHLO_COMPOSITE '",
                            composite_name,
                            "'");

    auto params = decomp_model->get_parameters();
    auto results = decomp_model->get_results();
    FRONT_END_OP_CONVERSION_CHECK(params.size() == node.get_input_size(),
                                  "STABLEHLO_COMPOSITE '",
                                  composite_name,
                                  "' decomposition subgraph has ",
                                  params.size(),
                                  " parameters but the call site provides ",
                                  node.get_input_size(),
                                  " inputs");

    // Splice parent inputs into the decomposition by replacing each Parameter's
    // single output with the matching call-site input. The Parameter nodes
    // become unreferenced after this and drop out during graph cleanup.
    for (size_t i = 0; i < params.size(); ++i) {
        params[i]->output(0).replace(node.get_input(static_cast<int>(i)));
    }

    OutputVector outputs;
    outputs.reserve(results.size());
    for (const auto& result : results) {
        outputs.push_back(result->input_value(0));
    }
    return outputs;
}

/// \brief Dispatcher for STABLEHLO_COMPOSITE operations.
/// Routes to specific composite translators based on the composite name; if
/// none is registered, falls back to inlining the decomposition subgraph.
OutputVector stablehlo_composite(const ov::frontend::tensorflow_lite::NodeContext& node) {
    const auto decoder = node.get_decoder();
    FRONT_END_GENERAL_CHECK(decoder != nullptr, "Decoder is required for STABLEHLO_COMPOSITE translation");

    auto composite_name = decoder->get_attribute("composite_name").as<std::string>();

    // Dispatch to appropriate translator based on composite name
    if (composite_name == "odml.rms_norm") {
        return translate_odml_rms_norm(node);
    }

    // Unknown composite — every well-formed STABLEHLO_COMPOSITE has a
    // decomposition subgraph; inline it instead of failing the conversion.
    return translate_decomposition_fallback(node, composite_name);
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
