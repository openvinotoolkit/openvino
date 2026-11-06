// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/fp16_compression/keep_rope_angles_in_f32_for_models.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <vector>

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/convolution_base.hpp"
#include "openvino/op/util/read_value_base.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;

namespace {

// Upper bound on a RoPE angle sub-graph; anything larger is not one
constexpr size_t max_angle_subgraph_size = 512;
constexpr float pi_f32 = 3.14159265358979323846f;

// Constant behind `output`: a Constant, a decompression Convert(Constant) (excluded from constant folding by design),
// or a foldable sub-graph
std::shared_ptr<v0::Constant> as_constant(const ov::Output<ov::Node>& output) {
    const auto node = output.get_node_shared_ptr();
    if (const auto constant = ov::as_type_ptr<v0::Constant>(node))
        return constant;
    if (ov::is_type<v0::Convert>(node)) {
        if (const auto constant = ov::as_type_ptr<v0::Constant>(node->get_input_node_shared_ptr(0)))
            return constant;
    }
    return ov::util::get_constant_from_source(output);
}

// Single value of a constant input, if it has one
bool single_value(const ov::Output<ov::Node>& output, float& value) {
    const auto constant = as_constant(output);
    if (!constant || ov::shape_size(constant->get_shape()) != 1)
        return false;
    value = constant->cast_vector<float>()[0];
    return true;
}

bool is_close(float value, float expected) {
    return std::fabs(value - expected) <= 1e-3f * std::fabs(expected);
}

// Operand of a binary node that is not `known` (the second input if `known` is null and the first one is not constant)
ov::Output<ov::Node> other_operand(const std::shared_ptr<ov::Node>& node, const ov::Node* known) {
    if (known)
        return node->get_input_node_ptr(0) == known ? node->input_value(1) : node->input_value(0);
    return as_constant(node->input_value(0)) ? node->input_value(0) : node->input_value(1);
}

bool other_operand_value(const std::shared_ptr<ov::Node>& node, const ov::Node* known, float& value) {
    return single_value(other_operand(node, known), value);
}

// theta ^ linspace(0, 1, N) * pi / 2, as that sub-graph or constant-folded (first band: theta ^ 0 * pi / 2)
bool is_ltx2_frequencies(const ov::Output<ov::Node>& freqs) {
    if (const auto table = as_constant(freqs)) {
        return ov::shape_size(table->get_shape()) >= 2 && is_close(table->cast_vector<float>()[0], pi_f32 / 2.0f);
    }
    auto node = freqs.get_node_shared_ptr();
    if (ov::is_type<v0::Convert>(node))
        node = node->get_input_node_shared_ptr(0);
    float value = 0.0f;
    if (!ov::is_type<v1::Divide>(node) || !single_value(node->input_value(1), value) || !is_close(value, 2.0f))
        return false;
    const auto scaled = node->get_input_node_shared_ptr(0);
    if (!ov::is_type<v1::Multiply>(scaled))
        return false;
    const auto power_input = ov::is_type<v1::Power>(scaled->get_input_node_ptr(0)) ? 0 : 1;
    return ov::is_type<v1::Power>(scaled->get_input_node_ptr(power_input)) &&
           single_value(scaled->input_value(1 - power_input), value) && is_close(value, pi_f32);
}

// Collects the angle sub-graph feeding `angle` (inputs and constants only). Returns false if it depends on activations
bool collect_angle_subgraph(ov::Node* angle, std::vector<ov::Node*>& subgraph) {
    std::unordered_set<ov::Node*> seen{angle};
    std::vector<ov::Node*> stack{angle};
    while (!stack.empty()) {
        auto* node = stack.back();
        stack.pop_back();
        if (ov::is_type_any_of<v0::MatMul, ov::op::util::ConvolutionBase, ov::op::util::ReadValueBase>(node) ||
            subgraph.size() >= max_angle_subgraph_size)
            return false;
        if (ov::is_type_any_of<v0::Parameter, ov::op::util::ShapeOfBase>(node))
            continue;
        subgraph.push_back(node);
        for (const auto& input : node->input_values()) {
            if (seen.insert(input.get_node()).second)
                stack.push_back(input.get_node());
        }
    }
    return true;
}

// Marks the Sin/Cos consumers of `angle` and the angle sub-graph to keep them in f32 under f16 and bf16
bool keep_angle_chain_in_f32(const std::shared_ptr<ov::Node>& angle) {
    ov::NodeVector sin_cos;
    for (const auto& input : angle->output(0).get_target_inputs()) {
        auto* user = input.get_node();
        if (ov::is_type_any_of<v0::Sin, v0::Cos>(user))
            sin_cos.push_back(user->shared_from_this());
    }
    const bool has_sin = std::any_of(sin_cos.begin(), sin_cos.end(), [](const std::shared_ptr<ov::Node>& node) {
        return ov::is_type<v0::Sin>(node);
    });
    const bool has_cos = std::any_of(sin_cos.begin(), sin_cos.end(), [](const std::shared_ptr<ov::Node>& node) {
        return ov::is_type<v0::Cos>(node);
    });
    std::vector<ov::Node*> subgraph;
    if (!has_sin || !has_cos || !collect_angle_subgraph(angle.get(), subgraph))
        return false;
    for (const auto& node : sin_cos) {
        ov::disable_conversion(node, ov::element::f16);
        // Sin/Cos survive later fusions of the angle chain; the CPU plugin grows a continuous f32 region from them
        ov::disable_conversion(node, ov::element::f32, ov::element::bf16);
    }
    for (auto* node : subgraph)
        ov::disable_conversion(node->shared_from_this(), ov::element::f16);
    return true;
}

}  // namespace

namespace ov::pass {

KeepLTX2RopeAnglesInF32::KeepLTX2RopeAnglesInF32() {
    MATCHER_SCOPE(KeepLTX2RopeAnglesInF32);
    using namespace ov::pass::pattern;

    // (grid * 2 - 1) * freqs -> transpose(-1, -2) + flatten (a Reshape for a single position axis) -> Sin
    auto scaled = wrap_type<v1::Multiply>({any_input(), any_input()});
    auto centered = wrap_type<v1::Add>({scaled, any_input()});
    auto angles = wrap_type<v1::Multiply>({centered, any_input()});
    auto layout = wrap_type<v1::Transpose, v1::Reshape>({angles, any_input()});
    auto flat = optional<v1::Reshape>({layout, any_input()});
    auto sin = wrap_type<v0::Sin>({flat});

    matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pm = m.get_pattern_value_map();
        const auto scaled_node = pm.at(scaled).get_node_shared_ptr();
        const auto centered_node = pm.at(centered).get_node_shared_ptr();
        const auto angles_node = pm.at(angles).get_node_shared_ptr();
        float value = 0.0f;
        if (!other_operand_value(scaled_node, nullptr, value) || !is_close(value, 2.0f) ||
            !other_operand_value(centered_node, scaled_node.get(), value) || !is_close(value, -1.0f) ||
            !is_ltx2_frequencies(other_operand(angles_node, centered_node.get())))
            return false;
        keep_angle_chain_in_f32(pm.at(sin).get_node_shared_ptr()->get_input_node_shared_ptr(0));
        return false;
    };

    auto m = std::make_shared<Matcher>(sin, matcher_name);
    this->register_matcher(m, callback);
}

KeepRopeAnglesInF32ForModels::KeepRopeAnglesInF32ForModels(const std::vector<Model>& models) {
    for (const auto model : models) {
        switch (model) {
        case Model::LTX2:
            add_matcher<KeepLTX2RopeAnglesInF32>();
            break;
        }
    }
}

}  // namespace ov::pass
