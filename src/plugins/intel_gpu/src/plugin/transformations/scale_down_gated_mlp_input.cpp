// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scale_down_gated_mlp_input.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

namespace {
// `value` multiplied by `factor` and folded into a Constant, or nullptr if it does not fold or an FP16 element
// would leave the normal range (then the scaling would not be exact).
std::shared_ptr<ov::Node> fold_scaled(const ov::Output<ov::Node>& value, const std::shared_ptr<ov::Node>& factor) {
    std::shared_ptr<ov::op::v0::Constant> folded = ov::util::get_constant_from_source(std::make_shared<ov::op::v1::Multiply>(value, factor));
    if (!folded || folded->get_output_partial_shape(0) != value.get_partial_shape()) {
        return nullptr;
    }
    if (folded->get_element_type() == ov::element::f16) {
        constexpr float min_normal_f16 = 6.103515625e-05F;
        for (const float v : folded->cast_vector<float>()) {
            if (std::isinf(v) || (v != 0.F && std::abs(v) < min_normal_f16)) {
                return nullptr;
            }
        }
    }
    folded->set_friendly_name(value.get_node()->get_friendly_name() + "_gated_mlp_scaled");
    ov::copy_runtime_info(value.get_node_shared_ptr(), std::static_pointer_cast<ov::Node>(folded));
    return folded;
}

std::shared_ptr<ov::Node> scalar(const ov::element::Type& type, float value) {
    return ov::op::v0::Constant::create(type, ov::Shape{}, {value});
}

// Factor for a per-output-channel tensor of `shape`: `value` on channels [begin, end) of `n_axis`, 1 on the others.
std::shared_ptr<ov::Node> channel_factor(const ov::element::Type& type, const ov::PartialShape& shape, size_t n_axis, size_t begin, size_t end, float value) {
    if (!shape.is_static() || n_axis >= shape.size()) {
        return nullptr;
    }
    const auto n = static_cast<size_t>(shape[n_axis].get_length());
    if (begin >= end || end > n) {
        return nullptr;
    }
    ov::Shape factor_shape(shape.size(), 1);
    factor_shape[n_axis] = n;
    std::vector<float> values(n, 1.F);
    std::fill(values.begin() + static_cast<std::ptrdiff_t>(begin), values.begin() + static_cast<std::ptrdiff_t>(end), value);
    return ov::op::v0::Constant::create(type, factor_shape, values);
}

// Ops between the up projection and the product that a scalar factor passes through unchanged.
bool is_transparent(const ov::Node* node) {
    if (ov::is_type<ov::op::v1::Reshape>(node) || ov::is_type<ov::op::v0::Squeeze>(node) || ov::is_type<ov::op::v0::Unsqueeze>(node)) {
        return true;
    }
    return ov::is_type<ov::op::v0::Convert>(node) && node->get_input_element_type(0).is_real() && node->get_output_element_type(0).is_real();
}
}  // namespace

ScaleDownGatedMLPInput::ScaleDownGatedMLPInput(float scale_factor) {
    using namespace ov::pass::pattern;

    // FullyConnectedCompressed takes a varying number of inputs, so the gated product on input 0 is checked in the callback.
    auto down_m = wrap_type<op::FullyConnectedCompressed>();

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto down = ov::as_type_ptr<op::FullyConnectedCompressed>(m.get_match_root());
        if (!down || transformation_callback(down)) {
            return false;
        }

        const auto product = ov::as_type_ptr<ov::op::v1::Multiply>(down->get_input_node_shared_ptr(0));
        if (!product || product->get_output_element_type(0) != ov::element::f16 || product->get_output_target_inputs(0).size() != 1) {
            return false;
        }
        // The up branch is the operand that is not the activated gate.
        const bool gate_first = ov::is_type<ov::op::v4::Swish>(product->get_input_node_ptr(0));
        if (!gate_first && !ov::is_type<ov::op::v4::Swish>(product->get_input_node_ptr(1))) {
            return false;
        }

        // Every input that the factor 1/scale_factor goes to, with its factor: the constant of each Add on the up
        // branch, then the decompression scale and the bias of the up projection.
        std::vector<std::pair<ov::Input<ov::Node>, std::shared_ptr<ov::Node>>> targets;
        auto up = product->input_value(gate_first ? 1 : 0);
        while (true) {
            if (up.get_target_inputs().size() != 1) {
                return false;
            }
            auto* const node = up.get_node();
            if (is_transparent(node)) {
                up = node->input_value(0);
            } else if (ov::is_type<ov::op::v1::Add>(node)) {
                const bool const_0 = ov::util::get_constant_from_source(node->input_value(0)) != nullptr;
                const bool const_1 = ov::util::get_constant_from_source(node->input_value(1)) != nullptr;
                if (const_0 == const_1) {
                    return false;
                }
                const size_t c = const_1 ? 1 : 0;
                targets.emplace_back(node->input(c), scalar(node->get_input_element_type(c), 1.F / scale_factor));
                up = node->input_value(1 - c);
            } else {
                break;
            }
        }

        // The up projection is its own FC, or one output of the VariadicSplit after FullyConnectedHorizontalFusion merged
        // the gate/up pair; then only the up channels of the merged FC are scaled, and GLUFusion still matches.
        std::shared_ptr<op::FullyConnectedCompressed> up_fc;
        size_t begin = 0, end = 0;
        if (const auto split = ov::as_type_ptr<ov::op::v1::VariadicSplit>(up.get_node_shared_ptr())) {
            up_fc = ov::as_type_ptr<op::FullyConnectedCompressed>(split->get_input_node_shared_ptr(0));
            const auto axis = ov::util::get_constant_from_source(split->input_value(1));
            const auto lengths = ov::util::get_constant_from_source(split->input_value(2));
            if (!up_fc || up_fc->get_output_target_inputs(0).size() != 1 || !axis || !lengths) {
                return false;
            }
            const auto rank = up_fc->get_output_partial_shape(0).rank();
            const auto axis_value = axis->cast_vector<int64_t>()[0];
            const auto sizes = lengths->cast_vector<int64_t>();
            const bool negative_size = std::any_of(sizes.begin(), sizes.end(), [](int64_t s) {
                return s < 0;
            });
            if (rank.is_dynamic() || (axis_value != -1 && axis_value != rank.get_length() - 1) || negative_size) {
                return false;
            }
            begin = static_cast<size_t>(std::accumulate(sizes.begin(), sizes.begin() + static_cast<std::ptrdiff_t>(up.get_index()), int64_t{0}));
            end = begin + static_cast<size_t>(sizes[up.get_index()]);
        } else {
            up_fc = ov::as_type_ptr<op::FullyConnectedCompressed>(up.get_node_shared_ptr());
        }
        if (!up_fc) {
            return false;
        }
        // A merged FC needs per-channel factors; an FC of its own takes a scalar one.
        auto up_factor = [&](size_t input, size_t n_axis) -> std::shared_ptr<ov::Node> {
            const auto& type = up_fc->get_input_element_type(input);
            if (end == 0) {
                return scalar(type, 1.F / scale_factor);
            }
            return channel_factor(type, up_fc->get_input_partial_shape(input), n_axis, begin, end, 1.F / scale_factor);
        };
        const auto& scale_shape = up_fc->get_input_partial_shape(3);
        const size_t scale_n_axis = up_fc->get_transpose_b_scale() ? 0 : std::max<size_t>(scale_shape.size(), 1) - 1;
        targets.emplace_back(up_fc->input(3), up_factor(3, scale_n_axis));
        if (!ov::is_type<op::Placeholder>(up_fc->get_input_node_ptr(2))) {
            targets.emplace_back(up_fc->input(2), up_factor(2, std::max<size_t>(up_fc->get_input_partial_shape(2).size(), 1) - 1));
        }
        // The down projection takes the factor back, so its output and its bias stay as they are.
        targets.emplace_back(down->input(3), scalar(down->get_input_element_type(3), scale_factor));

        // Fold everything before changing the graph, so that a failed fold leaves the model intact.
        std::vector<std::shared_ptr<ov::Node>> folded;
        for (const auto& [input, factor] : targets) {
            folded.push_back(factor ? fold_scaled(input.get_source_output(), factor) : nullptr);
            if (!folded.back()) {
                return false;
            }
        }
        for (size_t i = 0; i < targets.size(); ++i) {
            targets[i].first.replace_source_output(folded[i]);
        }
        return true;
    };

    auto m = std::make_shared<Matcher>(down_m, "ScaleDownGatedMLPInput");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
