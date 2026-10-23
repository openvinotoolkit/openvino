// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "widen_gather_matmul_weights.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <unordered_set>
#include <vector>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/reference/convert.hpp"
#include "openvino/runtime/tensor.hpp"
#include "ov_ops/gather_matmul.hpp"

namespace {

struct Widening {
    ov::element::Type_t from;
    ov::element::Type_t to;
};

// Narrow types GatherMatmul has no executor for, paired with the narrowest type it does support
// that represents them losslessly. u2 holds [0..3] and u4 a nibble, so the values survive.
constexpr std::array<Widening, 1> kWidenings{{{ov::element::u2, ov::element::u4}}};

// Re-emit `src` with the same values stored as `to`. element::iterator handles the sub-byte
// (un)packing, so this is just a value copy between two integer types.
// One explicit branch per kWidenings entry: the iterator element types are template arguments.
std::shared_ptr<ov::op::v0::Constant> widen_constant(const ov::op::v0::Constant& src, ov::element::Type to) {
    ov::Tensor dst(to, src.get_shape());
    const size_t count = ov::shape_size(src.get_shape());
    if (src.get_element_type() == ov::element::u2 && to == ov::element::u4) {
        ov::reference::convert(ov::element::iterator<ov::element::u2>(static_cast<const int8_t*>(src.get_data_ptr())),
                               ov::element::iterator<ov::element::u4>(static_cast<int8_t*>(dst.data())),
                               count);
    } else {
        OPENVINO_THROW("WidenGatherMatmulWeights: no widening implemented for ",
                       src.get_element_type(),
                       " -> ",
                       to);
    }
    return std::make_shared<ov::op::v0::Constant>(dst);
}

// Is `out` consumed, through the rest of a CompressedWeightsBlock, as a GatherMatmul's weight
// input? Traversal is restricted to the block's own op types -- all of which preserve the element
// type -- so it cannot wander into the surrounding model.
bool feeds_gather_matmul_weights(const ov::Output<ov::Node>& out) {
    constexpr size_t kWeightsInput = 1;
    std::vector<ov::Output<ov::Node>> stack{out};
    std::unordered_set<const ov::Node*> visited;
    while (!stack.empty()) {
        const auto cur = stack.back();
        stack.pop_back();
        for (const auto& target : cur.get_target_inputs()) {
            auto* node = target.get_node();
            if (ov::is_type<ov::op::internal::GatherMatmul>(node)) {
                if (target.get_index() == kWeightsInput) {
                    return true;
                }
                continue;
            }
            if (!ov::is_type_any_of<ov::op::v1::Subtract,
                                    ov::op::v1::Multiply,
                                    ov::op::v1::Reshape,
                                    ov::op::v1::Transpose,
                                    ov::op::v0::Convert>(node) ||
                !visited.insert(node).second) {
                continue;
            }
            for (const auto& o : node->outputs()) {
                stack.push_back(o);
            }
        }
    }
    return false;
}

}  // namespace

ov::intel_cpu::WidenGatherMatmulWeights::WidenGatherMatmulWeights(
    const std::vector<ov::element::Type>& supported_weights_types) {
    MATCHER_SCOPE(WidenGatherMatmulWeights);

    const auto supported = [&supported_weights_types](ov::element::Type t) {
        return std::find(supported_weights_types.begin(), supported_weights_types.end(), t) !=
               supported_weights_types.end();
    };

    // Only widenings this build actually needs: source unsupported, target supported.
    std::vector<Widening> widenings;
    for (const auto& w : kWidenings) {
        if (!supported(w.from) && supported(w.to)) {
            widenings.push_back(w);
        }
    }
    if (widenings.empty()) {
        return;
    }

    const auto is_widenable = [widenings](const ov::Output<ov::Node>& out) {
        return std::any_of(widenings.begin(), widenings.end(), [&](const Widening& w) {
            return out.get_element_type() == w.from;
        });
    };

    // Match the head of the dequantization subgraph (Constant -> Convert), not the whole
    // CompressedWeightsBlock: the block's shape between there and the GatherMatmul varies
    // (Reshape / Transpose / extra Convert), and only the Constant's storage type changes here.
    auto weights = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(is_widenable);
    auto convert = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({weights});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto weights_out = pattern_map.at(weights);
        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(weights_out.get_node_shared_ptr());
        if (!constant) {
            return false;
        }

        // Widening costs real bytes, so do it only where it buys the compressed path: a weight
        // input of a GatherMatmul. Walk forward from the Convert through the remaining
        // CompressedWeightsBlock shapes only (Subtract / Multiply / Reshape / Transpose / Convert,
        // all element-type-preserving), so the search stays inside the dequantization subgraph
        // instead of escaping into the rest of the model.
        if (!feeds_gather_matmul_weights(pattern_map.at(convert))) {
            return false;
        }

        ov::element::Type to;
        for (const auto& w : kWidenings) {
            if (constant->get_element_type() == w.from) {
                to = w.to;
                break;
            }
        }
        if (to == ov::element::dynamic) {
            return false;
        }

        auto widened = widen_constant(*constant, to);
        widened->set_friendly_name(constant->get_friendly_name());
        ov::copy_runtime_info(constant, widened);
        ov::replace_node(constant, widened);
        return true;
    };

    this->register_matcher(std::make_shared<ov::pass::pattern::Matcher>(convert, matcher_name), callback);
}
