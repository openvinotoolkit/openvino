// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "collapse_unqdq.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "../logging.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/normalize_l2.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace opp = ov::pass::pattern;

namespace {

ov::Output<ov::Node> preserve_output_type(const ov::Output<ov::Node>& replacement,
                                          const ov::element::Type& expected_type,
                                          const std::string& friendly_name) {
    if (replacement.get_element_type() == expected_type) {
        return replacement;
    }

    auto convert = std::make_shared<ov::op::v0::Convert>(replacement, expected_type);
    convert->set_friendly_name(friendly_name);
    return convert;
}

constexpr float kF16Max = 65504.f;
// Calibrated ranges are not hard bounds, so keep one binade of headroom below the f16 maximum.
constexpr float kRescaleTarget = 32768.f;

std::shared_ptr<ov::Node> single_consumer(const std::shared_ptr<ov::Node>& node) {
    const auto targets = node->output(0).get_target_inputs();
    return targets.size() == 1 ? targets.begin()->get_node()->shared_from_this() : nullptr;
}

bool is_input_of(const std::shared_ptr<ov::Node>& node, std::size_t idx, const std::shared_ptr<ov::Node>& producer) {
    return node && node->get_input_node_ptr(idx) == producer.get();
}

// Returns the Multiply ending the FQ -> Convert -> Convert -> Subtract -> Multiply chain that
// CollapseUNQDQ replaces with the FQ input, or nullptr if `fq` does not start such a chain.
std::shared_ptr<ov::Node> unqdq_chain_end(const std::shared_ptr<ov::Node>& fq) {
    const auto cvt_q = single_consumer(fq);
    if (!ov::is_type<ov::op::v0::Convert>(cvt_q)) {
        return nullptr;
    }
    const auto cvt_dq = single_consumer(cvt_q);
    if (!ov::is_type<ov::op::v0::Convert>(cvt_dq)) {
        return nullptr;
    }
    const auto subtract = single_consumer(cvt_dq);
    if (!ov::is_type<ov::op::v1::Subtract>(subtract) || !is_input_of(subtract, 0, cvt_dq)) {
        return nullptr;
    }
    const auto multiply = single_consumer(subtract);
    if (!ov::is_type<ov::op::v1::Multiply>(multiply) || !is_input_of(multiply, 0, subtract)) {
        return nullptr;
    }
    return multiply;
}

// True if `multiply` is the dequantize scale of a chain CollapseUNQDQ would remove, which would
// also drop or block a factor inserted between its Subtract and itself.
bool is_dequantize_multiply(const std::shared_ptr<ov::Node>& multiply) {
    for (auto&& value : multiply->input_values()) {
        const auto subtract = value.get_node_shared_ptr();
        if (!ov::is_type<ov::op::v1::Subtract>(subtract)) {
            continue;
        }
        const auto cvt_dq = subtract->get_input_node_shared_ptr(0);
        if (ov::is_type<ov::op::v0::Convert>(cvt_dq) &&
            ov::is_type<ov::op::v0::Convert>(cvt_dq->get_input_node_shared_ptr(0)) &&
            ov::is_type<ov::op::v0::FakeQuantize>(cvt_dq->get_input_node_ptr(0)->get_input_node_shared_ptr(0))) {
            return true;
        }
    }
    return false;
}

float fq_range(const std::shared_ptr<ov::op::v0::FakeQuantize>& fq) {
    float range = 0.f;
    for (std::size_t idx : {1u, 2u}) {
        const auto bound = ov::util::get_constant_from_source(fq->input_value(idx));
        if (!bound) {
            return 0.f;
        }
        for (float v : bound->cast_vector<float>()) {
            range = std::max(range, std::fabs(v));
        }
    }
    return range;
}

// Matches Divide(x, Sqrt(Add|Maximum(ReduceSum(Power(x, 2)), eps))), the ONNX LpNormalization
// (p = 2) decomposition. Like NormalizeL2, it is invariant to a scalar factor on x while eps is
// negligible against the sum of squares.
bool is_l2_normalize(const std::shared_ptr<ov::Node>& divide) {
    if (!ov::is_type<ov::op::v1::Divide>(divide)) {
        return false;
    }
    const auto sqrt = divide->get_input_node_shared_ptr(1);
    if (!ov::is_type<ov::op::v0::Sqrt>(sqrt)) {
        return false;
    }
    const auto eps = sqrt->get_input_node_shared_ptr(0);
    if (!ov::is_type<ov::op::v1::Add>(eps) && !ov::is_type<ov::op::v1::Maximum>(eps)) {
        return false;
    }
    const auto sum = eps->get_input_node_shared_ptr(0);
    if (!ov::is_type<ov::op::v1::ReduceSum>(sum) || !ov::util::get_constant_from_source(eps->input_value(1))) {
        return false;
    }
    const auto power = sum->get_input_node_shared_ptr(0);
    if (!ov::is_type<ov::op::v1::Power>(power) || power->input_value(0) != divide->input_value(0)) {
        return false;
    }
    const auto exponent = ov::util::get_constant_from_source(power->input_value(1));
    return exponent && exponent->cast_vector<float>() == std::vector<float>{2.f};
}

// True if `power` is the sum-of-squares branch of L2 normalizations of its input and nothing else.
bool feeds_only_l2_normalize(const std::shared_ptr<ov::Node>& power) {
    const auto sum = single_consumer(power);
    const auto eps = sum ? single_consumer(sum) : nullptr;
    const auto sqrt = eps ? single_consumer(eps) : nullptr;
    if (!ov::is_type<ov::op::v0::Sqrt>(sqrt)) {
        return false;
    }
    for (auto&& target : sqrt->output(0).get_target_inputs()) {
        const auto divide = target.get_node()->shared_from_this();
        if (!is_l2_normalize(divide) || divide->input_value(0) != power->input_value(0)) {
            return false;
        }
    }
    return true;
}

// Walks every path from `start`, collecting the FQs it passes through. Succeeds only if each path
// is linear in `start` and ends in an L2 normalization, so a factor on `start` cancels there.
bool collect_scaled_region(const ov::Output<ov::Node>& start,
                           std::vector<std::shared_ptr<ov::op::v0::FakeQuantize>>& fqs) {
    std::unordered_set<ov::Node*> scaled{start.get_node()};
    std::vector<ov::Output<ov::Node>> queue{start};
    std::vector<std::shared_ptr<ov::Node>> add_subs;
    std::vector<std::shared_ptr<ov::Node>> products;
    while (!queue.empty()) {
        const auto value = queue.back();
        queue.pop_back();
        for (auto&& target : value.get_target_inputs()) {
            const auto node = target.get_node()->shared_from_this();
            auto output = node;
            if (const auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(node)) {
                // The chain collapses to identity, so the factor passes through it
                output = target.get_index() == 0 ? unqdq_chain_end(fq) : nullptr;
                if (!output) {
                    return false;
                }
                fqs.push_back(fq);
            } else if ((ov::is_type<ov::op::v0::NormalizeL2>(node) || is_l2_normalize(node)) &&
                       target.get_index() == 0) {
                continue;
            } else if (ov::is_type<ov::op::v1::Power>(node)) {
                if (!feeds_only_l2_normalize(node)) {
                    return false;
                }
                continue;
            } else if (ov::is_type<ov::op::v1::Add>(node) || ov::is_type<ov::op::v1::Subtract>(node)) {
                add_subs.push_back(node);
            } else if (ov::is_type<ov::op::v1::Multiply>(node) || ov::is_type<ov::op::v0::MatMul>(node)) {
                products.push_back(node);
            } else if (target.get_index() != 0 || !node->get_output_element_type(0).is_real() ||
                       !(ov::is_type<ov::op::v0::Convert>(node) || ov::is_type<ov::op::v1::Reshape>(node) ||
                         ov::is_type<ov::op::v0::Squeeze>(node) || ov::is_type<ov::op::v0::Unsqueeze>(node) ||
                         ov::is_type<ov::op::v1::Transpose>(node) || ov::is_type<ov::op::v1::ReduceSum>(node))) {
                LOG_DEBUG("RescaleFP16OverflowQDQ: path from " << start.get_node()->get_friendly_name()
                                                               << " reaches unsupported " << node->get_type_name()
                                                               << " " << node->get_friendly_name());
                return false;
            }
            if (scaled.insert(output.get()).second) {
                queue.push_back(output->output(0));
            }
        }
    }
    const auto is_scaled = [&](const std::shared_ptr<ov::Node>& node, std::size_t idx) {
        return scaled.count(node->get_input_node_ptr(idx)) > 0;
    };
    // A sum stays linear only if both terms carry the factor; a product only if exactly one does
    return std::all_of(add_subs.begin(),
                       add_subs.end(),
                       [&](const std::shared_ptr<ov::Node>& node) {
                           return is_scaled(node, 0) && is_scaled(node, 1);
                       }) &&
           std::all_of(products.begin(), products.end(), [&](const std::shared_ptr<ov::Node>& node) {
               return is_scaled(node, 0) != is_scaled(node, 1);
           });
}

}  // namespace

class CollapseUNQDQChain final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::CollapseUNQDQChain");

    CollapseUNQDQChain() {
        // Pattern:
        //   FakeQuantize(input, ...)
        //     → Convert (quantize, e.g. to u16)
        //       → Convert (dequantize, back to f32)
        //         → Subtract (zero-point)
        //           → Multiply (scale)   ← anchor
        auto input = opp::any_input();
        auto fake_q = opp::wrap_type<ov::op::v0::FakeQuantize>(
            {input, opp::any_input(), opp::any_input(), opp::any_input(), opp::any_input()});
        auto cvt_q = opp::wrap_type<ov::op::v0::Convert>({fake_q});
        auto cvt_dq = opp::wrap_type<ov::op::v0::Convert>({cvt_q});
        auto subtract = opp::wrap_type<ov::op::v1::Subtract>({cvt_dq, opp::any_input()});
        auto multiply = opp::wrap_type<ov::op::v1::Multiply>({subtract, opp::any_input()});

        // Note: use [=] so pattern nodes stay alive inside the callback.
        ov::matcher_pass_callback callback = [=](opp::Matcher& m) {
            auto& node_to_output = m.get_pattern_value_map();

            const auto matched_multiply = ov::as_type_ptr<ov::op::v1::Multiply>(m.get_match_root());
            const auto matched_input = node_to_output.at(input);

            ov::Output<ov::Node> replacement = matched_input;
            replacement = preserve_output_type(replacement,
                                               matched_multiply->get_output_element_type(0),
                                               matched_multiply->get_friendly_name());
            ov::replace_node(matched_multiply, ov::OutputVector{replacement});
            return true;
        };

        register_matcher(std::make_shared<opp::Matcher>(multiply, "CollapseUNQDQChain"), callback);
    }
};

ov::npuw::CollapseUNQDQ::CollapseUNQDQ() {
    add_matcher<CollapseUNQDQChain>();
}

bool ov::npuw::RescaleFP16OverflowQDQ::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    std::unordered_set<ov::Node*> handled;
    for (auto&& node : model->get_ordered_ops()) {
        const auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(node);
        if (!fq || handled.count(fq.get()) || fq_range(fq) <= kF16Max) {
            continue;
        }
        // The producer overflows before the FQ sees its output, so the factor goes on its input
        const auto producer = fq->get_input_node_shared_ptr(0);
        if (!(ov::is_type<ov::op::v1::Multiply>(producer) && !is_dequantize_multiply(producer)) &&
            !ov::is_type<ov::op::v0::MatMul>(producer)) {
            continue;
        }
        auto input = producer->input(0);
        std::vector<std::shared_ptr<ov::op::v0::FakeQuantize>> fqs;
        if (!input.get_element_type().is_real() || !collect_scaled_region(producer->output(0), fqs)) {
            continue;
        }
        float range = 0.f;
        for (auto&& region_fq : fqs) {
            range = std::max(range, fq_range(region_fq));
            handled.insert(region_fq.get());
        }
        const int k = static_cast<int>(std::ceil(std::log2(range / kRescaleTarget)));
        const auto factor = ov::op::v0::Constant::create(input.get_element_type(), ov::Shape{}, {std::ldexp(1.f, -k)});
        input.replace_source_output(std::make_shared<ov::op::v1::Multiply>(input.get_source_output(), factor));
        LOG_INFO("RescaleFP16OverflowQDQ: scaled " << producer->get_friendly_name() << " by 2^-" << k << " for range "
                                                   << range);
        changed = true;
    }
    return changed;
}
