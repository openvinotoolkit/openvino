// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "collapse_unqdq.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../logging.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
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

constexpr const char* kRescaledKey = "npuw_fp16_rescaled";

// FakeQuantize(input) -> Convert (quantize) -> Convert (dequantize) -> Subtract (zero point) -> Multiply (scale).
// Shared by CollapseUNQDQ and the rescale guards so both agree on what a removable chain is. Multiply is
// commutative, so the matcher also accepts Multiply(scale, Subtract).
struct UNQDQPattern {
    std::shared_ptr<ov::Node> input;
    std::shared_ptr<ov::Node> fake_q;
    std::shared_ptr<ov::Node> subtract;
    std::shared_ptr<ov::Node> multiply;
};

UNQDQPattern make_unqdq_pattern() {
    UNQDQPattern p;
    p.input = opp::any_input();
    p.fake_q = opp::wrap_type<ov::op::v0::FakeQuantize>(
        {p.input, opp::any_input(), opp::any_input(), opp::any_input(), opp::any_input()});
    auto cvt_q = opp::wrap_type<ov::op::v0::Convert>({p.fake_q});
    auto cvt_dq = opp::wrap_type<ov::op::v0::Convert>({cvt_q});
    p.subtract = opp::wrap_type<ov::op::v1::Subtract>({cvt_dq, opp::any_input()});
    p.multiply = opp::wrap_type<ov::op::v1::Multiply>({p.subtract, opp::any_input()});
    return p;
}

std::shared_ptr<ov::Node> single_consumer(const std::shared_ptr<ov::Node>& node) {
    if (!node) {
        return nullptr;
    }
    const auto targets = node->output(0).get_target_inputs();
    return targets.size() == 1 ? targets.begin()->get_node()->shared_from_this() : nullptr;
}

// Returns the Multiply ending the chain that CollapseUNQDQ replaces with the FQ input, or nullptr if
// `fq` does not start such a chain with a single consumer at each step.
std::shared_ptr<ov::Node> unqdq_chain_end(const std::shared_ptr<ov::Node>& fq, const UNQDQPattern& pattern) {
    const auto multiply = single_consumer(single_consumer(single_consumer(single_consumer(fq))));
    if (!multiply) {
        return nullptr;
    }
    opp::Matcher matcher(pattern.multiply);
    if (!matcher.match(multiply->output(0)) ||
        matcher.get_pattern_value_map().at(pattern.fake_q).get_node() != fq.get()) {
        return nullptr;
    }
    return multiply;
}

// True if `value` is the zero-point Subtract of a chain. A factor inserted after it would complete
// the pattern, and CollapseUNQDQ would then drop the factor along with the chain.
bool is_dequantize_subtract(const ov::Output<ov::Node>& value, const UNQDQPattern& pattern) {
    return opp::Matcher(pattern.subtract).match(value);
}

// True if `multiply` is the dequantize scale of a chain CollapseUNQDQ would remove, which would
// also drop or block a factor inserted between its Subtract and itself.
bool is_dequantize_multiply(const std::shared_ptr<ov::Node>& multiply, const UNQDQPattern& pattern) {
    return opp::Matcher(pattern.multiply).match(multiply->output(0));
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
// (p = 2) decomposition. Like NormalizeL2, it is invariant to a factor s on x once eps is scaled by s^2.
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

// Returns the eps Add/Maximum if `power` is the sum-of-squares branch of L2 normalizations of its
// input and nothing else, or nullptr otherwise.
std::shared_ptr<ov::Node> l2_normalize_eps(const std::shared_ptr<ov::Node>& power) {
    const auto eps = single_consumer(single_consumer(power));
    const auto sqrt = single_consumer(eps);
    if (!ov::is_type<ov::op::v0::Sqrt>(sqrt)) {
        return nullptr;
    }
    for (auto&& target : sqrt->output(0).get_target_inputs()) {
        const auto divide = target.get_node()->shared_from_this();
        if (!is_l2_normalize(divide) || divide->input_value(0) != power->input_value(0)) {
            return nullptr;
        }
    }
    return eps;
}

struct ScaledRegion {
    std::vector<std::shared_ptr<ov::op::v0::FakeQuantize>> fqs;
    std::vector<std::shared_ptr<ov::op::v0::NormalizeL2>> norms;
    std::vector<std::shared_ptr<ov::Node>> eps_nodes;
};

// Walks every path from `start`, collecting the FQs it passes through and the L2 normalizations it
// ends in. Succeeds only if each path is linear in `start` and ends in an L2 normalization, so a
// factor on `start` cancels there.
bool collect_scaled_region(const ov::Output<ov::Node>& start, const UNQDQPattern& pattern, ScaledRegion& region) {
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
                output = target.get_index() == 0 ? unqdq_chain_end(fq, pattern) : nullptr;
                if (!output) {
                    return false;
                }
                region.fqs.push_back(fq);
            } else if (const auto norm = ov::as_type_ptr<ov::op::v0::NormalizeL2>(node)) {
                if (target.get_index() != 0) {
                    return false;
                }
                region.norms.push_back(norm);
                continue;
            } else if (is_l2_normalize(node) && target.get_index() == 0) {
                // Its eps is collected through the Power branch, which always reads the same value
                continue;
            } else if (ov::is_type<ov::op::v1::Power>(node)) {
                const auto eps = target.get_index() == 0 ? l2_normalize_eps(node) : nullptr;
                if (!eps) {
                    return false;
                }
                region.eps_nodes.push_back(eps);
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

// Returns eps * eps_scale as a Constant of the same type and shape, or nullptr if any value is not
// exactly representable there.
std::shared_ptr<ov::op::v0::Constant> scale_eps(const ov::Output<ov::Node>& eps, double eps_scale) {
    const auto value = ov::util::get_constant_from_source(eps);
    if (!value) {
        return nullptr;
    }
    auto values = value->cast_vector<double>();
    for (auto& v : values) {
        v *= eps_scale;
    }
    auto scaled = ov::op::v0::Constant::create(value->get_element_type(), value->get_shape(), values);
    return scaled->cast_vector<double>() == values ? scaled : nullptr;
}

}  // namespace

class CollapseUNQDQChain final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::CollapseUNQDQChain");

    CollapseUNQDQChain() {
        const auto pattern = make_unqdq_pattern();

        // Note: capture by value so pattern nodes stay alive inside the callback.
        ov::matcher_pass_callback callback = [=](opp::Matcher& m) {
            auto& node_to_output = m.get_pattern_value_map();

            const auto matched_multiply = ov::as_type_ptr<ov::op::v1::Multiply>(m.get_match_root());
            const auto matched_input = node_to_output.at(pattern.input);

            ov::Output<ov::Node> replacement = matched_input;
            replacement = preserve_output_type(replacement,
                                               matched_multiply->get_output_element_type(0),
                                               matched_multiply->get_friendly_name());
            ov::replace_node(matched_multiply, ov::OutputVector{replacement});
            return true;
        };

        register_matcher(std::make_shared<opp::Matcher>(pattern.multiply, "CollapseUNQDQChain"), callback);
    }
};

ov::npuw::CollapseUNQDQ::CollapseUNQDQ() {
    add_matcher<CollapseUNQDQChain>();
}

bool ov::npuw::RescaleFP16OverflowQDQ::run_on_model(const std::shared_ptr<ov::Model>& model) {
    const auto pattern = make_unqdq_pattern();
    bool changed = false;
    for (auto&& node : model->get_ordered_ops()) {
        const auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(node);
        if (!fq || fq->get_rt_info().count(kRescaledKey) || !(fq_range(fq) > kF16Max)) {
            continue;
        }
        // The producer overflows before the FQ sees its output, so the factor goes on its input.
        // Weights sit on MatMul input 1, so only input 0 is scaled.
        const auto producer = fq->get_input_node_shared_ptr(0);
        if (!(ov::is_type<ov::op::v1::Multiply>(producer) && !is_dequantize_multiply(producer, pattern)) &&
            !ov::is_type<ov::op::v0::MatMul>(producer)) {
            continue;
        }
        auto input = producer->input(0);
        if (!input.get_element_type().is_real() || is_dequantize_subtract(input.get_source_output(), pattern)) {
            continue;
        }
        ScaledRegion region;
        if (!collect_scaled_region(producer->output(0), pattern, region)) {
            continue;
        }
        float range = 0.f;
        bool rescaled = false;
        for (auto&& region_fq : region.fqs) {
            range = std::max(range, fq_range(region_fq));
            rescaled = rescaled || region_fq->get_rt_info().count(kRescaledKey) > 0;
        }
        if (rescaled || !std::isfinite(range)) {
            continue;
        }

        // Validate every new value before touching the graph
        const int k = static_cast<int>(std::ceil(std::log2(range / kRescaleTarget)));
        const double scale = std::ldexp(1.0, -k);
        const auto factor = ov::op::v0::Constant::create(input.get_element_type(), ov::Shape{}, {scale});
        if (factor->cast_vector<double>()[0] != scale) {
            LOG_DEBUG("RescaleFP16OverflowQDQ: 2^-" << k << " is not representable in " << input.get_element_type());
            continue;
        }
        const double eps_scale = scale * scale;
        std::vector<std::shared_ptr<ov::op::v0::Constant>> scaled_eps;
        bool representable = true;
        for (auto&& eps_node : region.eps_nodes) {
            scaled_eps.push_back(scale_eps(eps_node->input_value(1), eps_scale));
            representable = representable && scaled_eps.back();
        }
        std::vector<float> norm_eps;
        for (auto&& norm : region.norms) {
            const double eps = norm->get_eps() * eps_scale;
            norm_eps.push_back(static_cast<float>(eps));
            representable = representable && static_cast<double>(norm_eps.back()) == eps;
        }
        if (!representable) {
            LOG_DEBUG("RescaleFP16OverflowQDQ: eps of " << producer->get_friendly_name()
                                                        << " region is not representable after scaling");
            continue;
        }

        // Scale eps by s^2 so the normalizations stay exact for small activations too
        for (std::size_t i = 0; i < region.eps_nodes.size(); ++i) {
            const auto& eps_node = region.eps_nodes[i];
            scaled_eps[i]->set_friendly_name(eps_node->get_friendly_name() + "/fp16_rescaled_eps");
            ov::copy_runtime_info(eps_node->get_input_node_shared_ptr(1), scaled_eps[i]);
            eps_node->input(1).replace_source_output(scaled_eps[i]);
        }
        for (std::size_t i = 0; i < region.norms.size(); ++i) {
            const auto& norm = region.norms[i];
            const auto scaled_norm = std::make_shared<ov::op::v0::NormalizeL2>(norm->input_value(0),
                                                                               norm->input_value(1),
                                                                               norm_eps[i],
                                                                               norm->get_eps_mode());
            scaled_norm->set_friendly_name(norm->get_friendly_name());
            ov::copy_runtime_info(norm, scaled_norm);
            ov::replace_node(norm, scaled_norm);
        }

        factor->set_friendly_name(producer->get_friendly_name() + "/fp16_rescale_factor");
        const auto multiply = std::make_shared<ov::op::v1::Multiply>(input.get_source_output(), factor);
        multiply->set_friendly_name(producer->get_friendly_name() + "/fp16_rescale");
        ov::copy_runtime_info(producer, {factor, multiply});
        input.replace_source_output(multiply);
        // FQ bounds stay as calibrated, so mark the region to keep a second run from scaling it again
        for (auto&& region_fq : region.fqs) {
            region_fq->get_rt_info()[kRescaledKey] = true;
        }
        LOG_INFO("RescaleFP16OverflowQDQ: scaled " << producer->get_friendly_name() << " by 2^-" << k << " for range "
                                                   << range);
        changed = true;
    }
    return changed;
}