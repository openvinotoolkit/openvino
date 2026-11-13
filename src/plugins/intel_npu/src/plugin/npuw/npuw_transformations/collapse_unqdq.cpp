// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "collapse_unqdq.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
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
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
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

// FakeQuantize(input) -> Convert (quantize) -> Convert (dequantize) -> Subtract (zero point) -> Multiply (scale).
// Shared by CollapseUNQDQ and the rescale guards so both agree on what a removable chain is. Multiply is
// commutative, so the matcher also accepts Multiply(scale, Subtract).
struct UNQDQPattern {
    std::shared_ptr<ov::Node> input;
    std::shared_ptr<ov::Node> fake_q;
    std::shared_ptr<ov::Node> subtract;
    std::shared_ptr<ov::Node> multiply;
};

UNQDQPattern make_unqdq_pattern(const std::shared_ptr<ov::Node>& input = opp::any_input()) {
    UNQDQPattern p;
    p.input = input;
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

// The region this pass rescales: UNQDQ -> MatMul -> UNQDQ -> L2 normalization, where the
// normalization is NormalizeL2 or the ONNX LpNormalization (p = 2) decomposition
// Divide(x, Sqrt(Add|Maximum(ReduceSum(Power(x, 2)), eps))). Both are invariant to a factor s on x
// once eps is scaled by s^2.
struct OverflowPattern {
    UNQDQPattern in;
    std::shared_ptr<ov::Node> matmul;
    UNQDQPattern out;
    std::shared_ptr<ov::Node> norm;
    std::shared_ptr<ov::Node> power;
    std::shared_ptr<ov::Node> eps;
    std::shared_ptr<ov::Node> divide;
    std::shared_ptr<ov::Node> root;
};

OverflowPattern make_overflow_pattern() {
    OverflowPattern p;
    p.in = make_unqdq_pattern();
    p.matmul = opp::wrap_type<ov::op::v0::MatMul>({p.in.multiply, opp::any_input()});
    p.out = make_unqdq_pattern(p.matmul);
    p.norm = opp::wrap_type<ov::op::v0::NormalizeL2>({p.out.multiply, opp::any_input()});
    p.power = opp::wrap_type<ov::op::v1::Power>({p.out.multiply, opp::wrap_type<ov::op::v0::Constant>()});
    const auto sum = opp::wrap_type<ov::op::v1::ReduceSum>({p.power, opp::any_input()});
    p.eps = opp::wrap_type<ov::op::v1::Add, ov::op::v1::Maximum>({sum, opp::wrap_type<ov::op::v0::Constant>()});
    const auto sqrt = opp::wrap_type<ov::op::v0::Sqrt>({p.eps});
    p.divide = opp::wrap_type<ov::op::v1::Divide>({p.out.multiply, sqrt});
    p.root = std::make_shared<opp::op::Or>(ov::OutputVector{p.norm, p.divide});
    return p;
}

bool has_single_consumer(const std::shared_ptr<ov::Node>& node) {
    return node->output(0).get_target_inputs().size() == 1;
}

// The matched values must feed nothing else, or the factor would leak past the normalization
bool is_isolated(const OverflowPattern& p, const ov::pass::pattern::PatternValueMap& map) {
    const auto node = [&](const std::shared_ptr<ov::Node>& pattern) {
        return map.at(pattern).get_node_shared_ptr();
    };
    const auto in_end = node(p.in.multiply);
    const auto out_end = node(p.out.multiply);
    if (unqdq_chain_end(node(p.in.fake_q), p.in) != in_end || !has_single_consumer(in_end) ||
        !has_single_consumer(node(p.matmul)) || unqdq_chain_end(node(p.out.fake_q), p.out) != out_end) {
        return false;
    }
    if (map.count(p.norm)) {
        return has_single_consumer(out_end);
    }
    // Decomposed form: x feeds only the Divide and the Power, and each step in between has one consumer
    const auto power = node(p.power);
    const auto exponent = ov::as_type_ptr<ov::op::v0::Constant>(power->get_input_node_shared_ptr(1));
    return out_end->output(0).get_target_inputs().size() == 2 &&
           node(p.divide)->input_value(0) == power->input_value(0) && has_single_consumer(power) &&
           has_single_consumer(power->get_output_target_inputs(0).begin()->get_node()->shared_from_this()) &&
           has_single_consumer(node(p.eps)) &&
           has_single_consumer(node(p.eps)->get_output_target_inputs(0).begin()->get_node()->shared_from_this()) &&
           exponent->cast_vector<float>() == std::vector<float>{2.f};
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
    const auto p = make_overflow_pattern();
    bool changed = false;
    for (auto&& node : model->get_ordered_ops()) {
        opp::Matcher matcher(p.root);
        if (!matcher.match(node->output(0))) {
            continue;
        }
        const auto& map = matcher.get_pattern_value_map();
        if (!is_isolated(p, map)) {
            continue;
        }
        const auto fq_in = ov::as_type_ptr<ov::op::v0::FakeQuantize>(map.at(p.in.fake_q).get_node_shared_ptr());
        const auto fq_out = ov::as_type_ptr<ov::op::v0::FakeQuantize>(map.at(p.out.fake_q).get_node_shared_ptr());
        const float range_in = fq_range(fq_in);
        const float range_out = fq_range(fq_out);
        if (!std::isfinite(range_in) || !std::isfinite(range_out) || !(std::max(range_in, range_out) > kF16Max)) {
            continue;
        }

        // If the first chain overflows, its producer already does, so the factor goes on the
        // producer's input; otherwise on the MatMul input. Weights sit on MatMul input 1.
        const bool scale_producer = range_in > kF16Max;
        const auto matmul = map.at(p.matmul).get_node_shared_ptr();
        const auto producer = scale_producer ? fq_in->get_input_node_shared_ptr(0) : matmul;
        if (scale_producer && (!has_single_consumer(producer) || !((ov::is_type<ov::op::v1::Multiply>(producer) &&
                                                                    !is_dequantize_multiply(producer, p.in)) ||
                                                                   ov::is_type<ov::op::v0::MatMul>(producer)))) {
            continue;
        }
        auto input = producer->input(0);
        if (!input.get_element_type().is_real() || is_dequantize_subtract(input.get_source_output(), p.in)) {
            continue;
        }
        const float range = scale_producer ? std::max(range_in, range_out) : range_out;

        // Validate every new value before touching the graph
        const int k = static_cast<int>(std::ceil(std::log2(range / kRescaleTarget)));
        const double scale = std::ldexp(1.0, -k);
        const auto factor = ov::op::v0::Constant::create(input.get_element_type(), ov::Shape{}, {scale});
        if (factor->cast_vector<double>()[0] != scale) {
            LOG_DEBUG("RescaleFP16OverflowQDQ: 2^-" << k << " is not representable in " << input.get_element_type());
            continue;
        }
        const double eps_scale = scale * scale;
        const auto norm = map.count(p.norm)
                              ? ov::as_type_ptr<ov::op::v0::NormalizeL2>(map.at(p.norm).get_node_shared_ptr())
                              : nullptr;
        const auto eps_node = norm ? nullptr : map.at(p.eps).get_node_shared_ptr();
        const auto scaled_eps = norm ? nullptr : scale_eps(eps_node->input_value(1), eps_scale);
        const double norm_eps = norm ? norm->get_eps() * eps_scale : 0.0;
        if (norm ? static_cast<double>(static_cast<float>(norm_eps)) != norm_eps : !scaled_eps) {
            LOG_DEBUG("RescaleFP16OverflowQDQ: eps after " << producer->get_friendly_name()
                                                           << " is not representable after scaling");
            continue;
        }

        // Scale eps by s^2 so the normalization stays exact for small activations too
        if (norm) {
            const auto scaled_norm = std::make_shared<ov::op::v0::NormalizeL2>(norm->input_value(0),
                                                                               norm->input_value(1),
                                                                               static_cast<float>(norm_eps),
                                                                               norm->get_eps_mode());
            scaled_norm->set_friendly_name(norm->get_friendly_name());
            ov::copy_runtime_info(norm, scaled_norm);
            ov::replace_node(norm, scaled_norm);
        } else {
            scaled_eps->set_friendly_name(eps_node->get_friendly_name() + "/fp16_rescaled_eps");
            ov::copy_runtime_info(eps_node->get_input_node_shared_ptr(1), scaled_eps);
            eps_node->input(1).replace_source_output(scaled_eps);
        }

        factor->set_friendly_name(producer->get_friendly_name() + "/fp16_rescale_factor");
        const auto multiply = std::make_shared<ov::op::v1::Multiply>(input.get_source_output(), factor);
        multiply->set_friendly_name(producer->get_friendly_name() + "/fp16_rescale");
        ov::copy_runtime_info(producer, {factor, multiply});
        input.replace_source_output(multiply);
        // The FQ bounds stay as calibrated and would quantize the scaled values on a 2^k coarser grid,
        // so remove both chains here, as CollapseUNQDQ does
        for (const auto& [fq, end] :
             {std::pair{fq_in, map.at(p.in.multiply)}, std::pair{fq_out, map.at(p.out.multiply)}}) {
            const auto chain_end = end.get_node_shared_ptr();
            ov::replace_node(chain_end,
                             ov::OutputVector{preserve_output_type(fq->input_value(0),
                                                                   chain_end->get_output_element_type(0),
                                                                   chain_end->get_friendly_name())});
        }
        LOG_INFO("RescaleFP16OverflowQDQ: scaled " << producer->get_friendly_name() << " by 2^-" << k << " for range "
                                                   << range);
        changed = true;
    }
    return changed;
}
