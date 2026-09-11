// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/softmax_decomposition.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/log.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/op/convert_saturation.hpp"
#include "snippets/op/convert_truncation.hpp"
#include "snippets/op/powerstatic.hpp"
#include "snippets/op/reduce.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::snippets::pass {

namespace {

// A decomposed FakeQuantize is at most a clamp, a scale, a shift, a rounding and a narrowing.
// Sixteen is clear of that and bounds the walk on graphs that are not this pattern.
constexpr size_t max_walk_depth = 16;

// What the walk knows about the deferred operand where it has got to. `peak` bounds its magnitude
// and is exactly 1.0 at the Softmax, since exp(s - rowmax) attains 1 on every row and never
// exceeds it. A step that scales the operand moves `peak` with it, so `peak > 1.0` is also the
// test for "a quantization scale has been crossed": nothing else can raise it. `type` is what the
// operand is held in at that point, so that a scale can be refused when its own result would no
// longer fit -- the deferred operand is the larger of the two forms and is the one that overflows
// first.
struct DeferredOperand {
    double peak = 1.0;
    ov::element::Type type;
};

// A quantizer's scale, zero point or clamp bound is a scalar or one value per channel. Reading a
// constant costs a full conversion to float, so an operand far larger than that is not one of
// these and is refused before it is decoded rather than after.
constexpr size_t max_constant_elements = 4096;

// Lowest and highest element of `node`'s constant input `idx`. False when that input is not a
// constant whose values this can read.
bool constant_extremes(const ov::Node* node, size_t idx, float& lowest, float& highest) {
    const auto* constant = ov::as_type<const ov::op::v0::Constant>(node->get_input_node_ptr(idx));
    if (constant == nullptr || constant->get_element_type().bitwidth() < 8 ||
        ov::shape_size(constant->get_shape()) > max_constant_elements) {
        return false;
    }
    const auto values = constant->cast_vector<float>();
    if (values.empty() || std::any_of(values.begin(), values.end(), [](float value) {
            return std::isnan(value);
        })) {
        return false;
    }
    const auto bounds = std::minmax_element(values.begin(), values.end());
    lowest = *bounds.first;
    highest = *bounds.second;
    return true;
}

// `type` is known to represent a value of magnitude `peak`. This is a whitelist rather than a
// fallthrough on is_integral(), which is merely !is_real() and would admit string and the sub-byte
// float types; anything whose range is not derived here is declined. boolean is deliberately absent
// even though its range is trivially known: a cast to it is the predicate x != 0, which is not a
// narrowing of a magnitude and does not commute with a scale.
bool type_can_represent(const ov::element::Type& type, double peak) {
    if (!std::isfinite(peak)) {
        return false;
    }
    // A finite double cannot exceed f64's range, and f32 arithmetic could not have carried a value
    // past f32's range to reach this point. bf16 shares f32's exponent range.
    if (type == ov::element::f64) {
        return true;
    }
    if (type == ov::element::f32 || type == ov::element::bf16) {
        return peak <= static_cast<double>(std::numeric_limits<float>::max());
    }
    if (type == ov::element::f16) {
        return peak <= 65504.0;
    }
    static constexpr std::array<ov::element::Type_t, 10> integral_types{ov::element::Type_t::i4,
                                                                        ov::element::Type_t::i8,
                                                                        ov::element::Type_t::i16,
                                                                        ov::element::Type_t::i32,
                                                                        ov::element::Type_t::i64,
                                                                        ov::element::Type_t::u4,
                                                                        ov::element::Type_t::u8,
                                                                        ov::element::Type_t::u16,
                                                                        ov::element::Type_t::u32,
                                                                        ov::element::Type_t::u64};
    if (std::find(integral_types.begin(), integral_types.end(), type) == integral_types.end()) {
        return false;
    }
    const auto bits = static_cast<int>(type.bitwidth());
    return peak <= std::ldexp(1.0, type.is_signed() ? bits - 1 : bits) - 1.0;
}

// The rescale multiplies the dequantized product, which the deferral leaves a factor of the row sum
// larger than it would otherwise be, and the row sum is at least 1 and at most the reduced
// dimension. Only the types with at least f32's exponent range are taken to hold that; f16 would
// reach infinity on a long enough row and is declined rather than reasoned about. This says nothing
// about the i32 accumulator, which `accumulator_absorbs_deferred_operand` covers.
bool absorbs_deferred_magnitude(const ov::element::Type& type) {
    return type == ov::element::f32 || type == ov::element::f64 || type == ov::element::bf16;
}

bool is_shape_preserving_single_output(const ov::Node* node, size_t data_idx) {
    return node->get_output_size() == 1 && node->get_output_partial_shape(0) == node->get_input_partial_shape(data_idx);
}

bool has_constant_operand(const ov::Node* node, size_t data_idx, float& lowest, float& highest) {
    return node->get_input_size() == 2 && constant_extremes(node, 1 - data_idx, lowest, highest);
}

// A step that a positive per-row scale commutes with: a scale by a positive constant, or a shift by
// zero.
bool cross_scaling_step(const ov::Node* node, size_t data_idx, DeferredOperand& operand) {
    float lowest = 0.F;
    float highest = 0.F;
    if (!has_constant_operand(node, data_idx, lowest, highest)) {
        return false;
    }
    if (ov::is_type<ov::op::v1::Multiply>(node)) {
        if (lowest <= 0.F || !std::isfinite(highest)) {
            return false;
        }
        operand.peak *= highest;
        return type_can_represent(operand.type, operand.peak);
    }
    // Only with the data on the numerator: c / x is a reciprocal, not a scale, and its magnitude
    // grows without bound as the operand shrinks.
    if (ov::is_type<ov::op::v1::Divide>(node) && data_idx == 0) {
        if (lowest <= 0.F || !std::isfinite(highest)) {
            return false;
        }
        operand.peak /= lowest;
        return type_can_represent(operand.type, operand.peak);
    }
    // Commuting a scale past x + c would scale c with it. Subtract is accepted on the numerator
    // side only, since 0 - x negates and the operand is relied on downstream to be non-negative.
    if (ov::is_type<ov::op::v1::Add>(node) || (ov::is_type<ov::op::v1::Subtract>(node) && data_idx == 0)) {
        return lowest == 0.F && highest == 0.F;
    }
    return false;
}

// Rounding and narrowing to an integer type are the two steps that do not commute with a scale at
// all: they are meaning-preserving here only as the quantization step itself, applied to an operand
// a scale has already spread over the target grid. Applied to values in (0, 1] they collapse them
// -- a bare Round would send every probability to 0 but the deferred row maximum to 1, which is a
// different function, not a differently quantized one.
//
// `peak` is exactly 1.0 until a scale moves it, so this tests "a scale came first", which is
// necessary rather than sufficient: a gain barely above 1 passes it and still rounds the two forms
// differently. Nothing tighter is available here, because the grid the rounding lands on is only
// known at the narrowing that follows; a real FakeQuantize cannot produce such a gain, since
// levels >= 2 over a range the operand fills gives at least the operand's own magnitude back.
bool is_quantization_step(const DeferredOperand& operand) {
    return operand.peak > 1.0;
}

// One step of the quantizer between the softmax and the matmul.
//
// A clamp may be crossed only when it does not bite `peak`: a range calibrated on the normalized
// probabilities generally stops below it, and clipping there cannot be undone by the rescale applied
// after the matmul.
bool cross_quantizer_step(const ov::Node* node, size_t data_idx, DeferredOperand& operand) {
    if (!is_shape_preserving_single_output(node, data_idx)) {
        return false;
    }
    if (ov::is_type<ov::op::v5::Round>(node)) {
        if (!is_quantization_step(operand)) {
            return false;
        }
        // A code can round up, so what leaves is bounded by the ceiling rather than by `peak`. The
        // accumulator bound is derived from this, and a fraction of a code matters there.
        operand.peak = std::ceil(operand.peak);
        return true;
    }
    if (ov::is_type_any_of<ov::op::v0::Convert,
                           ov::snippets::op::ConvertSaturation,
                           ov::snippets::op::ConvertTruncation>(node)) {
        const auto& to = node->get_output_element_type(0);
        if ((!to.is_real() && !is_quantization_step(operand)) || !type_can_represent(to, operand.peak)) {
            return false;
        }
        operand.type = to;
        return true;
    }
    float lowest = 0.F;
    float highest = 0.F;
    if (ov::is_type<ov::op::v1::Maximum>(node)) {
        return has_constant_operand(node, data_idx, lowest, highest) && highest <= 0.F;
    }
    if (ov::is_type<ov::op::v1::Minimum>(node)) {
        return has_constant_operand(node, data_idx, lowest, highest) && lowest >= operand.peak;
    }
    return cross_scaling_step(node, data_idx, operand);
}

// FuseTransposeBrgemm runs before this pass and folds a transpose into whichever port it sat on,
// recording it as a non-planar layout there. Under either fold the product's axes no longer
// correspond to the row sums' and a row would be divided by another row's sum, which a shape
// comparison cannot see. Operand 1 is not checked: it does not carry the probabilities.
bool has_planar_probability_layouts(const std::shared_ptr<ov::snippets::op::Brgemm>& brgemm) {
    const auto& in_desc = lowered::PortDescriptorUtils::get_port_descriptor_ptr(brgemm->input(0));
    const auto& out_desc = lowered::PortDescriptorUtils::get_port_descriptor_ptr(brgemm->output(0));
    return utils::is_planar_layout(in_desc->get_layout()) && utils::is_planar_layout(out_desc->get_layout());
}

// The normalized operand's row sums to the quantization grid maximum however long the row is, so the
// i32 accumulator never had to be reasoned about. The deferred operand's does not: every element is
// up to `peak` in its own right, so the accumulation grows with the reduced dimension and this is
// the one bound the deferral can actually break. Brgemm::get_output_type only yields i32 for an i8
// weight operand, which is what bounds the other factor.
bool accumulator_absorbs_deferred_operand(const std::shared_ptr<ov::snippets::op::Brgemm>& brgemm, double peak) {
    const auto& shape = brgemm->get_input_partial_shape(0);
    if (shape.rank().is_dynamic() || shape.size() < 2 || shape[shape.size() - 1].is_dynamic()) {
        return false;
    }
    constexpr double weight_peak = 128.0;
    const auto reduced_dimension = static_cast<double>(shape[shape.size() - 1].get_length());
    return peak * reduced_dimension * weight_peak <= static_cast<double>(std::numeric_limits<int32_t>::max());
}

// The matmul that consumes this softmax, when the graph is a quantized attention chain
//     Softmax -> <quantizer> -> Brgemm accumulating in i32
// and a null pointer for every other graph, with `operand` left describing however far it got. Only
// operand 0 carries the probabilities.
std::shared_ptr<ov::snippets::op::Brgemm> find_int8_consumer(const ov::Output<ov::Node>& softmax,
                                                             DeferredOperand& operand) {
    operand.type = softmax.get_element_type();
    ov::Output<ov::Node> cursor = softmax;
    for (size_t hop = 0; hop < max_walk_depth; ++hop) {
        const auto targets = cursor.get_target_inputs();
        if (targets.size() != 1) {
            return nullptr;
        }
        const auto target = *targets.begin();
        auto* next = target.get_node();
        if (ov::is_type<ov::snippets::op::Brgemm>(next)) {
            const auto brgemm = ov::as_type_ptr<ov::snippets::op::Brgemm>(next->shared_from_this());
            const bool accumulates_in_i32 = brgemm->get_output_element_type(0) == ov::element::i32;
            if (target.get_index() != 0 || !accumulates_in_i32) {
                return nullptr;
            }
            return brgemm;
        }
        if (!cross_quantizer_step(next, target.get_index(), operand)) {
            return nullptr;
        }
        cursor = next->output(0);
    }
    return nullptr;
}

// An integer matmul accumulates in i32 and the return to real arithmetic is a separate node, so the
// rescale is anchored on that node's output. There is no walk to do: a narrowing to a smaller
// integer type would clip an accumulator the deferral has made larger, and an elementwise step whose
// operands are still integers cannot produce a real result, so the conversion has to be the matmul's
// immediate and only consumer.
ov::Output<ov::Node> find_dequantized_output(const std::shared_ptr<ov::snippets::op::Brgemm>& brgemm) {
    const auto targets = brgemm->output(0).get_target_inputs();
    if (targets.size() != 1) {
        return {};
    }
    const auto target = *targets.begin();
    auto* next = target.get_node();
    const bool is_conversion = ov::is_type_any_of<ov::op::v0::Convert,
                                                  ov::snippets::op::ConvertSaturation,
                                                  ov::snippets::op::ConvertTruncation>(next);
    if (!is_conversion || !is_shape_preserving_single_output(next, target.get_index()) ||
        !next->get_output_element_type(0).is_real()) {
        return {};
    }
    return next->output(0);
}

// Multiplying `anchor` by the row sums leaves `anchor`'s shape intact. A dynamic dimension
// broadcast-merges with anything, so it is declined rather than merged away vacuously.
bool rescale_preserves_shape(const ov::Output<ov::Node>& anchor, const ov::Output<ov::Node>& row_sum_reciprocal) {
    const auto& anchor_shape = anchor.get_partial_shape();
    const auto& sums_shape = row_sum_reciprocal.get_partial_shape();
    if (anchor_shape.is_dynamic() || sums_shape.is_dynamic()) {
        return false;
    }
    auto merged = anchor_shape;
    if (!ov::PartialShape::broadcast_merge_into(merged, sums_shape, ov::op::AutoBroadcastType::NUMPY)) {
        return false;
    }
    return merged == anchor_shape;
}

// Where the reciprocal row sums should be applied instead of to the Softmax, and the matmul they
// have to stay ordered against. An empty `anchor` when any precondition of the reassociation does
// not hold on the graph, in which case the caller emits the decomposition exactly as it always was.
struct Deferral {
    ov::Output<ov::Node> anchor;
    std::shared_ptr<ov::snippets::op::Brgemm> brgemm;
};

Deferral find_deferral(const ov::Output<ov::Node>& softmax, const ov::Output<ov::Node>& row_sum_reciprocal) {
    DeferredOperand operand;
    const auto brgemm = find_int8_consumer(softmax, operand);
    if (!brgemm || !has_planar_probability_layouts(brgemm) ||
        !accumulator_absorbs_deferred_operand(brgemm, operand.peak)) {
        return {};
    }
    const auto anchor = find_dequantized_output(brgemm);
    if (anchor.get_node() == nullptr) {
        return {};
    }
    // The row sums carry the softmax input's type and the anchor whatever the dequantization
    // converted to. This pass runs before PropagatePrecision, so those need not agree yet, and
    // multiplying operands of different types would throw out of the callback rather than decline.
    if (anchor.get_element_type() != row_sum_reciprocal.get_element_type() ||
        !absorbs_deferred_magnitude(anchor.get_element_type()) ||
        !rescale_preserves_shape(anchor, row_sum_reciprocal)) {
        return {};
    }
    return {anchor, brgemm};
}

}  // namespace

SoftmaxDecomposition::SoftmaxDecomposition(bool defer_normalization) {
    MATCHER_SCOPE(SoftmaxDecomposition);
    auto softmax_v1_m = ov::pass::pattern::wrap_type<ov::op::v1::Softmax>();
    auto softmax_v8_m = ov::pass::pattern::wrap_type<ov::op::v8::Softmax>();
    auto softmax_m = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{softmax_v1_m, softmax_v8_m});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::SoftmaxDecomposition")
        auto softmax = m.get_match_root();

        const auto& pshape = softmax->get_input_partial_shape(0);
        OPENVINO_ASSERT(!pshape.rank().is_dynamic(), "SoftmaxDecomposition doesn't support dynamic ranks");
        const auto rank = pshape.size();

        const auto axis = ov::snippets::utils::get_softmax_axis(softmax);
        if (!axis) {
            OPENVINO_THROW("Unexpected node matched");
        }
        const auto normalized_axis = static_cast<size_t>(*axis);

        const auto& softmax_input = softmax->input_value(0);
        const auto reduce_max = std::make_shared<ov::snippets::op::ReduceMax>(softmax_input, normalized_axis);
        ov::snippets::op::ReduceBase::compute_and_set_reduce_subtensors(reduce_max);
        const auto subtract = std::make_shared<ov::op::v1::Subtract>(softmax_input, reduce_max);
        const auto exp = std::make_shared<ov::op::v0::Exp>(subtract);

        const auto reduce_sum = std::make_shared<ov::snippets::op::ReduceSum>(exp, normalized_axis);
        ov::snippets::op::ReduceBase::compute_and_set_reduce_subtensors(reduce_sum);
        const auto power = std::make_shared<ov::snippets::op::PowerStatic>(reduce_sum, -1.F);

        OPENVINO_ASSERT(normalized_axis < rank, "Softmax has incorrect axis");
        std::vector<size_t> subtensor(rank, 1);
        for (size_t i = normalized_axis; i < rank; ++i) {
            subtensor[i] = utils::get_full_dim_value();
        }
        lowered::PortDescriptorUtils::set_port_descriptor(power->input(0), subtensor);
        lowered::PortDescriptorUtils::set_port_descriptor(power->output(0), std::move(subtensor));

        // On a quantized attention chain the reciprocal row sums are applied to the output of the
        // matmul that consumes the softmax rather than to the softmax itself, so that what reaches
        // the quantizer is exp(s - rowmax) instead of the normalized probabilities. Every
        // precondition is checked on the graph; any of them failing leaves the decomposition
        // exactly as it was.
        // The reassociation holds only when the axis the row sums reduce is the axis the matmul
        // contracts. Both tokenizers pin the softmax to the last axis well before this pass, but
        // nothing here would otherwise notice a body where that is not so.
        const bool reduces_contracted_axis = normalized_axis + 1 == rank;
        auto deferral = defer_normalization && reduces_contracted_axis
                            ? find_deferral(softmax->output(0), power->output(0))
                            : Deferral{};
        if (deferral.anchor.get_node() != nullptr) {
            const auto sinks = deferral.anchor.get_target_inputs();
            // With the rescale moved past it, the matmul no longer consumes the row sums, so a
            // topological sort is free to schedule the reduction after the matmul -- which reads
            // the exp tile a second time and inverts the buffer lifetimes the memory solver then
            // has to place. The control dependency is what orders the two. It is not removed
            // afterwards: it is the only record of an ordering the data flow no longer implies, and
            // ov::replace_node carries it across any later replacement of either end.
            deferral.brgemm->add_control_dependency(power);
            const auto rescale = std::make_shared<ov::op::v1::Multiply>(deferral.anchor, power);
            for (const auto& sink : sinks) {
                sink.replace_source_output(rescale->output(0));
            }
            copy_runtime_info(softmax, {reduce_max, subtract, exp, reduce_sum, power, rescale});
            const auto softmax_name = softmax->get_friendly_name();
            rescale->set_friendly_name(softmax_name + "/rescale");
            // Tensor names move with the value their consumers read. The rescale now supplies what
            // the anchor used to, so a body whose output is the anchor would otherwise lose its
            // output tensor names -- an interface change, not a labelling one. The Softmax's own
            // names stay where replace_node_update_name puts them, on the Exp: the value
            // softmax(s) is no longer materialized anywhere, and the Exp is the point in the graph
            // it used to occupy. Its single consumer is the quantizer, so nothing reads it by name.
            const auto anchor_names = deferral.anchor.get_names();
            deferral.anchor.set_names({});
            rescale->output(0).set_names(anchor_names);
            OPENVINO_DEBUG("SoftmaxDecomposition deferred the normalization of ",
                           softmax_name,
                           " past ",
                           deferral.brgemm->get_friendly_name());
            // The Softmax's friendly name goes to the Exp, as it does to the Multiply on the path
            // below: replace_node_update_name puts it there, and a matcher pass leaving the matched
            // node's name on no node at all would lose it from dumps and profiling.
            return ov::replace_node_update_name(softmax, exp);
        }

        const auto multiply = std::make_shared<ov::op::v1::Multiply>(exp, power);
        copy_runtime_info(softmax, {reduce_max, subtract, exp, reduce_sum, power, multiply});
        return ov::replace_node_update_name(softmax, multiply);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(softmax_m, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::snippets::pass
