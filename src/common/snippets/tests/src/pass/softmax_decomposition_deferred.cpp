// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/subtract.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/op/convert_saturation.hpp"
#include "snippets/op/powerstatic.hpp"
#include "snippets/op/reduce.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/pass/softmax_decomposition.hpp"
#include "snippets/utils/utils.hpp"

using namespace ov;

/* SoftmaxDecomposition applies the row sums to the output of the matmul that consumes the softmax
 * when, and only when, that matmul is an integer one reached across a quantizer that still
 * represents the deferred operand. There is one declining test per precondition, so that removing
 * any single guard turns a test red.
 */

namespace {

// The chain between the softmax and the quantizer. It may add parameters of its own, and is invoked
// once per model so that the two never share a node.
using ChainBuilder = std::function<Output<Node>(const Output<Node>&, ParameterVector&)>;

std::vector<size_t> row_subtensor(size_t rank, size_t axis) {
    std::vector<size_t> subtensor(rank, 1);
    for (size_t i = axis; i < rank; ++i) {
        subtensor[i] = ov::snippets::utils::get_full_dim_value();
    }
    return subtensor;
}

std::shared_ptr<Node> make_shifted_exp(const Output<Node>& scores, size_t axis) {
    const auto reduce_max = std::make_shared<ov::snippets::op::ReduceMax>(scores, axis);
    ov::snippets::op::ReduceBase::compute_and_set_reduce_subtensors(reduce_max);
    const auto subtract = std::make_shared<ov::op::v1::Subtract>(scores, reduce_max);
    return std::make_shared<ov::op::v0::Exp>(subtract);
}

std::shared_ptr<Node> make_row_sum_reciprocal(const Output<Node>& exp, size_t axis) {
    const auto reduce_sum = std::make_shared<ov::snippets::op::ReduceSum>(exp, axis);
    ov::snippets::op::ReduceBase::compute_and_set_reduce_subtensors(reduce_sum);
    const auto power = std::make_shared<ov::snippets::op::PowerStatic>(reduce_sum, -1.F);
    // The pass sets these, so a reference that omits them is not quite what it emits.
    const auto subtensor = row_subtensor(exp.get_partial_shape().size(), axis);
    ov::snippets::lowered::PortDescriptorUtils::set_port_descriptor(power->input(0), subtensor);
    ov::snippets::lowered::PortDescriptorUtils::set_port_descriptor(power->output(0), subtensor);
    return power;
}

// exp(x - rowmax) / rowsum, i.e. what the pass emits when it declines to defer.
std::shared_ptr<Node> make_normalized_softmax(const Output<Node>& scores, size_t axis) {
    const auto exp = make_shifted_exp(scores, axis);
    return std::make_shared<ov::op::v1::Multiply>(exp, make_row_sum_reciprocal(exp, axis));
}

// What CommonFakeQuantizeDecomposition leaves between the softmax and the matmul: a clamp to the
// calibrated range, then a scale, a rounding and a narrowing. The clamp is emitted unconditionally,
// so `input_high` decides whether the deferred operand -- exactly 1.0 on every row -- survives it,
// and `scale` against `narrow_to` decides whether it still fits the narrowing.
std::shared_ptr<Node> make_quantizer(const Output<Node>& probabilities,
                                     float input_high,
                                     float scale,
                                     const element::Type& narrow_to,
                                     bool saturating = false) {
    const auto& type = probabilities.get_element_type();
    const auto low = ov::op::v0::Constant::create(type, Shape{}, {0.F});
    const auto high = ov::op::v0::Constant::create(type, Shape{}, {input_high});
    const auto clamped_low = std::make_shared<ov::op::v1::Maximum>(probabilities, low);
    const auto clamped = std::make_shared<ov::op::v1::Minimum>(clamped_low, high);
    const auto step = ov::op::v0::Constant::create(type, Shape{}, {scale});
    const auto scaled = std::make_shared<ov::op::v1::Multiply>(clamped, step);
    const auto rounded = std::make_shared<ov::op::v5::Round>(scaled, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
    if (saturating) {
        return std::make_shared<ov::snippets::op::ConvertSaturation>(rounded, narrow_to);
    }
    return std::make_shared<ov::op::v0::Convert>(rounded, narrow_to);
}

// A matmul whose layouts are recorded where the pass reads them. Passing them to the ctor would not
// do: `layout_c` there only permutes the inferred shape, and validate_and_infer_types reverts it
// from the port descriptor, which is also where FuseTransposeBrgemm writes a fold.
std::shared_ptr<ov::snippets::op::Brgemm> make_brgemm(const Output<Node>& a,
                                                      const Output<Node>& b,
                                                      const std::vector<size_t>& input_layout = {},
                                                      const std::vector<size_t>& output_layout = {}) {
    const auto brgemm = std::make_shared<ov::snippets::op::Brgemm>(a, b);
    const std::vector<size_t> subtensor(2, ov::snippets::utils::get_full_dim_value());
    ov::snippets::lowered::PortDescriptorUtils::set_port_descriptor(brgemm->input(0), subtensor, input_layout);
    ov::snippets::lowered::PortDescriptorUtils::set_port_descriptor(brgemm->output(0), subtensor, output_layout);
    brgemm->validate_and_infer_types();
    return brgemm;
}

const Shape default_scores{1, 2, 8, 8};
const Shape default_values{1, 2, 8, 4};

// Shapes whose head and row extents are equal, so permuting them leaves the product's shape
// untouched. The shape check then cannot decline, and the layout is the only thing left that can
// tell a folded transpose from a planar one. Used by the three layout tests below.
const Shape equal_extent_scores{1, 8, 8, 8};
const Shape equal_extent_values{1, 8, 8, 4};

// One graph shape, varied by whichever precondition a test pins.
struct Chain {
    ChainBuilder between_softmax_and_quantizer = [](const Output<Node>& x, ParameterVector&) {
        return x;
    };
    float input_high = 1.F;
    float quantizer_scale = 255.F;
    element::Type narrow_to = element::u8;
    element::Type scores_type = element::f32;
    element::Type dequantize_to = element::f32;
    bool saturating_narrowing = false;
    int axis = -1;
    PartialShape scores = default_scores;
    PartialShape values = default_values;
    std::vector<size_t> input_layout{};
    std::vector<size_t> output_layout{};
    bool scale_the_accumulator = false;
    bool narrow_the_accumulator = false;
    bool fork_the_softmax = false;
    bool fork_the_accumulator = false;
};

}  // namespace

class SoftmaxDecompositionDeferredTests : public TransformationTestsF {
public:
    SoftmaxDecompositionDeferredTests() {
        comparator.enable(FunctionsComparator::CONST_VALUES);
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ov::snippets::pass::SoftmaxDecomposition>(true);
    }

    // Which of the three forms of the same chain to build: what the pass is given, what it emits
    // when it declines, and what it emits when it defers.
    enum class Form : uint8_t { Original, Normalized, Deferred };

    // `Softmax -> chain -> quantizer -> i8 Brgemm -> ... -> Convert(real)`.
    static std::shared_ptr<Model> build(const Chain& chain, Form form) {
        const auto rank = static_cast<int>(chain.scores.rank().get_length());
        const auto axis = static_cast<size_t>(chain.axis < 0 ? chain.axis + rank : chain.axis);
        auto scores = std::make_shared<ov::op::v0::Parameter>(chain.scores_type, chain.scores);
        auto values = std::make_shared<ov::op::v0::Parameter>(element::i8, chain.values);
        ParameterVector parameters{scores, values};
        Output<Node> probabilities;
        std::shared_ptr<Node> row_sum_reciprocal;
        if (form == Form::Original) {
            probabilities = std::make_shared<ov::op::v8::Softmax>(scores, chain.axis);
        } else if (form == Form::Normalized) {
            probabilities = make_normalized_softmax(scores, axis);
        } else {
            const auto exp = make_shifted_exp(scores, axis);
            row_sum_reciprocal = make_row_sum_reciprocal(exp, axis);
            probabilities = exp;
        }
        const auto quantized = make_quantizer(chain.between_softmax_and_quantizer(probabilities, parameters),
                                              chain.input_high,
                                              chain.quantizer_scale,
                                              chain.narrow_to,
                                              chain.saturating_narrowing);
        const auto brgemm = make_brgemm(quantized, values, chain.input_layout, chain.output_layout);
        if (row_sum_reciprocal) {
            brgemm->add_control_dependency(row_sum_reciprocal);
        }
        Output<Node> accumulator = brgemm->output(0);
        if (chain.scale_the_accumulator) {
            const auto gain = ov::op::v0::Constant::create(element::i32, Shape{}, {2});
            accumulator = std::make_shared<ov::op::v1::Multiply>(accumulator, gain);
        }
        if (chain.narrow_the_accumulator) {
            accumulator = std::make_shared<ov::op::v0::Convert>(accumulator, element::i8);
        }
        OutputVector outputs;
        if (chain.fork_the_accumulator) {
            outputs.push_back(std::make_shared<ov::op::v0::Convert>(accumulator, chain.dequantize_to));
        }
        Output<Node> dequantized = std::make_shared<ov::op::v0::Convert>(accumulator, chain.dequantize_to);
        if (row_sum_reciprocal) {
            dequantized = std::make_shared<ov::op::v1::Multiply>(dequantized, row_sum_reciprocal);
        }
        outputs.push_back(dequantized);
        if (chain.fork_the_softmax) {
            outputs.push_back(std::make_shared<ov::op::v0::Exp>(probabilities));
        }
        return std::make_shared<Model>(outputs, parameters);
    }

    // The pass leaves the chain alone: the reference normalizes the softmax in place.
    void expect_decline(const Chain& chain) {
        model = build(chain, Form::Original);
        model_ref = build(chain, Form::Normalized);
    }

    // The pass rewrites the chain: the reference applies the row sums after the dequantization.
    // Here the rescale is the body's last node, so the Result's friendly name derives from a node
    // the pass created and named after the Softmax -- it cannot match a reference whose equivalent
    // node was auto-named. The tensor names on that output, which are the part a caller can
    // observe, are still compared.
    void expect_defer(const Chain& chain) {
        disable_result_friendly_names_check();
        model = build(chain, Form::Original);
        model_ref = build(chain, Form::Deferred);
    }
};

TEST_F(SoftmaxDecompositionDeferredTests, DeferredPastInt8Brgemm) {
    {
        auto scores = std::make_shared<ov::op::v0::Parameter>(element::f32, default_scores);
        auto values = std::make_shared<ov::op::v0::Parameter>(element::i8, default_values);
        auto softmax = std::make_shared<ov::op::v8::Softmax>(scores, -1);
        auto brgemm = make_brgemm(make_quantizer(softmax, 1.F, 255.F, element::u8), values);
        auto dequantized = std::make_shared<ov::op::v0::Convert>(brgemm, element::f32);
        auto scale = ov::op::v0::Constant::create(element::f32, Shape{}, {1.F / 255.F});
        auto rescaled = std::make_shared<ov::op::v1::Multiply>(dequantized, scale);
        model = std::make_shared<Model>(OutputVector{rescaled}, ParameterVector{scores, values});
    }
    {
        auto scores = std::make_shared<ov::op::v0::Parameter>(element::f32, default_scores);
        auto values = std::make_shared<ov::op::v0::Parameter>(element::i8, default_values);
        auto exp = make_shifted_exp(scores, 3);
        auto row_sum_reciprocal = make_row_sum_reciprocal(exp, 3);
        auto brgemm = make_brgemm(make_quantizer(exp, 1.F, 255.F, element::u8), values);
        brgemm->add_control_dependency(row_sum_reciprocal);
        auto dequantized = std::make_shared<ov::op::v0::Convert>(brgemm, element::f32);
        // The row sums land on the first real-typed value after the matmul, ahead of the
        // dequantization multiply rather than after it.
        auto normalized = std::make_shared<ov::op::v1::Multiply>(dequantized, row_sum_reciprocal);
        auto scale = ov::op::v0::Constant::create(element::f32, Shape{}, {1.F / 255.F});
        auto rescaled = std::make_shared<ov::op::v1::Multiply>(normalized, scale);
        model_ref = std::make_shared<Model>(OutputVector{rescaled}, ParameterVector{scores, values});
    }
}

// With the anchor at the body output the rescale becomes the Result's producer, so it has to take
// over the tensor names identifying that value. The graph comparator's tensor-name check, seeded by
// CopyTensorNamesToRefModel, is what fails if it does not.
TEST_F(SoftmaxDecompositionDeferredTests, DeferredWhenAnchorIsBodyOutput) {
    disable_result_friendly_names_check();
    {
        auto scores = std::make_shared<ov::op::v0::Parameter>(element::f32, default_scores);
        auto values = std::make_shared<ov::op::v0::Parameter>(element::i8, default_values);
        auto softmax = std::make_shared<ov::op::v8::Softmax>(scores, -1);
        auto brgemm = make_brgemm(make_quantizer(softmax, 1.F, 255.F, element::u8), values);
        auto dequantized = std::make_shared<ov::op::v0::Convert>(brgemm, element::f32);
        model = std::make_shared<Model>(OutputVector{dequantized}, ParameterVector{scores, values});
    }
    {
        auto scores = std::make_shared<ov::op::v0::Parameter>(element::f32, default_scores);
        auto values = std::make_shared<ov::op::v0::Parameter>(element::i8, default_values);
        auto exp = make_shifted_exp(scores, 3);
        auto row_sum_reciprocal = make_row_sum_reciprocal(exp, 3);
        auto brgemm = make_brgemm(make_quantizer(exp, 1.F, 255.F, element::u8), values);
        brgemm->add_control_dependency(row_sum_reciprocal);
        auto dequantized = std::make_shared<ov::op::v0::Convert>(brgemm, element::f32);
        auto normalized = std::make_shared<ov::op::v1::Multiply>(dequantized, row_sum_reciprocal);
        model_ref = std::make_shared<Model>(OutputVector{normalized}, ParameterVector{scores, values});
    }
}

TEST_F(SoftmaxDecompositionDeferredTests, DeferredWithEqualExtentsAndPlanarLayout) {
    {
        auto scores = std::make_shared<ov::op::v0::Parameter>(element::f32, equal_extent_scores);
        auto values = std::make_shared<ov::op::v0::Parameter>(element::i8, equal_extent_values);
        auto softmax = std::make_shared<ov::op::v8::Softmax>(scores, -1);
        auto brgemm = make_brgemm(make_quantizer(softmax, 1.F, 255.F, element::u8), values);
        auto dequantized = std::make_shared<ov::op::v0::Convert>(brgemm, element::f32);
        auto scale = ov::op::v0::Constant::create(element::f32, Shape{}, {1.F / 255.F});
        auto rescaled = std::make_shared<ov::op::v1::Multiply>(dequantized, scale);
        model = std::make_shared<Model>(OutputVector{rescaled}, ParameterVector{scores, values});
    }
    {
        auto scores = std::make_shared<ov::op::v0::Parameter>(element::f32, equal_extent_scores);
        auto values = std::make_shared<ov::op::v0::Parameter>(element::i8, equal_extent_values);
        auto exp = make_shifted_exp(scores, 3);
        auto row_sum_reciprocal = make_row_sum_reciprocal(exp, 3);
        auto brgemm = make_brgemm(make_quantizer(exp, 1.F, 255.F, element::u8), values);
        brgemm->add_control_dependency(row_sum_reciprocal);
        auto dequantized = std::make_shared<ov::op::v0::Convert>(brgemm, element::f32);
        auto normalized = std::make_shared<ov::op::v1::Multiply>(dequantized, row_sum_reciprocal);
        auto scale = ov::op::v0::Constant::create(element::f32, Shape{}, {1.F / 255.F});
        auto rescaled = std::make_shared<ov::op::v1::Multiply>(normalized, scale);
        model_ref = std::make_shared<Model>(OutputVector{rescaled}, ParameterVector{scores, values});
    }
}

TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredPastFoldedOutputTranspose) {
    Chain chain;
    chain.scores = equal_extent_scores;
    chain.values = equal_extent_values;
    chain.output_layout = {0, 2, 1, 3};
    expect_decline(chain);
}

// FuseTransposeBrgemm folds a transpose into whichever port it sat on, so the operand port has to
// be checked too.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredPastFoldedOperandTranspose) {
    Chain chain;
    chain.scores = equal_extent_scores;
    chain.values = equal_extent_values;
    chain.input_layout = {0, 2, 1, 3};
    expect_decline(chain);
}

// A range calibrated on the normalized probabilities stops below 1.0, and the deferred operand
// reaches exactly 1.0 on every row. Deferring into that clamp would flatten every row's largest
// weight onto one code point, which the later rescale could not undo. Both bounds here are ones
// OpenVINO's own int8 MHA reference models put on this exact tensor: MHAINT8MatMulFunction's
// 0.820726 and MHAINT8MatMulTypeRelaxedFunction's 0.245, in ov_snippets_models/src/subgraph_mha.cpp.
// Neither reaches 1.0, so on both of those graphs the pass declines -- calibrated attention
// probabilities do not get close to 1, and widening that range is not this pass's business.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredIntoNarrowerQuantizerRange) {
    Chain chain;
    chain.input_high = 0.820726F;
    expect_decline(chain);
}

TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredIntoTypeRelaxedQuantizerRange) {
    Chain chain;
    chain.input_high = 0.245F;
    expect_decline(chain);
}

// A clamp from below above zero is not the identity on the deferred operand either.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredIntoRaisedQuantizerFloor) {
    Chain chain;
    chain.between_softmax_and_quantizer = [](const Output<Node>& x, ParameterVector&) -> Output<Node> {
        const auto floor = ov::op::v0::Constant::create(x.get_element_type(), Shape{}, {0.5F});
        return std::make_shared<ov::op::v1::Maximum>(x, floor);
    };
    expect_decline(chain);
}

// The narrowing has to represent the operand at the scale reached by then: 311 does not fit u8.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredWhenScaleOutgrowsTheNarrowing) {
    Chain chain;
    chain.quantizer_scale = 311.F;
    expect_decline(chain);
}

// ... and a signed narrowing stops at 127, not 255, so the bound has to respect signedness.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredWhenScaleOutgrowsASignedNarrowing) {
    Chain chain;
    chain.quantizer_scale = 200.F;
    chain.narrow_to = element::i8;
    expect_decline(chain);
}

// A scale crossed before a clamp raises the operand's bound, so the clamp is compared against that
// bound rather than against 1.0.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredWhenScaleOutgrowsClampBound) {
    Chain chain;
    chain.between_softmax_and_quantizer = [](const Output<Node>& x, ParameterVector&) -> Output<Node> {
        const auto gain = ov::op::v0::Constant::create(x.get_element_type(), Shape{}, {4.F});
        const auto amplified = std::make_shared<ov::op::v1::Multiply>(x, gain);
        const auto bound = ov::op::v0::Constant::create(x.get_element_type(), Shape{}, {1.5F});
        return std::make_shared<ov::op::v1::Minimum>(amplified, bound);
    };
    expect_decline(chain);
}

// A negative factor is not a positive scale: it would invert the clamp reasoning below it.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredAcrossNegativeScale) {
    Chain chain;
    chain.between_softmax_and_quantizer = [](const Output<Node>& x, ParameterVector&) -> Output<Node> {
        const auto gain = ov::op::v0::Constant::create(x.get_element_type(), Shape{}, {-1.F});
        return std::make_shared<ov::op::v1::Multiply>(x, gain);
    };
    expect_decline(chain);
}

// c / x is a reciprocal, not a scale: its magnitude grows without bound as the operand shrinks.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredAcrossReciprocal) {
    Chain chain;
    chain.between_softmax_and_quantizer = [](const Output<Node>& x, ParameterVector&) -> Output<Node> {
        const auto numerator = ov::op::v0::Constant::create(x.get_element_type(), Shape{}, {2.F});
        return std::make_shared<ov::op::v1::Divide>(numerator, x);
    };
    expect_decline(chain);
}

// 0 - x negates an operand the clamp rules rely on being non-negative.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredAcrossNegation) {
    Chain chain;
    chain.between_softmax_and_quantizer = [](const Output<Node>& x, ParameterVector&) -> Output<Node> {
        const auto zero = ov::op::v0::Constant::create(x.get_element_type(), Shape{}, {0.F});
        return std::make_shared<ov::op::v1::Subtract>(zero, x);
    };
    expect_decline(chain);
}

// Commuting a scale past x + c would scale c with it.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredAcrossNonZeroShift) {
    Chain chain;
    chain.between_softmax_and_quantizer = [](const Output<Node>& x, ParameterVector&) -> Output<Node> {
        const auto shift = ov::op::v0::Constant::create(x.get_element_type(), Shape{}, {1.F});
        return std::make_shared<ov::op::v1::Add>(x, shift);
    };
    expect_decline(chain);
}

// A join with a second activation is not a quantizer at all.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredAcrossActivationJoin) {
    Chain chain;
    chain.between_softmax_and_quantizer = [](const Output<Node>& x, ParameterVector& parameters) -> Output<Node> {
        auto bias = std::make_shared<ov::op::v0::Parameter>(x.get_element_type(), x.get_partial_shape());
        parameters.push_back(bias);
        return std::make_shared<ov::op::v1::Add>(x, bias);
    };
    expect_decline(chain);
}

// An op the whitelist does not name is not crossed, whatever its operands look like.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredAcrossAnUnrecognizedOp) {
    Chain chain;
    chain.between_softmax_and_quantizer = [](const Output<Node>& x, ParameterVector&) -> Output<Node> {
        const auto shape = ov::op::v0::Constant::create(element::i32, Shape{3}, {1, 16, 8});
        return std::make_shared<ov::op::v1::Reshape>(x, shape, false);
    };
    expect_decline(chain);
}

// A crossable op still has to preserve the shape of its data input, or an element no longer belongs
// to the row whose sum would rescale it. A per-head scale broadcasts, and is otherwise admissible.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredAcrossABroadcastingScale) {
    Chain chain;
    chain.scores = Shape{1, 1, 8, 8};
    chain.between_softmax_and_quantizer = [](const Output<Node>& x, ParameterVector&) -> Output<Node> {
        // The factors are 1.0 so that the bound they carry cannot make the clamp below decline
        // first; the only thing wrong with this step is that it changes the shape.
        const auto gain = ov::op::v0::Constant::create(x.get_element_type(), Shape{1, 2, 1, 1}, {1.F, 1.F});
        return std::make_shared<ov::op::v1::Multiply>(x, gain);
    };
    expect_decline(chain);
}

// The row sums reduce the softmax axis, so it has to be the axis the matmul contracts. The product
// is square here so that the shape check cannot decline first and mask this one.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredWhenSoftmaxAxisIsNotContracted) {
    Chain chain;
    chain.axis = -2;
    chain.values = default_scores;
    expect_decline(chain);
}

// replace_node_update_name rewrites every consumer of the softmax, so a second one would silently
// receive the unnormalized exponential.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredWhenSoftmaxHasASecondConsumer) {
    Chain chain;
    chain.fork_the_softmax = true;
    expect_decline(chain);
}

// A second consumer of an intermediate accumulator would keep the un-rescaled value.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredWhenAccumulatorHasASecondConsumer) {
    Chain chain;
    chain.fork_the_accumulator = true;
    expect_decline(chain);
}

// The deferral makes the accumulator up to a factor of K larger, so a narrowing on that leg would
// clip where it did not before.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredPastNarrowedAccumulator) {
    Chain chain;
    chain.narrow_the_accumulator = true;
    expect_decline(chain);
}

// ... and a scale on that leg can overflow it, so it is not crossed while the value is an integer.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredPastScaledAccumulator) {
    Chain chain;
    chain.scale_the_accumulator = true;
    expect_decline(chain);
}

// This pass runs before PropagatePrecision, so the row sums and the dequantized product need not
// agree in type yet. Multiplying them together would throw out of the callback rather than decline.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredWhenRescaleOperandTypesDisagree) {
    Chain chain;
    chain.dequantize_to = element::bf16;
    expect_decline(chain);
}

// f16 has neither f32's exponent range nor a bound derived here, so it is declined rather than
// assumed to hold an accumulator the deferral has enlarged.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredWhenAnchorCannotAbsorbTheGrowth) {
    Chain chain;
    chain.scores_type = element::f16;
    chain.dequantize_to = element::f16;
    expect_decline(chain);
}

// Each narrowing is checked against its own type's range, not one shared bound. This puts the
// operand past f16's range while leaving the rest of the chain admissible.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredWhenScaleOutgrowsAnF16Conversion) {
    Chain chain;
    chain.between_softmax_and_quantizer = [](const Output<Node>& x, ParameterVector&) -> Output<Node> {
        const auto gain = ov::op::v0::Constant::create(x.get_element_type(), Shape{}, {1e5F});
        const auto amplified = std::make_shared<ov::op::v1::Multiply>(x, gain);
        const auto narrowed = std::make_shared<ov::op::v0::Convert>(amplified, element::f16);
        return std::make_shared<ov::op::v0::Convert>(narrowed, x.get_element_type());
    };
    chain.input_high = 1e5F;
    chain.quantizer_scale = 1e-3F;
    expect_decline(chain);
}

// A cast to boolean is the predicate x != 0, not a narrowing of a magnitude: it maps the deferred
// operand and the normalized one to the same thing, so what follows no longer distinguishes them and
// the rescale would divide a result the operand never scaled. Its range trivially admits the operand
// at this scale, which is exactly why the range check alone must not be what decides it.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredAcrossABooleanConversion) {
    Chain chain;
    chain.between_softmax_and_quantizer = [](const Output<Node>& x, ParameterVector&) -> Output<Node> {
        const auto gain = ov::op::v0::Constant::create(x.get_element_type(), Shape{}, {2.F});
        const auto amplified = std::make_shared<ov::op::v1::Multiply>(x, gain);
        const auto narrowed = std::make_shared<ov::op::v0::Convert>(amplified, element::boolean);
        return std::make_shared<ov::op::v0::Convert>(narrowed, x.get_element_type());
    };
    chain.input_high = 2.F;
    chain.quantizer_scale = 100.F;
    expect_decline(chain);
}

// A conversion to a type whose range this does not reason about is declined even when a bound could
// be derived for it: u1 would admit the operand at this scale, and is still not crossed.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredAcrossAnUnreasonedType) {
    Chain chain;
    chain.between_softmax_and_quantizer = [](const Output<Node>& x, ParameterVector&) -> Output<Node> {
        const auto narrowed = std::make_shared<ov::op::v0::Convert>(x, element::u1);
        return std::make_shared<ov::op::v0::Convert>(narrowed, x.get_element_type());
    };
    expect_decline(chain);
}

// A dynamic dimension broadcast-merges with anything, so the shape check would hold vacuously.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredOnDynamicShapes) {
    Chain chain;
    chain.scores = PartialShape::dynamic(4);
    chain.values = PartialShape::dynamic(4);
    expect_decline(chain);
}

TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredPastF32Brgemm) {
    {
        auto scores = std::make_shared<ov::op::v0::Parameter>(element::f32, default_scores);
        auto values = std::make_shared<ov::op::v0::Parameter>(element::f32, default_values);
        auto softmax = std::make_shared<ov::op::v8::Softmax>(scores, -1);
        model = std::make_shared<Model>(OutputVector{make_brgemm(softmax, values)}, ParameterVector{scores, values});
    }
    {
        auto scores = std::make_shared<ov::op::v0::Parameter>(element::f32, default_scores);
        auto values = std::make_shared<ov::op::v0::Parameter>(element::f32, default_values);
        model_ref = std::make_shared<Model>(OutputVector{make_brgemm(make_normalized_softmax(scores, 3), values)},
                                            ParameterVector{scores, values});
    }
}

// Only operand 0 carries the probabilities, so the row sums do not correspond to a product whose
// operand 1 they came from.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredWhenProbabilitiesAreOnOperandOne) {
    const auto build = [](const Output<Node>& probabilities, const Output<Node>& other) {
        const auto quantized = make_quantizer(probabilities, 1.F, 127.F, element::i8);
        const auto brgemm = make_brgemm(other, quantized);
        return std::make_shared<ov::op::v0::Convert>(brgemm, element::f32);
    };
    {
        auto scores = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 8, 8});
        auto other = std::make_shared<ov::op::v0::Parameter>(element::u8, Shape{1, 2, 8, 8});
        auto softmax = std::make_shared<ov::op::v8::Softmax>(scores, -1);
        model = std::make_shared<Model>(OutputVector{build(softmax, other)}, ParameterVector{scores, other});
    }
    {
        auto scores = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 8, 8});
        auto other = std::make_shared<ov::op::v0::Parameter>(element::u8, Shape{1, 2, 8, 8});
        model_ref = std::make_shared<Model>(OutputVector{build(make_normalized_softmax(scores, 3), other)},
                                            ParameterVector{scores, other});
    }
}

// Rounding and narrowing to an integer are the two steps that do not commute with a scale at all.
// They are admissible here only as the quantization step itself, applied to an operand a scale has
// already spread over the target grid. Applied straight to values in (0, 1] they are a different
// function, not a differently quantized one: a bare rounding sends every normalized probability on
// a row of eight to 0 but the deferred row maximum to 1, and a bare narrowing to u8 truncates the
// same way. The range check cannot see either -- 1.0 fits u8 comfortably.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredAcrossAnUnscaledRounding) {
    Chain chain;
    chain.between_softmax_and_quantizer = [](const Output<Node>& x, ParameterVector&) -> Output<Node> {
        return std::make_shared<ov::op::v5::Round>(x, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
    };
    expect_decline(chain);
}

TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredAcrossAnUnscaledNarrowing) {
    Chain chain;
    chain.between_softmax_and_quantizer = [](const Output<Node>& x, ParameterVector&) -> Output<Node> {
        const auto narrowed = std::make_shared<ov::op::v0::Convert>(x, element::u8);
        return std::make_shared<ov::op::v0::Convert>(narrowed, x.get_element_type());
    };
    expect_decline(chain);
}

// A scale that shrinks the operand does not make a rounding admissible either: the grid it lands on
// is coarser than the operand's own range, not finer.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredAcrossARoundingBehindAShrinkingScale) {
    Chain chain;
    chain.between_softmax_and_quantizer = [](const Output<Node>& x, ParameterVector&) -> Output<Node> {
        const auto gain = ov::op::v0::Constant::create(x.get_element_type(), Shape{}, {0.5F});
        const auto shrunk = std::make_shared<ov::op::v1::Multiply>(x, gain);
        return std::make_shared<ov::op::v5::Round>(shrunk, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
    };
    expect_decline(chain);
}

// The normalized operand's row sums to the grid maximum however long the row is; the deferred one's
// grows with it. 8 rows of u8 codes against i8 weights cannot overflow i32, but 2^25 of them can, so
// the reduced dimension has to be part of the decision.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredWhenTheAccumulatorCannotHoldTheDeferredRow) {
    Chain chain;
    chain.scores = PartialShape{1, 1, 8, 1 << 25};
    chain.values = PartialShape{1, 1, 1 << 25, 4};
    expect_decline(chain);
}

TEST_F(SoftmaxDecompositionDeferredTests, DeferredWhenTheAccumulatorIsWideEnoughForTheRow) {
    Chain chain;
    chain.scores = PartialShape{1, 1, 8, 1 << 15};
    chain.values = PartialShape{1, 1, 1 << 15, 4};
    expect_defer(chain);
}

// FakeQuantizeDecomposition emits snippets' own saturating conversion, not ov::op::v0::Convert, so
// that is the narrowing the pass actually meets in production.
TEST_F(SoftmaxDecompositionDeferredTests, DeferredAcrossASaturatingNarrowing) {
    Chain chain;
    chain.saturating_narrowing = true;
    expect_defer(chain);
}

// bf16 shares f32's exponent range, so it can hold a product the deferral has enlarged.
TEST_F(SoftmaxDecompositionDeferredTests, DeferredWhenBothRescaleOperandsAreBf16) {
    Chain chain;
    chain.scores_type = element::bf16;
    chain.dequantize_to = element::bf16;
    expect_defer(chain);
}

// A scale on the numerator of a divide, and shifts by zero, all commute with the row sums.
TEST_F(SoftmaxDecompositionDeferredTests, DeferredAcrossADivideAndZeroShifts) {
    Chain chain;
    chain.between_softmax_and_quantizer = [](const Output<Node>& x, ParameterVector&) -> Output<Node> {
        const auto divisor = ov::op::v0::Constant::create(x.get_element_type(), Shape{}, {4.F});
        const auto divided = std::make_shared<ov::op::v1::Divide>(x, divisor);
        const auto zero = ov::op::v0::Constant::create(x.get_element_type(), Shape{}, {0.F});
        const auto shifted = std::make_shared<ov::op::v1::Add>(divided, zero);
        return std::make_shared<ov::op::v1::Subtract>(shifted, zero);
    };
    chain.input_high = 0.25F;
    chain.quantizer_scale = 1020.F;
    expect_defer(chain);
}

// The walk is bounded, so a chain of admissible steps longer than the bound is declined rather than
// followed. Seventeen zero shifts is one more than the bound allows.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredAcrossAChainLongerThanTheWalk) {
    Chain chain;
    chain.between_softmax_and_quantizer = [](const Output<Node>& x, ParameterVector&) -> Output<Node> {
        Output<Node> cursor = x;
        const auto zero = ov::op::v0::Constant::create(x.get_element_type(), Shape{}, {0.F});
        for (int i = 0; i < 17; ++i) {
            cursor = std::make_shared<ov::op::v1::Add>(cursor, zero);
        }
        return cursor;
    };
    expect_decline(chain);
}

// A scale this pass cannot read the extremes of is not a scale it can commute with: a sub-byte
// constant it will not decode, or a constant carrying NaN or an infinity. The narrowing to u4 ahead
// of the sub-byte scale is admissible on its own -- the operand has been spread over that grid and
// fits it -- so it is the constant's own width that declines.
TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredAcrossASubByteScale) {
    Chain chain;
    chain.between_softmax_and_quantizer = [](const Output<Node>& x, ParameterVector&) -> Output<Node> {
        const auto spread = ov::op::v0::Constant::create(x.get_element_type(), Shape{}, {8.F});
        const auto amplified = std::make_shared<ov::op::v1::Multiply>(x, spread);
        const auto narrowed = std::make_shared<ov::op::v0::Convert>(amplified, element::u4);
        const auto gain = ov::op::v0::Constant::create(element::u4, Shape{}, {2});
        const auto scaled = std::make_shared<ov::op::v1::Multiply>(narrowed, gain);
        return std::make_shared<ov::op::v0::Convert>(scaled, x.get_element_type());
    };
    expect_decline(chain);
}

TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredAcrossANaNScale) {
    Chain chain;
    chain.between_softmax_and_quantizer = [](const Output<Node>& x, ParameterVector&) -> Output<Node> {
        const auto gain =
            ov::op::v0::Constant::create(x.get_element_type(), Shape{}, {std::numeric_limits<float>::quiet_NaN()});
        return std::make_shared<ov::op::v1::Multiply>(x, gain);
    };
    expect_decline(chain);
}

TEST_F(SoftmaxDecompositionDeferredTests, NotDeferredAcrossAnInfiniteScale) {
    Chain chain;
    chain.between_softmax_and_quantizer = [](const Output<Node>& x, ParameterVector&) -> Output<Node> {
        const auto gain =
            ov::op::v0::Constant::create(x.get_element_type(), Shape{}, {std::numeric_limits<float>::infinity()});
        return std::make_shared<ov::op::v1::Multiply>(x, gain);
    };
    expect_decline(chain);
}

// The deferral is off unless a consumer asks for it, so the same chain that DeferredPastInt8Brgemm
// rewrites has to come out normalized in place under a default-constructed pass.
class SoftmaxDecompositionDefaultTests : public TransformationTestsF {
public:
    SoftmaxDecompositionDefaultTests() {
        comparator.enable(FunctionsComparator::CONST_VALUES);
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ov::snippets::pass::SoftmaxDecomposition>();
    }
};

TEST_F(SoftmaxDecompositionDefaultTests, NormalizesInPlaceOnADeferrableChain) {
    {
        auto scores = std::make_shared<ov::op::v0::Parameter>(element::f32, default_scores);
        auto values = std::make_shared<ov::op::v0::Parameter>(element::i8, default_values);
        auto softmax = std::make_shared<ov::op::v8::Softmax>(scores, -1);
        auto brgemm = make_brgemm(make_quantizer(softmax, 1.F, 255.F, element::u8), values);
        auto dequantized = std::make_shared<ov::op::v0::Convert>(brgemm, element::f32);
        model = std::make_shared<Model>(OutputVector{dequantized}, ParameterVector{scores, values});
    }
    {
        auto scores = std::make_shared<ov::op::v0::Parameter>(element::f32, default_scores);
        auto values = std::make_shared<ov::op::v0::Parameter>(element::i8, default_values);
        auto brgemm = make_brgemm(make_quantizer(make_normalized_softmax(scores, 3), 1.F, 255.F, element::u8), values);
        auto dequantized = std::make_shared<ov::op::v0::Convert>(brgemm, element::f32);
        model_ref = std::make_shared<Model>(OutputVector{dequantized}, ParameterVector{scores, values});
    }
}

// The opt-in is a caller's decision, not a property of the body, so init_config() cannot recover it
// the way it recovers the rest of SubgraphConfig. The CPU plugin clones the tokenized Subgraph
// before anything transforms it, so a flag that does not survive cloning is a flag no plugin can
// use -- and nothing else in the suite would notice, because the pass tests construct the pass
// directly.
TEST(SoftmaxDecompositionSubgraphConfig, DeferralSurvivesCloning) {
    auto inner = std::make_shared<ov::op::v0::Parameter>(element::f32, default_scores);
    auto softmax = std::make_shared<ov::op::v8::Softmax>(inner, -1);
    auto body = std::make_shared<Model>(OutputVector{softmax}, ParameterVector{inner});
    auto outer = std::make_shared<ov::op::v0::Parameter>(element::f32, default_scores);
    auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(OutputVector{outer}, body);

    EXPECT_FALSE(subgraph->defer_softmax_normalization());
    subgraph->set_defer_softmax_normalization(true);

    const auto cloned = subgraph->clone();
    const auto with_new_inputs =
        ov::as_type_ptr<ov::snippets::op::Subgraph>(subgraph->clone_with_new_inputs(OutputVector{outer}));
    ASSERT_TRUE(with_new_inputs);
    EXPECT_TRUE(cloned->defer_softmax_normalization());
    EXPECT_TRUE(with_new_inputs->defer_softmax_normalization());
}
