// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Unit tests for the half-precision sibling exception in
// ov::frontend::tensorflow_lite::dequantize_inputs.
//
// Default behavior dequantizes TFLQuantize-wrapped inputs to f32. The narrow
// exception switches the dequant target to f16/bf16 when:
//   1) the TFLQuantize wraps a Constant (i.e. a quantized weight), AND
//   2) some non-quantized sibling input is f16 or bf16 (a half-precision
//      activation produced earlier in the graph).
// Non-constant TFLQuantize (i.e. quantized activations destined for the
// FakeQuantize lowering path) must remain at f32 to preserve the assumptions
// of downstream Low Precision Transformations.

#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "openvino/core/node_output.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/tensorflow_lite/quantization_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
// Relative include into the frontend's private headers. We intentionally avoid
// adding the FE's private src/ to the test target's include path because doing
// so would shadow the shared utils.hpp used by other test sources.
#include "../src/tflite_ops/tflite_quantize.hpp"

// Forward-declare to avoid pulling in utils.hpp, which transitively includes the
// flatbuffer schema headers that are private to the frontend's build.
namespace ov {
namespace frontend {
namespace tensorflow_lite {
TENSORFLOW_LITE_FRONTEND_API void dequantize_inputs(OutputVector& deq_inputs);
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov

using namespace ov;
using ov::frontend::tensorflow_lite::QuantizationInfo;
using ov::frontend::tensorflow_lite::TFLQuantize;
using ov::frontend::tensorflow_lite::dequantize_inputs;

namespace {

std::shared_ptr<QuantizationInfo> make_per_tensor_quant_info() {
    return std::make_shared<QuantizationInfo>(std::vector<float>{0.5f},
                                              std::vector<int64_t>{0},
                                              /*axis=*/0);
}

// Builds a TFLQuantize wrapping either a Constant (weight path) or a Parameter
// (activation path). Returns the TFLQuantize output.
Output<Node> make_tfl_quantize(bool weight_is_constant, element::Type storage_type) {
    std::shared_ptr<Node> producer;
    if (weight_is_constant) {
        // Use a 4x4 i8 weight; the exact dtype isn't checked by dequantize_inputs,
        // only that the producer is a Constant.
        producer = op::v0::Constant::create(storage_type, Shape{4, 4}, std::vector<int8_t>(16, 1));
    } else {
        producer = std::make_shared<op::v0::Parameter>(storage_type, PartialShape{1, 16});
    }
    return std::make_shared<TFLQuantize>(producer->output(0), make_per_tensor_quant_info(), storage_type)->output(0);
}

// Runs dequantize_inputs on a [sibling, tfl_quantize] OutputVector and returns
// the inserted Convert's destination type. Asserts that a Convert was actually
// inserted at the TFLQuantize slot.
element::Type run_dequant(bool weight_is_constant,
                          element::Type weight_storage_type,
                          element::Type sibling_type) {
    auto sibling = std::make_shared<op::v0::Parameter>(sibling_type, PartialShape{1, 16})->output(0);
    auto tfl_q = make_tfl_quantize(weight_is_constant, weight_storage_type);

    OutputVector inputs{sibling, tfl_q};
    dequantize_inputs(inputs);

    auto convert = ov::as_type_ptr<op::v0::Convert>(inputs[1].get_node_shared_ptr());
    EXPECT_TRUE(convert) << "dequantize_inputs did not insert a Convert at the TFLQuantize slot";
    if (!convert)
        return element::dynamic;
    return convert->get_destination_type();
}

}  // namespace

// ----- Constant weight + half-precision sibling: exception fires -----

TEST(TFLDequantizeInputs, ConstantWeightWithF16SiblingDequantsToF16) {
    EXPECT_EQ(run_dequant(/*weight_is_constant=*/true,
                          /*weight_storage_type=*/element::i4,
                          /*sibling_type=*/element::f16),
              element::f16);
}

TEST(TFLDequantizeInputs, ConstantWeightWithBF16SiblingDequantsToBF16) {
    EXPECT_EQ(run_dequant(/*weight_is_constant=*/true,
                          /*weight_storage_type=*/element::i8,
                          /*sibling_type=*/element::bf16),
              element::bf16);
}

// ----- Backward-compat: f32 sibling (or no sibling) must keep f32 -----

TEST(TFLDequantizeInputs, ConstantWeightWithF32SiblingStaysF32) {
    EXPECT_EQ(run_dequant(/*weight_is_constant=*/true,
                          /*weight_storage_type=*/element::i8,
                          /*sibling_type=*/element::f32),
              element::f32);
}

TEST(TFLDequantizeInputs, ConstantWeightNoSiblingStaysF32) {
    // Only the TFLQuantize-wrapped weight, no sibling input at all.
    auto tfl_q = make_tfl_quantize(/*weight_is_constant=*/true, element::i8);
    OutputVector inputs{tfl_q};
    dequantize_inputs(inputs);

    auto convert = ov::as_type_ptr<op::v0::Convert>(inputs[0].get_node_shared_ptr());
    ASSERT_TRUE(convert);
    EXPECT_EQ(convert->get_destination_type(), element::f32);
}

// ----- Non-constant (quantized activation) must NEVER get retyped -----
// This is the key safety property: the FakeQuantize lowering branch of
// TFLQuantizeReplacer feeds LPT, which has f32 assumptions.

TEST(TFLDequantizeInputs, NonConstantActivationWithF16SiblingStaysF32) {
    EXPECT_EQ(run_dequant(/*weight_is_constant=*/false,
                          /*weight_storage_type=*/element::i8,
                          /*sibling_type=*/element::f16),
              element::f32);
}

TEST(TFLDequantizeInputs, NonConstantActivationWithBF16SiblingStaysF32) {
    EXPECT_EQ(run_dequant(/*weight_is_constant=*/false,
                          /*weight_storage_type=*/element::i8,
                          /*sibling_type=*/element::bf16),
              element::f32);
}

// ----- Mixed-input ordering shouldn't matter: half_sibling search must be
// independent of input position relative to the TFLQuantize. -----

TEST(TFLDequantizeInputs, F16SiblingDetectedRegardlessOfOrdering) {
    // TFLQuantize comes first, f16 sibling second.
    auto tfl_q = make_tfl_quantize(/*weight_is_constant=*/true, element::i4);
    auto sibling = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 16})->output(0);
    OutputVector inputs{tfl_q, sibling};
    dequantize_inputs(inputs);

    auto convert = ov::as_type_ptr<op::v0::Convert>(inputs[0].get_node_shared_ptr());
    ASSERT_TRUE(convert);
    EXPECT_EQ(convert->get_destination_type(), element::f16);
    // Sibling slot must be untouched.
    EXPECT_EQ(inputs[1].get_element_type(), element::f16);
    EXPECT_EQ(inputs[1].get_node_shared_ptr(), sibling.get_node_shared_ptr());
}
