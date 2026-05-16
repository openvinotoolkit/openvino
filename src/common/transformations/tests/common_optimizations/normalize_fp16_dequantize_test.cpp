// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/normalize_fp16_dequantize.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <optional>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace testing;
using namespace ov;

// ---------------------------------------------------------------------------
// Helper: build an FP16-dequantize chain
//   FQ(data, il, ih, ol, oh) -> Conv1(->int_type) -> Conv2(->f16)
//     -> [Subtract(zp_f16)]  -> Multiply(scale_f16)
// ---------------------------------------------------------------------------
static std::shared_ptr<Model> build_fp16_dq_model(const Shape& data_shape,
                                                   float il,
                                                   float ih,
                                                   float ol,
                                                   float oh,
                                                   size_t levels,
                                                   element::Type int_type,
                                                   float scale_f16_val,
                                                   std::optional<float> zp_f16_val = std::nullopt) {
    auto data = std::make_shared<opset1::Parameter>(element::f32, data_shape);
    auto fq_il = opset1::Constant::create(element::f32, Shape{}, {il});
    auto fq_ih = opset1::Constant::create(element::f32, Shape{}, {ih});
    auto fq_ol = opset1::Constant::create(element::f32, Shape{}, {ol});
    auto fq_oh = opset1::Constant::create(element::f32, Shape{}, {oh});
    auto fq = std::make_shared<opset1::FakeQuantize>(data, fq_il, fq_ih, fq_ol, fq_oh, levels);
    auto conv1 = std::make_shared<opset1::Convert>(fq, int_type);
    auto conv2 = std::make_shared<opset1::Convert>(conv1, element::f16);

    Output<Node> dq = conv2;
    if (zp_f16_val) {
        auto zp = opset1::Constant::create(element::f16, Shape{}, {ov::float16(*zp_f16_val)});
        dq = std::make_shared<opset1::Subtract>(dq, zp);
    }
    auto scale = opset1::Constant::create(element::f16, Shape{}, {ov::float16(scale_f16_val)});
    auto mul = std::make_shared<opset1::Multiply>(dq, scale);
    return std::make_shared<Model>(OutputVector{mul}, ParameterVector{data});
}

// ---------------------------------------------------------------------------
// Helper: build the reference model produced by NormalizeDequantizeFP16
//   FQ(data, il, ih, ol, oh) -> Conv1(->int_type) -> Conv2(->f32)
//     -> [Subtract(zp_f32)]  -> Multiply(scale_f32)  -> Convert(->f16)
//
// The FQ node is left unchanged -- il/ih/ol/oh are the same as the input model.
// ---------------------------------------------------------------------------
static std::shared_ptr<Model> build_fp32_dq_ref_model(const Shape& data_shape,
                                                       float il,
                                                       float ih,
                                                       float ol,
                                                       float oh,
                                                       size_t levels,
                                                       element::Type int_type,
                                                       float scale_f32,
                                                       std::optional<float> zp_f32 = std::nullopt) {
    auto data = std::make_shared<opset1::Parameter>(element::f32, data_shape);

    auto fq_il = opset1::Constant::create(element::f32, Shape{}, {il});
    auto fq_ih = opset1::Constant::create(element::f32, Shape{}, {ih});
    auto fq_ol = opset1::Constant::create(element::f32, Shape{}, {ol});
    auto fq_oh = opset1::Constant::create(element::f32, Shape{}, {oh});
    auto fq = std::make_shared<opset1::FakeQuantize>(data, fq_il, fq_ih, fq_ol, fq_oh, levels);
    auto conv1 = std::make_shared<opset1::Convert>(fq, int_type);
    auto conv2 = std::make_shared<opset1::Convert>(conv1, element::f32);

    Output<Node> dq = conv2;
    if (zp_f32) {
        auto zp_cst = opset1::Constant::create(element::f32, Shape{}, {*zp_f32});
        dq = std::make_shared<opset1::Subtract>(dq, zp_cst);
    }
    auto scale_cst = opset1::Constant::create(element::f32, Shape{}, {scale_f32});
    auto mul = std::make_shared<opset1::Multiply>(dq, scale_cst);
    auto cast_f16 = std::make_shared<opset1::Convert>(mul, element::f16);
    return std::make_shared<Model>(OutputVector{cast_f16}, ParameterVector{data});
}

// ---------------------------------------------------------------------------
// Test 1: u16 quantization, no zero point
//
//   FQ(il=-1, ih=1, ol=0, oh=65535, levels=65536)
//     -> Convert(u16) -> Convert(f16) -> Multiply(Const[f16, 0.25])
//
// Expected after pass (FQ unchanged):
//   FQ(il=-1, ih=1, ol=0, oh=65535, levels=65536)
//     -> Convert(u16) -> Convert(f32) -> Multiply(Const[f32, 0.25])
//     -> Convert(f16)
// ---------------------------------------------------------------------------
TEST_F(TransformationTestsF, NormalizeDequantizeFP16_U16_NoZeroPoint) {
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);

    const float scale_f16_as_f32 = float(ov::float16(0.25f));  // exactly 0.25

    model = build_fp16_dq_model({1, 3, 32, 32}, -1.0f, 1.0f, 0.0f, 65535.0f, 65536, element::u16, 0.25f);
    manager.register_pass<ov::pass::NormalizeDequantizeFP16>();

    model_ref =
        build_fp32_dq_ref_model({1, 3, 32, 32}, -1.0f, 1.0f, 0.0f, 65535.0f, 65536, element::u16, scale_f16_as_f32);
}

// ---------------------------------------------------------------------------
// Test 2: u16 quantization, with zero point
//
//   FQ(il=-1, ih=1, ol=0, oh=65535, levels=65536)
//     -> Convert(u16) -> Convert(f16) -> Subtract(Const[f16, 8])
//     -> Multiply(Const[f16, 0.25])
//
// Expected after pass (FQ unchanged):
//   FQ(il=-1, ih=1, ol=0, oh=65535)
//     -> Convert(u16) -> Convert(f32) -> Subtract(Const[f32, 8])
//     -> Multiply(Const[f32, 0.25]) -> Convert(f16)
// ---------------------------------------------------------------------------
TEST_F(TransformationTestsF, NormalizeDequantizeFP16_U16_WithZeroPoint) {
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);

    const float scale_f16_as_f32 = float(ov::float16(0.25f));  // exactly 0.25
    const float zp_f16_as_f32 = float(ov::float16(8.0f));      // exactly 8.0

    model =
        build_fp16_dq_model({1, 3, 32, 32}, -1.0f, 1.0f, 0.0f, 65535.0f, 65536, element::u16, 0.25f, zp_f16_as_f32);
    manager.register_pass<ov::pass::NormalizeDequantizeFP16>();

    model_ref = build_fp32_dq_ref_model({1, 3, 32, 32},
                                        -1.0f,
                                        1.0f,
                                        0.0f,
                                        65535.0f,
                                        65536,
                                        element::u16,
                                        scale_f16_as_f32,
                                        zp_f16_as_f32);
}

// ---------------------------------------------------------------------------
// Test 3: i8 quantization, no zero point
//
//   FQ(il=-1, ih=1, ol=-128, oh=127, levels=256)
//     -> Convert(i8) -> Convert(f16) -> Multiply(Const[f16, 0.5])
//
// Expected after pass (FQ unchanged):
//   FQ(il=-1, ih=1, ol=-128, oh=127)
//     -> Convert(i8) -> Convert(f32) -> Multiply(Const[f32, 0.5]) -> Convert(f16)
// ---------------------------------------------------------------------------
TEST_F(TransformationTestsF, NormalizeDequantizeFP16_I8_NoZeroPoint) {
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);

    const float scale_f16_as_f32 = float(ov::float16(0.5f));  // exactly 0.5

    model = build_fp16_dq_model({1, 3, 32, 32}, -1.0f, 1.0f, -128.0f, 127.0f, 256, element::i8, 0.5f);
    manager.register_pass<ov::pass::NormalizeDequantizeFP16>();

    model_ref =
        build_fp32_dq_ref_model({1, 3, 32, 32}, -1.0f, 1.0f, -128.0f, 127.0f, 256, element::i8, scale_f16_as_f32);
}

// ---------------------------------------------------------------------------
// Negative test: pass must NOT fire when Conv2 already outputs f32
//   FQ -> Convert(u16) -> Convert(f32) -> Multiply(scale_f32)
// ---------------------------------------------------------------------------
TEST_F(TransformationTestsF, NormalizeDequantizeFP16_NoEffect_WhenAlreadyF32) {
    auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 3, 32, 32});
    auto fq_il = opset1::Constant::create(element::f32, Shape{}, {-1.0f});
    auto fq_ih = opset1::Constant::create(element::f32, Shape{}, {1.0f});
    auto fq_ol = opset1::Constant::create(element::f32, Shape{}, {0.0f});
    auto fq_oh = opset1::Constant::create(element::f32, Shape{}, {65535.0f});
    auto fq = std::make_shared<opset1::FakeQuantize>(data, fq_il, fq_ih, fq_ol, fq_oh, 65536);
    auto conv1 = std::make_shared<opset1::Convert>(fq, element::u16);
    auto conv2 = std::make_shared<opset1::Convert>(conv1, element::f32);  // f32, not f16
    auto scale = opset1::Constant::create(element::f32, Shape{}, {0.25f});
    auto mul = std::make_shared<opset1::Multiply>(conv2, scale);
    model = std::make_shared<Model>(OutputVector{mul}, ParameterVector{data});

    // model_ref is not set -> TearDown clones model before pass and compares;
    // an unchanged model means the pass correctly had no effect.
    manager.register_pass<ov::pass::NormalizeDequantizeFP16>();
}
