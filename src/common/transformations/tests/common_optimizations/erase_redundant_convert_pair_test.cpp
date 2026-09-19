// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "transformations/common_optimizations/erase_redundant_convert_pair.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "openvino/pass/manager.hpp"

// Identity Convert (source dtype already matches destination) is bypassed.
TEST_F(TransformationTestsF, EraseRedundantConvertPairIdentity) {
    {
        auto x = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 4});
        auto cvt = std::make_shared<ov::op::v0::Convert>(x, ov::element::f32);
        auto relu = std::make_shared<ov::opset1::Relu>(cvt);
        model = std::make_shared<ov::Model>(relu, ov::ParameterVector{x}, "model");
    }
    manager.register_pass<ov::pass::EraseRedundantConvertPair>();
    {
        auto x = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 4});
        auto relu = std::make_shared<ov::opset1::Relu>(x);
        model_ref = std::make_shared<ov::Model>(relu, ov::ParameterVector{x}, "model_ref");
    }
}

// Round-trip pair wide -> Convert(narrow) -> Convert(wide): outer Convert is
// replaced with the original wide source.
TEST_F(TransformationTestsF, EraseRedundantConvertPairRoundTrip) {
    {
        auto x = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 4});
        auto narrow = std::make_shared<ov::op::v0::Convert>(x, ov::element::bf16);
        auto wide = std::make_shared<ov::op::v0::Convert>(narrow, ov::element::f32);
        auto relu = std::make_shared<ov::opset1::Relu>(wide);
        model = std::make_shared<ov::Model>(relu, ov::ParameterVector{x}, "model");
    }
    manager.register_pass<ov::pass::EraseRedundantConvertPair>();
    {
        auto x = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 4});
        auto relu = std::make_shared<ov::opset1::Relu>(x);
        model_ref = std::make_shared<ov::Model>(relu, ov::ParameterVector{x}, "model_ref");
    }
}

// Widening pair (narrow_type.bitwidth() >= src_type.bitwidth()) is not a
// round-trip and must be left untouched.
TEST_F(TransformationTestsF, EraseRedundantConvertPairNoMatchOnWideningPair) {
    const ov::Shape shape{1, 4};
    {
        auto x = std::make_shared<ov::opset1::Parameter>(ov::element::f16, shape);
        auto mid = std::make_shared<ov::op::v0::Convert>(x, ov::element::f32);
        auto wide = std::make_shared<ov::op::v0::Convert>(mid, ov::element::f64);
        auto relu = std::make_shared<ov::opset1::Relu>(wide);
        model = std::make_shared<ov::Model>(relu, ov::ParameterVector{x}, "model");
    }
    manager.register_pass<ov::pass::EraseRedundantConvertPair>();
    {
        auto x = std::make_shared<ov::opset1::Parameter>(ov::element::f16, shape);
        auto mid = std::make_shared<ov::op::v0::Convert>(x, ov::element::f32);
        auto wide = std::make_shared<ov::op::v0::Convert>(mid, ov::element::f64);
        auto relu = std::make_shared<ov::opset1::Relu>(wide);
        model_ref = std::make_shared<ov::Model>(relu, ov::ParameterVector{x}, "model_ref");
    }
}
