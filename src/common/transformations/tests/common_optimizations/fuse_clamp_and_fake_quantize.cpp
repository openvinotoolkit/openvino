// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_clamp_and_fake_quantize.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <tuple>
#include <utility>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/opsets/opset8_decl.hpp"

namespace ov::test {

using FuseClampAndFakeQuantizeParams = std::tuple<std::pair<float, float>,  // clamp range
                                                  std::pair<float, float>,  // fq input range
                                                  bool>;                    // whether Clamp is expected to be fused

class FuseClampAndFakeQuantizeTestP : public testing::WithParamInterface<FuseClampAndFakeQuantizeParams>,
                                      public TransformationTestsF {};

TEST_P(FuseClampAndFakeQuantizeTestP, CompareFunctions) {
    const auto& [clamp_range, fq_range, should_transform] = GetParam();
    const Shape data_shape{1, 3, 8, 8};

    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, data_shape);
        auto clamp = std::make_shared<op::v0::Clamp>(data, clamp_range.first, clamp_range.second);
        auto input_low = op::v0::Constant::create(element::f32, Shape{1}, {fq_range.first});
        auto input_high = op::v0::Constant::create(element::f32, Shape{1}, {fq_range.second});
        auto output_low = op::v0::Constant::create(element::f32, Shape{1}, {0.f});
        auto output_high = op::v0::Constant::create(element::f32, Shape{1}, {255.f});
        auto fq = std::make_shared<op::v0::FakeQuantize>(clamp, input_low, input_high, output_low, output_high, 256);

        model = std::make_shared<Model>(OutputVector{fq}, ParameterVector{data});
        manager.register_pass<ov::pass::FuseClampAndFakeQuantize>();
    }

    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, data_shape);
        std::shared_ptr<Node> fq_input = data;
        if (!should_transform) {
            fq_input = std::make_shared<op::v0::Clamp>(fq_input, clamp_range.first, clamp_range.second);
        }

        auto input_low = op::v0::Constant::create(element::f32, Shape{1}, {fq_range.first});
        auto input_high = op::v0::Constant::create(element::f32, Shape{1}, {fq_range.second});
        auto output_low = op::v0::Constant::create(element::f32, Shape{1}, {0.f});
        auto output_high = op::v0::Constant::create(element::f32, Shape{1}, {255.f});
        auto fq = std::make_shared<op::v0::FakeQuantize>(fq_input, input_low, input_high, output_low, output_high, 256);

        model_ref = std::make_shared<Model>(OutputVector{fq}, ParameterVector{data});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         FuseClampAndFakeQuantizeTestP,
                         ::testing::Values(FuseClampAndFakeQuantizeParams({0.f, 10.f}, {1.f, 4.f}, true),
                                           FuseClampAndFakeQuantizeParams({1.f, 4.f}, {1.f, 4.f}, true),
                                           FuseClampAndFakeQuantizeParams({0.f, 2.f}, {1.f, 4.f}, false),
                                           FuseClampAndFakeQuantizeParams({2.f, 8.f}, {1.f, 4.f}, false),
                                           FuseClampAndFakeQuantizeParams({1.f, 4.f}, {0.f, 10.f}, false)));

}  // namespace ov::test