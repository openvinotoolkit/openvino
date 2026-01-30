// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

namespace ov {
namespace test {

class InterpolateWithPostOps : public SubgraphBaseStaticTest, public ::testing::WithParamInterface<bool> {
public:
    static std::string getTestCaseName(const ::testing::TestParamInfo<bool>& info) {
        return info.param ? "Interpolate_NoFuse_NCHWAsNHWC" : "Interpolate_Fuse_DefaultAxes";
    }

protected:
    bool NCHWAsNHWC_NoFuse = false;
    void SetUp() override {
        NCHWAsNHWC_NoFuse = GetParam();
        ov::element::Type netPrecision = ov::element::f32;
        targetDevice = ov::test::utils::DEVICE_CPU;

        std::shared_ptr<ov::Model> raw_function;
        if (NCHWAsNHWC_NoFuse) {
            auto input_shape = ov::Shape{1, 3, 128, 128};
            auto mul_const_shape = ov::Shape{1, 1, 1, 128};
            auto add_const_shape = ov::Shape{1, 1, 1, 128};
            auto input = std::make_shared<ov::op::v0::Parameter>(netPrecision, input_shape);
            auto sizes = ov::op::v0::Constant::create(ov::element::i64, {2}, {256, 256});
            auto axes = ov::op::v0::Constant::create(ov::element::i64, {2}, {1, 2});
            auto interpolate = std::make_shared<ov::op::v11::Interpolate>(
                input,
                sizes,
                axes,
                ov::op::v11::Interpolate::InterpolateAttrs{
                    ov::op::v11::Interpolate::InterpolateMode::BILINEAR_PILLOW,
                    ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
                    {0, 0, 0, 0},
                    {0, 0, 0, 0},
                    ov::op::v11::Interpolate::CoordinateTransformMode::HALF_PIXEL,
                    ov::op::v11::Interpolate::NearestMode::FLOOR,
                    false,
                    -0.75f});
            auto mul_const = ov::op::v0::Constant::create(netPrecision, mul_const_shape, {1.0f});
            auto add_const = ov::op::v0::Constant::create(netPrecision, add_const_shape, {2.0f});
            auto mul = std::make_shared<ov::op::v1::Multiply>(interpolate, mul_const);
            auto add = std::make_shared<ov::op::v1::Add>(mul, add_const);
            auto result = std::make_shared<ov::op::v0::Result>(add);
            raw_function = std::make_shared<ov::Model>(result,
                                                       ov::ParameterVector{input},
                                                       "Interpolate_with_post_ops_NoFuse_NCHWAsNHWC");
        } else {
            auto input_shape = ov::Shape{1, 3, 128, 128};
            auto mul_const_shape = ov::Shape{1, 3, 1, 1};
            auto add_const_shape = ov::Shape{1, 3, 1, 1};
            auto input = std::make_shared<ov::op::v0::Parameter>(netPrecision, input_shape);
            auto sizes = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, 3, 256, 128});
            auto interpolate = std::make_shared<ov::op::v11::Interpolate>(
                input,
                sizes,
                ov::op::v11::Interpolate::InterpolateAttrs{
                    ov::op::v11::Interpolate::InterpolateMode::LINEAR_ONNX,
                    ov::op::v11::Interpolate::ShapeCalcMode::SIZES,
                    {0, 0, 0, 0},
                    {0, 0, 0, 0},
                    ov::op::v11::Interpolate::CoordinateTransformMode::HALF_PIXEL,
                    ov::op::v11::Interpolate::NearestMode::FLOOR,
                    false,
                    -0.75f});
            auto mul_const = ov::op::v0::Constant::create(netPrecision, mul_const_shape, {1.0f});
            auto add_const = ov::op::v0::Constant::create(netPrecision, add_const_shape, {2.0f});
            auto mul = std::make_shared<ov::op::v1::Multiply>(interpolate, mul_const);
            auto add = std::make_shared<ov::op::v1::Add>(mul, add_const);
            auto result = std::make_shared<ov::op::v0::Result>(add);
            raw_function = std::make_shared<ov::Model>(result,
                                                       ov::ParameterVector{input},
                                                       "Interpolate_with_post_ops_Fuse_DefaultAxes");
        }
        auto ppp_model = ov::preprocess::PrePostProcessor(raw_function);
        ppp_model.input().tensor().set_layout("NHWC");
        function = ppp_model.build();
    }
};

TEST_P(InterpolateWithPostOps, CheckInterpolateWithPostOps) {
    run();
    if (NCHWAsNHWC_NoFuse) {
        CPUTestUtils::CheckNumberOfNodesWithTypes(compiledModel, {"Subgraph", "Eltwise"}, 1);
    } else {
        CPUTestUtils::CheckNumberOfNodesWithTypes(compiledModel, {"Subgraph", "Eltwise"}, 0);
    }
}

INSTANTIATE_TEST_SUITE_P(InterpolateWithPostOpsFusion,
                         InterpolateWithPostOps,
                         ::testing::Values(true, false),
                         InterpolateWithPostOps::getTestCaseName);

}  // namespace test
}  // namespace ov