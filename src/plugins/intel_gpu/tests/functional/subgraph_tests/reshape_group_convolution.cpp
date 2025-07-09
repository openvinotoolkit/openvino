// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/opsets/opset13_decl.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "intel_gpu/runtime/engine.hpp"

// CreateReshapeOp() for 4d->5d reshape when allow_new_shape_infer=false causes creation of weight reorder with 4d input and 5d output.
// The weight reorder with 4d input and 5d output is not supported in onednn. And runtime error during model compile happens.
// CreateReshapeOp() needs to create just reshape for the case.
// This functional test checks that weights reorder should have same input/output shape.

namespace {
using GroupConvolutionReorderWeightTestParams = typename std::tuple<
    ov::Shape,
    ov::element::Type
    >;

class GroupConvolutionReorderWeightTest : virtual public ov::test::SubgraphBaseStaticTest,
                   public testing::WithParamInterface<GroupConvolutionReorderWeightTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GroupConvolutionReorderWeightTestParams> &obj) {
        ov::Shape input_shape;
        ov::element::Type model_type;
        std::tie(
            input_shape,
            model_type) = obj.param;

        std::ostringstream result;
        result << "IS=" << ov::test::utils::partialShape2str({input_shape}) << "_";
        result << "netType=" << model_type << "_";
        result << "targetDevice=GPU_";

        return result.str();
    }

private:
    ov::Shape input_shape;

protected:
    void create_model() {
        std::tie(input_shape, inType) = GetParam();
        targetDevice = ov::test::utils::DEVICE_GPU;
        auto constant1 = ov::op::v0::Constant::create(ov::element::u8, {32, 1, 3, 3},
                                                    {106, 107, 105, 109, 108, 107, 106, 105,
                                                     106, 107, 105, 109, 108, 107, 106, 105,
                                                     106, 107, 105, 109, 108, 107, 106, 105,
                                                     106, 107, 105, 109, 108, 107, 106, 105,

                                                     106, 107, 105, 109, 108, 107, 106, 105,
                                                     106, 107, 105, 109, 108, 107, 106, 105,
                                                     106, 107, 105, 109, 108, 107, 106, 105,
                                                     106, 107, 105, 109, 108, 107, 106, 105,

                                                     106, 107, 105, 109, 108, 107, 106, 105,
                                                     106, 107, 105, 109, 108, 107, 106, 105,
                                                     106, 107, 105, 109, 108, 107, 106, 105,
                                                     106, 107, 105, 109, 108, 107, 106, 105,

                                                     106, 107, 105, 109, 108, 107, 106, 105,
                                                     106, 107, 105, 109, 108, 107, 106, 105,
                                                     106, 107, 105, 109, 108, 107, 106, 105,
                                                     106, 107, 105, 109, 108, 107, 106, 105,

                                                     106, 107, 105, 109, 108, 107, 106, 105,
                                                     106, 107, 105, 109, 108, 107, 106, 105,
                                                     106, 107, 105, 109, 108, 107, 106, 105,
                                                     106, 107, 105, 109, 108, 107, 106, 105,

                                                     106, 107, 105, 109, 108, 107, 106, 105,
                                                     106, 107, 105, 109, 108, 107, 106, 105,
                                                     106, 107, 105, 109, 108, 107, 106, 105,
                                                     106, 107, 105, 109, 108, 107, 106, 105,

                                                     106, 107, 105, 109, 108, 107, 106, 105,
                                                     106, 107, 105, 109, 108, 107, 106, 105,
                                                     106, 107, 105, 109, 108, 107, 106, 105,
                                                     106, 107, 105, 109, 108, 107, 106, 105,

                                                     106, 107, 105, 109, 108, 107, 106, 105,
                                                     106, 107, 105, 109, 108, 107, 106, 105,
                                                     106, 107, 105, 109, 108, 107, 106, 105,
                                                     106, 107, 105, 109, 108, 107, 106, 105,

                                                     106, 107, 105, 109, 108, 107, 106, 105,
                                                     106, 107, 105, 109, 108, 107, 106, 105,
                                                     106, 107, 105, 109, 108, 107, 106, 105,
                                                     106, 107, 105, 109, 108, 107, 106, 105,
                                                    });
        auto constant2 = ov::op::v0::Constant::create(ov::element::u8, {}, {110});
        auto constant3 = ov::op::v0::Constant::create(inType, {1, 1, 1, 1}, {0.291028});
        auto constant4 = ov::op::v0::Constant::create(ov::element::i32, {5}, {32, 1, 1, 3, 3});

        auto input = std::make_shared<ov::op::v0::Parameter>(inType, input_shape);

        auto convert1 = std::make_shared<ov::op::v0::Convert>(constant1, inType);
        auto convert2 = std::make_shared<ov::op::v0::Convert>(constant2, inType);
        auto subtract = std::make_shared<ov::op::v1::Subtract>(convert1, convert2);
        auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, constant3);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(multiply, constant4, false);

        auto group_convolution = std::make_shared<ov::op::v1::GroupConvolution>(input, reshape,
                                                                            ov::Strides {1, 1},
                                                                            ov::CoordinateDiff {1, 1},
                                                                            ov::CoordinateDiff {1, 1},
                                                                            ov::Strides {1, 1});
        group_convolution->set_friendly_name("group_convolution");

        auto output = std::make_shared<ov::op::v0::Result>(group_convolution->output(0));
        function = std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{input}, "group_convolution_model");

        if (inType == ov::element::f16) {
            abs_threshold = 0.25;
            rel_threshold = 0.25;
        }
    }
};

TEST_P(GroupConvolutionReorderWeightTest, Inference) {
    create_model();
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolutionReorderWeightTest,
                         GroupConvolutionReorderWeightTest,
                         ::testing::Combine(testing::Values(ov::Shape{1, 32, 112, 112}),
                                            testing::Values(ov::element::f16)),
                         GroupConvolutionReorderWeightTest::getTestCaseName);
}  // namespace
