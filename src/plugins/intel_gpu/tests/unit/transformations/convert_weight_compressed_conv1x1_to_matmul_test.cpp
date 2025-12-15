// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <intel_gpu/op/placeholder.hpp>
#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/pass/manager.hpp>
#include <ov_ops/type_relaxed.hpp>
#include <plugin/transformations/convert_weight_compressed_conv1x1_to_matmul.hpp>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "openvino/opsets/opset3_decl.hpp"
#include "openvino/opsets/opset7_decl.hpp"
#include "transformations/rt_info/decompression.hpp"

using namespace testing;
using namespace ov::intel_gpu;

TEST_F(TransformationTestsF, ConvertWeightCompressedConv1x1ToMatmulTest1) {
    ov::Strides strides{1, 1};
    ov::Strides dilations{1, 1};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f16, ov::Shape{1, 1, 2, 10});
        auto transpose_constant1 = ov::opset1::Constant::create(ov::element::i32, ov::Shape{4}, {0, 3, 1, 2});
        auto transpose_constant2 = ov::opset1::Constant::create(ov::element::i32, ov::Shape{4}, {0, 2, 3, 1});
        auto input2 = ov::opset1::Constant::create(ov::element::i4, ov::Shape{15, 10, 1, 1}, {1});
        auto input2_convert = std::make_shared<ov::op::v0::Convert>(input2, ov::element::f16);
        auto input2_scale = ov::opset1::Constant::create(ov::element::f16, ov::Shape{15, 10, 1, 1}, {1});
        auto mul = std::make_shared<ov::opset1::Multiply>(input2_convert, input2_scale);
        auto transpose1 = std::make_shared<ov::opset1::Transpose>(input1, transpose_constant1);
        auto conv1x1 = std::make_shared<ov::opset1::Convolution>(transpose1, mul, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);
        auto transpose2 = std::make_shared<ov::opset1::Transpose>(conv1x1, transpose_constant2);

        model = std::make_shared<ov::Model>(ov::OutputVector{transpose2}, ov::ParameterVector{input1});
        manager.register_pass<ConvertWeightCompressedConv1x1ToMatmul>();
    }
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f16, ov::Shape{1, 1, 2, 10});
        auto input2 = ov::opset1::Constant::create(ov::element::i4, ov::Shape{15, 10}, {1});
        auto input2_convert = std::make_shared<ov::op::v0::Convert>(input2, ov::element::f16);
        auto input2_scale = ov::opset1::Constant::create(ov::element::f16, ov::Shape{15, 10}, {1});
        auto mul = std::make_shared<ov::opset1::Multiply>(input2_convert, input2_scale);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(input1, mul, false, true);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{matmul}, ov::ParameterVector{input1});
    }
}

// Checked blocked cases
TEST(TransformationTests, ConvertWeightCompressedConv1x1ToMatmulExceptionTest_conv3x3) {
    auto CreateConv = [&]() {
        ov::Strides strides{1, 1};
        ov::Strides dilations{1, 1};
        ov::CoordinateDiff pads_begin{1, 1};
        ov::CoordinateDiff pads_end{1, 1};
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f16, ov::Shape{1, 1, 2, 1});
        auto transpose_constant1 = ov::opset1::Constant::create(ov::element::i32, ov::Shape{4}, {0, 3, 1, 2});
        auto transpose_constant2 = ov::opset1::Constant::create(ov::element::i32, ov::Shape{4}, {0, 2, 3, 1});
        auto input2 = ov::opset1::Constant::create(ov::element::i4, ov::Shape{1, 1, 3, 3}, {1});
        auto input2_convert = std::make_shared<ov::op::v0::Convert>(input2, ov::element::f16);
        auto input2_scale = ov::opset1::Constant::create(ov::element::f16, ov::Shape{1, 1, 3, 3}, {1});
        auto mul = std::make_shared<ov::opset1::Multiply>(input2_convert, input2_scale);
        auto transpose1 = std::make_shared<ov::opset1::Transpose>(input1, transpose_constant1);
        auto conv3x3 = std::make_shared<ov::opset1::Convolution>(transpose1, mul, strides, pads_begin, pads_end, dilations, ov::op::PadType::EXPLICIT);
        auto transpose2 = std::make_shared<ov::opset1::Transpose>(conv3x3, transpose_constant2);

        auto model = std::make_shared<ov::Model>(ov::OutputVector{transpose2}, ov::ParameterVector{input1});
        return model;
    };

    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<ov::intel_gpu::ConvertWeightCompressedConv1x1ToMatmul>();

    auto func = CreateConv();

    manager.run_passes(func);

    bool success = false;
    for (auto& ops : func->get_ops()) {
        std::string type_name(ops->get_type_name());
        if (type_name.find("MatMul") != std::string::npos) {
            success = true;
            break;
        }
    }
    ASSERT_TRUE(success == false);
}
