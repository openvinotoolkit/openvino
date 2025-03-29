// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/pass/manager.hpp>
#include <openvino/core/model.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/fake_quantize.hpp>
#include <openvino/op/binary_convolution.hpp>
#include <openvino/op/convolution.hpp>
#include <openvino/op/pad.hpp>
#include <plugin/transformations/binary_conv_to_conv.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/type/element_type.hpp"

using namespace testing;
using namespace ov::intel_gpu;

TEST_F(TransformationTestsF, ConvertBinaryConvolutionToConvolutionTest1) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{ 1, 256, 56, 56 });
        auto in_lo = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{ 1, 256, 1, 1 });
        auto in_hi = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{ 1, 256, 1, 1 });
        auto out_lo = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{ 1, 1, 1, 1 }, std::vector<float>{0.0f});
        auto out_hi = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{ 1, 1, 1, 1 }, std::vector<float>{1.0f});
        auto fq = std::make_shared<ov::op::v0::FakeQuantize>(input, in_lo, in_hi, out_lo, out_hi, 2);
        auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::u1, ov::Shape{ 32, 256, 3, 3 });
        auto binary_conv = std::make_shared<ov::op::v1::BinaryConvolution>(fq,
                                                                           weights,
                                                                           ov::Strides{1, 1},
                                                                           ov::CoordinateDiff{1, 1},
                                                                           ov::CoordinateDiff{1, 1},
                                                                           ov::Strides{1, 1},
                                                                           ov::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT,
                                                                           -1.0f);

        model = std::make_shared<ov::Model>(ov::NodeVector{ binary_conv }, ov::ParameterVector{ input });
        manager.register_pass<ConvertBinaryConvolutionToConvolution>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{ 1, 256, 56, 56 });
        auto in_lo = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{ 1, 256, 1, 1 });
        auto in_hi = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{ 1, 256, 1, 1 });
        auto out_lo = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{ 1, 1, 1, 1 }, std::vector<float>{-1.0f});
        auto out_hi = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{ 1, 1, 1, 1 }, std::vector<float>{1.0f});
        auto fq = std::make_shared<ov::op::v0::FakeQuantize>(input, in_lo, in_hi, out_lo, out_hi, 2);
        auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{ 32, 256, 3, 3 });

        auto pb = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{ 4 }, std::vector<int32_t>{0, 0, 1, 1});
        auto pe = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{ 4 }, std::vector<int32_t>{0, 0, 1, 1});
        auto pv = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{ }, std::vector<float>{1.0f});

        auto pad = std::make_shared<ov::op::v1::Pad>(fq, pb, pe, pv, ov::op::PadMode::CONSTANT);

        auto conv = std::make_shared<ov::op::v1::Convolution>(pad,
                                                              weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1});

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ conv }, ov::ParameterVector{ input });
    }
}
