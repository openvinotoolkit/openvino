// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"

#include "openvino/core/model.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "intel_gpu/op/deconvolution.hpp"
#include "intel_gpu/op/placeholder.hpp"

#include "plugin/transformations/convert_deconvolution.hpp"
#include "plugin/transformations/utils.hpp"

#include <memory>

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

TEST_F(TransformationTestsF, ConvertDeconvolutionToInternal_1) {
    ov::Strides strides{1, 1};
    ov::Strides dilations{1, 1};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    ov::CoordinateDiff output_padding{0, 0};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ 2, 1, 11, 13});
        auto weights_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 1, 1, 3, 3 }, { 1 });
        auto deconv = std::make_shared<ov::op::v1::ConvolutionBackpropData>(input,
                                                              weights_const,
                                                              strides,
                                                              pads_begin,
                                                              pads_end,
                                                              dilations,
                                                              ov::op::PadType::EXPLICIT);

        model = std::make_shared<ov::Model>(ov::NodeVector{ deconv }, ov::ParameterVector{ input });
        manager.register_pass<ConvertDeconvolutionToInternal>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{  2, 1, 11, 13 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 1, 1, 3, 3 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto deconv = std::make_shared<ov::intel_gpu::op::Deconvolution>(input,
                                                                     weights_const,
                                                                     no_bias,
                                                                     strides,
                                                                     pads_begin,
                                                                     pads_end,
                                                                     dilations,
                                                                     -1,
                                                                     ov::op::PadType::EXPLICIT,
                                                                     ov::element::f32,
                                                                     output_padding);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ deconv }, ov::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, ConvertDeconvolutionToInternal_2) {
    ov::Strides strides{1, 1};
    ov::Strides dilations{1, 1};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    ov::CoordinateDiff output_padding{0, 0};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ 2, 3, 11, 12 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 3, 4, 3, 3 }, { 1 });
        auto deconv = std::make_shared<ov::op::v1::ConvolutionBackpropData>(input,
                                                              weights_const,
                                                              strides,
                                                              pads_begin,
                                                              pads_end,
                                                              dilations,
                                                              ov::op::PadType::EXPLICIT);

        model = std::make_shared<ov::Model>(ov::NodeVector{ deconv }, ov::ParameterVector{ input });
        manager.register_pass<ConvertDeconvolutionToInternal>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{  2, 3, 11, 12 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 3, 4, 3, 3 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto deconv = std::make_shared<ov::intel_gpu::op::Deconvolution>(input,
                                                                     weights_const,
                                                                     no_bias,
                                                                     strides,
                                                                     pads_begin,
                                                                     pads_end,
                                                                     dilations,
                                                                     -1,
                                                                     ov::op::PadType::EXPLICIT,
                                                                     ov::element::f16,
                                                                     output_padding);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ deconv }, ov::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, ConvertDeconvolutionToInternal_3) {
    ov::Strides strides{1, 1};
    ov::Strides dilations{1, 1};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    ov::CoordinateDiff output_padding{0, 0};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ 2, 3, 11, 12 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 3, 4, 3, 3 }, { 1 });
        auto output_shape = ov::op::v0::Constant::create(ov::element::i16, ov::Shape{ 2 }, { 1 });
        auto deconv = std::make_shared<ov::op::v1::ConvolutionBackpropData>(input,
                                                              weights_const,
                                                              output_shape,
                                                              strides,
                                                              pads_begin,
                                                              pads_end,
                                                              dilations,
                                                              ov::op::PadType::EXPLICIT);

        model = std::make_shared<ov::Model>(ov::NodeVector{ deconv }, ov::ParameterVector{ input });
        manager.register_pass<ConvertDeconvolutionToInternal>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{  2, 3, 11, 12 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 3, 4, 3, 3 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto output_shape = ov::op::v0::Constant::create(ov::element::i16, ov::Shape{ 2 }, { 1 });
        auto deconv = std::make_shared<ov::intel_gpu::op::Deconvolution>(input,
                                                                     weights_const,
                                                                     no_bias,
                                                                     output_shape,
                                                                     strides,
                                                                     pads_begin,
                                                                     pads_end,
                                                                     dilations,
                                                                     -1,
                                                                     ov::op::PadType::EXPLICIT,
                                                                     ov::element::f16,
                                                                     output_padding);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ deconv }, ov::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, ConvertGroupConvolutionBackpropDataToInternal_1) {
    ov::Strides strides{1, 1};
    ov::Strides dilations{1, 1};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    ov::CoordinateDiff output_padding{0, 0};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ 6, 10, 11, 12});
        auto weights_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 10, 1, 2, 3, 3}, { 1 });
        auto deconv = std::make_shared<ov::op::v1::GroupConvolutionBackpropData>(input,
                                                                   weights_const,
                                                                   strides,
                                                                   pads_begin,
                                                                   pads_end,
                                                                   dilations,
                                                                   ov::op::PadType::EXPLICIT);

        model = std::make_shared<ov::Model>(ov::NodeVector{ deconv }, ov::ParameterVector{ input });
        manager.register_pass<ConvertDeconvolutionToInternal>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ 6, 10, 11, 12});
        auto weights_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 10, 1, 2, 3, 3}, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto deconv = std::make_shared<ov::intel_gpu::op::Deconvolution>(input,
                                                                     weights_const,
                                                                     no_bias,
                                                                     strides,
                                                                     pads_begin,
                                                                     pads_end,
                                                                     dilations,
                                                                     10,
                                                                     ov::op::PadType::EXPLICIT,
                                                                     ov::element::f32,
                                                                     output_padding);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ deconv }, ov::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, ConvertGroupConvolutionBackpropDataToInternal_2) {
    ov::Strides strides{1, 1};
    ov::Strides dilations{1, 1};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    ov::CoordinateDiff output_padding{0, 0};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ 2, 8, 11, 12 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 8, 1, 4, 3, 3 }, { 1 });
        auto deconv = std::make_shared<ov::op::v1::GroupConvolutionBackpropData>(input,
                                                                   weights_const,
                                                                   strides,
                                                                   pads_begin,
                                                                   pads_end,
                                                                   dilations,
                                                                   ov::op::PadType::EXPLICIT);

        model = std::make_shared<ov::Model>(ov::NodeVector{ deconv }, ov::ParameterVector{ input });
        manager.register_pass<ConvertDeconvolutionToInternal>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ 2, 8, 11, 12 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 8, 1, 4, 3, 3 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto deconv = std::make_shared<ov::intel_gpu::op::Deconvolution>(input,
                                                                     weights_const,
                                                                     no_bias,
                                                                     strides,
                                                                     pads_begin,
                                                                     pads_end,
                                                                     dilations,
                                                                     8,
                                                                     ov::op::PadType::EXPLICIT,
                                                                     ov::element::f16,
                                                                     output_padding);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ deconv }, ov::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, ConvertGroupConvolutionBackpropDataToInternal_3) {
    ov::Strides strides{1, 1};
    ov::Strides dilations{1, 1};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    ov::CoordinateDiff output_padding{0, 0};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ 2, 8, 11, 12 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 8, 1, 4, 3, 3 }, { 1 });
        auto output_shape = ov::op::v0::Constant::create(ov::element::i16, ov::Shape{ 2 }, { 1 });
        auto deconv = std::make_shared<ov::op::v1::GroupConvolutionBackpropData>(input,
                                                                   weights_const,
                                                                   output_shape,
                                                                   strides,
                                                                   pads_begin,
                                                                   pads_end,
                                                                   dilations,
                                                                   ov::op::PadType::EXPLICIT);

        model = std::make_shared<ov::Model>(ov::NodeVector{ deconv }, ov::ParameterVector{ input });
        manager.register_pass<ConvertDeconvolutionToInternal>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ 2, 8, 11, 12 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 8, 1, 4, 3, 3 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto output_shape = ov::op::v0::Constant::create(ov::element::i16, ov::Shape{ 2 }, { 1 });
        auto deconv = std::make_shared<ov::intel_gpu::op::Deconvolution>(input,
                                                                     weights_const,
                                                                     no_bias,
                                                                     output_shape,
                                                                     strides,
                                                                     pads_begin,
                                                                     pads_end,
                                                                     dilations,
                                                                     8,
                                                                     ov::op::PadType::EXPLICIT,
                                                                     ov::element::f16,
                                                                     output_padding);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ deconv }, ov::ParameterVector{ input });
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
