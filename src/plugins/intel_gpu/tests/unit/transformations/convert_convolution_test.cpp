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
#include "intel_gpu/op/convolution.hpp"
#include "intel_gpu/op/placeholder.hpp"

#include "plugin/transformations/convert_convolution.hpp"
#include "plugin/transformations/utils.hpp"

#include <memory>

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

TEST_F(TransformationTestsF, ConvertConvolutionToInternal_1) {
    ov::Strides strides{1, 1};
    ov::Strides dilations{1, 1};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ 2, 3, 11, 12 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 4, 3, 3, 3 }, { 1 });
        auto conv = std::make_shared<ov::op::v1::Convolution>(input,
                                                              weights_const,
                                                              strides,
                                                              pads_begin,
                                                              pads_end,
                                                              dilations,
                                                              ov::op::PadType::EXPLICIT);

        model = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{input});
        manager.register_pass<ConvertConvolutionToInternal>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{  2, 3, 11, 12 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 4, 3, 3, 3 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto conv = std::make_shared<ov::intel_gpu::op::Convolution>(input,
                                                                     weights_const,
                                                                     no_bias,
                                                                     strides,
                                                                     pads_begin,
                                                                     pads_end,
                                                                     dilations,
                                                                     -1,
                                                                     ov::op::PadType::EXPLICIT,
                                                                     ov::element::f32);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ConvertConvolutionToInternal_2) {
    ov::Strides strides{1, 1};
    ov::Strides dilations{1, 1};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ 2, 3, 11, 12 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 4, 3, 3, 3 }, { 1 });
        auto conv = std::make_shared<ov::op::v1::Convolution>(input,
                                                              weights_const,
                                                              strides,
                                                              pads_begin,
                                                              pads_end,
                                                              dilations,
                                                              ov::op::PadType::EXPLICIT);

        model = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{input});
        manager.register_pass<ConvertConvolutionToInternal>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{  2, 3, 11, 12 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 4, 3, 3, 3 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto conv = std::make_shared<ov::intel_gpu::op::Convolution>(input,
                                                                     weights_const,
                                                                     no_bias,
                                                                     strides,
                                                                     pads_begin,
                                                                     pads_end,
                                                                     dilations,
                                                                     -1,
                                                                     ov::op::PadType::EXPLICIT,
                                                                     ov::element::f16);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ConvertConvolutionToInternal_3) {
    ov::Strides strides{1, 1};
    ov::Strides dilations{1, 1};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{ 2, 3, 11, 12 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{ 4, 3, 3, 3 }, { 1 });
        auto wzp_const = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{ 1 }, { 1 });
        auto w_sub = make_type_relaxed<ov::op::v1::Subtract>(element::TypeVector{ov::element::u8, ov::element::u8},
                                                             element::TypeVector{ov::element::f32},
                                                             ov::op::TemporaryReplaceOutputType(weights_const, ov::element::f32).get(),
                                                             ov::op::TemporaryReplaceOutputType(wzp_const, ov::element::f32).get());
        auto conv = make_type_relaxed<ov::op::v1::Convolution>(element::TypeVector{ov::element::u8, ov::element::u8},
                                                               element::TypeVector{ov::element::f32},
                                                               ov::op::TemporaryReplaceOutputType(input, ov::element::f32).get(),
                                                               ov::op::TemporaryReplaceOutputType(w_sub, ov::element::f32).get(),
                                                               strides,
                                                               pads_begin,
                                                               pads_end,
                                                               dilations,
                                                               ov::op::PadType::EXPLICIT);

        model = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{input});
        manager.register_pass<ConvertConvolutionToInternal>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{  2, 3, 11, 12 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{ 4, 3, 3, 3 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto wzp_const = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{ 1 }, { 1 });
        auto no_azp = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto no_compensation = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto conv = std::make_shared<ov::intel_gpu::op::Convolution>(input,
                                                                     weights_const,
                                                                     no_bias,
                                                                     no_azp,
                                                                     wzp_const,
                                                                     no_compensation,
                                                                     strides,
                                                                     pads_begin,
                                                                     pads_end,
                                                                     dilations,
                                                                     -1,
                                                                     ov::op::PadType::EXPLICIT,
                                                                     ov::element::f32);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ConvertConvolutionToInternal_4) {
    ov::Strides strides{1, 1};
    ov::Strides dilations{1, 1};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{ 2, 3, 11, 12 });
        auto azp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1 }, { 1 });
        auto a_sub = make_type_relaxed<ov::op::v1::Subtract>(element::TypeVector{ov::element::u8, ov::element::u8},
                                                             element::TypeVector{ov::element::f32},
                                                             ov::op::TemporaryReplaceOutputType(input, ov::element::f32).get(),
                                                             ov::op::TemporaryReplaceOutputType(azp_const, ov::element::f32).get());

        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 4, 3, 3, 3 }, { 1 });
        auto wzp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1 }, { 1 });
        auto w_sub = make_type_relaxed<ov::op::v1::Subtract>(element::TypeVector{ov::element::u8, ov::element::u8},
                                                             element::TypeVector{ov::element::f32},
                                                             ov::op::TemporaryReplaceOutputType(weights_const, ov::element::f32).get(),
                                                             ov::op::TemporaryReplaceOutputType(wzp_const, ov::element::f32).get());

        auto conv = make_type_relaxed<ov::op::v1::Convolution>(element::TypeVector{ov::element::u8, ov::element::u8},
                                                               element::TypeVector{ov::element::f32},
                                                               ov::op::TemporaryReplaceOutputType(a_sub, ov::element::f32).get(),
                                                               ov::op::TemporaryReplaceOutputType(w_sub, ov::element::f32).get(),
                                                               strides,
                                                               pads_begin,
                                                               pads_end,
                                                               dilations,
                                                               ov::op::PadType::EXPLICIT);

        model = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{input});
        manager.register_pass<ConvertConvolutionToInternal>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{  2, 3, 11, 12 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 4, 3, 3, 3 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto azp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1 }, { 1 });
        auto wzp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1 }, { 1 });
        auto compensation = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 1, 4, 1, 1 }, { 1 });
        auto conv = std::make_shared<ov::intel_gpu::op::Convolution>(input,
                                                                     weights_const,
                                                                     no_bias,
                                                                     azp_const,
                                                                     wzp_const,
                                                                     compensation,
                                                                     strides,
                                                                     pads_begin,
                                                                     pads_end,
                                                                     dilations,
                                                                     -1,
                                                                     ov::op::PadType::EXPLICIT,
                                                                     ov::element::f32);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ConvertConvolutionToInternal_5) {
    ov::Strides strides{1};
    ov::Strides dilations{1};
    ov::CoordinateDiff pads_begin{0};
    ov::CoordinateDiff pads_end{2};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{ 2, 3, 11 });
        auto azp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1 }, { 1 });
        auto a_sub = make_type_relaxed<ov::op::v1::Subtract>(element::TypeVector{ov::element::u8, ov::element::u8},
                                                             element::TypeVector{ov::element::f32},
                                                             ov::op::TemporaryReplaceOutputType(input, ov::element::f32).get(),
                                                             ov::op::TemporaryReplaceOutputType(azp_const, ov::element::f32).get());

        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 4, 3, 3 }, { 1 });
        auto wzp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1 }, { 1 });
        auto w_sub = make_type_relaxed<ov::op::v1::Subtract>(element::TypeVector{ov::element::u8, ov::element::u8},
                                                             element::TypeVector{ov::element::f32},
                                                             ov::op::TemporaryReplaceOutputType(weights_const, ov::element::f32).get(),
                                                             ov::op::TemporaryReplaceOutputType(wzp_const, ov::element::f32).get());

        auto conv = make_type_relaxed<ov::op::v1::Convolution>(element::TypeVector{ov::element::u8, ov::element::u8},
                                                               element::TypeVector{ov::element::f32},
                                                               ov::op::TemporaryReplaceOutputType(a_sub, ov::element::f32).get(),
                                                               ov::op::TemporaryReplaceOutputType(w_sub, ov::element::f32).get(),
                                                               strides,
                                                               pads_begin,
                                                               pads_end,
                                                               dilations,
                                                               ov::op::PadType::EXPLICIT);

        model = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{input});
        manager.register_pass<ConvertConvolutionToInternal>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{  2, 3, 11 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 4, 3, 3 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto azp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1 }, { 1 });
        auto wzp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1 }, { 1 });
        auto compensation = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 1, 4, 1 }, { 1 });
        auto conv = std::make_shared<ov::intel_gpu::op::Convolution>(input,
                                                                     weights_const,
                                                                     no_bias,
                                                                     azp_const,
                                                                     wzp_const,
                                                                     compensation,
                                                                     strides,
                                                                     pads_begin,
                                                                     pads_end,
                                                                     dilations,
                                                                     -1,
                                                                     ov::op::PadType::EXPLICIT,
                                                                     ov::element::f32);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ConvertGroupConvolutionToInternal_1) {
    ov::Strides strides{1, 1};
    ov::Strides dilations{1, 1};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ 2, 8, 11, 12 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 2, 8, 4, 3, 3 }, { 1 });
        auto conv = std::make_shared<ov::op::v1::GroupConvolution>(input,
                                                                   weights_const,
                                                                   strides,
                                                                   pads_begin,
                                                                   pads_end,
                                                                   dilations,
                                                                   ov::op::PadType::EXPLICIT);

        model = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{input});
        manager.register_pass<ConvertConvolutionToInternal>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{ 2, 8, 11, 12 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 2, 8, 4, 3, 3 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto conv = std::make_shared<ov::intel_gpu::op::Convolution>(input,
                                                                     weights_const,
                                                                     no_bias,
                                                                     strides,
                                                                     pads_begin,
                                                                     pads_end,
                                                                     dilations,
                                                                     2,
                                                                     ov::op::PadType::EXPLICIT,
                                                                     ov::element::f32);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ConvertGroupConvolutionToInternal_2) {
    ov::Strides strides{1, 1};
    ov::Strides dilations{1, 1};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ 2, 8, 11, 12 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 2, 8, 4, 3, 3 }, { 1 });
        auto conv = std::make_shared<ov::op::v1::GroupConvolution>(input,
                                                                   weights_const,
                                                                   strides,
                                                                   pads_begin,
                                                                   pads_end,
                                                                   dilations,
                                                                   ov::op::PadType::EXPLICIT);

        model = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{input});
        manager.register_pass<ConvertConvolutionToInternal>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ 2, 8, 11, 12 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 2, 8, 4, 3, 3 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto conv = std::make_shared<ov::intel_gpu::op::Convolution>(input,
                                                                     weights_const,
                                                                     no_bias,
                                                                     strides,
                                                                     pads_begin,
                                                                     pads_end,
                                                                     dilations,
                                                                     2,
                                                                     ov::op::PadType::EXPLICIT,
                                                                     ov::element::f16);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ConvertGroupConvolutionToInternal_3) {
    ov::Strides strides{1, 1};
    ov::Strides dilations{1, 1};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{ 2, 8, 11, 12 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{ 2, 8, 4, 3, 3 }, { 1 });
        auto wzp_const = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{ 1 }, { 1 });
        auto w_sub = make_type_relaxed<ov::op::v1::Subtract>(element::TypeVector{ov::element::u8, ov::element::u8},
                                                             element::TypeVector{ov::element::f32},
                                                             ov::op::TemporaryReplaceOutputType(weights_const, ov::element::f32).get(),
                                                             ov::op::TemporaryReplaceOutputType(wzp_const, ov::element::f32).get());
        auto conv = make_type_relaxed<ov::op::v1::GroupConvolution>(element::TypeVector{ov::element::u8, ov::element::u8},
                                                                    element::TypeVector{ov::element::f32},
                                                                    ov::op::TemporaryReplaceOutputType(input, ov::element::f32).get(),
                                                                    ov::op::TemporaryReplaceOutputType(w_sub, ov::element::f32).get(),
                                                                    strides,
                                                                    pads_begin,
                                                                    pads_end,
                                                                    dilations,
                                                                    ov::op::PadType::EXPLICIT);

        model = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{input});
        manager.register_pass<ConvertConvolutionToInternal>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{  2, 8, 11, 12 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{ 2, 8, 4, 3, 3 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto wzp_const = ov::op::v0::Constant::create(ov::element::i8, ov::Shape{ 1 }, { 1 });
        auto no_azp = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto no_compensation = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto conv = std::make_shared<ov::intel_gpu::op::Convolution>(input,
                                                                     weights_const,
                                                                     no_bias,
                                                                     no_azp,
                                                                     wzp_const,
                                                                     no_compensation,
                                                                     strides,
                                                                     pads_begin,
                                                                     pads_end,
                                                                     dilations,
                                                                     2,
                                                                     ov::op::PadType::EXPLICIT,
                                                                     ov::element::f32);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ConvertGroupConvolutionToInternal_4) {
    ov::Strides strides{1, 1};
    ov::Strides dilations{1, 1};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{ 2, 8, 11, 12 });
        auto azp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1 }, { 1 });
        auto a_sub = make_type_relaxed<ov::op::v1::Subtract>(element::TypeVector{ov::element::u8, ov::element::u8},
                                                             element::TypeVector{ov::element::f32},
                                                             ov::op::TemporaryReplaceOutputType(input, ov::element::f32).get(),
                                                             ov::op::TemporaryReplaceOutputType(azp_const, ov::element::f32).get());

        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 2, 8, 4, 3, 3 }, { 1 });
        auto wzp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1 }, { 1 });
        auto w_sub = make_type_relaxed<ov::op::v1::Subtract>(element::TypeVector{ov::element::u8, ov::element::u8},
                                                             element::TypeVector{ov::element::f32},
                                                             ov::op::TemporaryReplaceOutputType(weights_const, ov::element::f32).get(),
                                                             ov::op::TemporaryReplaceOutputType(wzp_const, ov::element::f32).get());

        auto conv = make_type_relaxed<ov::op::v1::GroupConvolution>(element::TypeVector{ov::element::u8, ov::element::u8},
                                                                    element::TypeVector{ov::element::f32},
                                                                    ov::op::TemporaryReplaceOutputType(a_sub, ov::element::f32).get(),
                                                                    ov::op::TemporaryReplaceOutputType(w_sub, ov::element::f32).get(),
                                                                    strides,
                                                                    pads_begin,
                                                                    pads_end,
                                                                    dilations,
                                                                    ov::op::PadType::EXPLICIT);

        model = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{input});
        manager.register_pass<ConvertConvolutionToInternal>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{ 2, 8, 11, 12 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 2, 8, 4, 3, 3 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto azp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1 }, { 1 });
        auto wzp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1 }, { 1 });
        auto compensation = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 1, 16, 1, 1 }, { 1 });
        auto conv = std::make_shared<ov::intel_gpu::op::Convolution>(input,
                                                                     weights_const,
                                                                     no_bias,
                                                                     azp_const,
                                                                     wzp_const,
                                                                     compensation,
                                                                     strides,
                                                                     pads_begin,
                                                                     pads_end,
                                                                     dilations,
                                                                     2,
                                                                     ov::op::PadType::EXPLICIT,
                                                                     ov::element::f32);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ConvertGroupConvolutionToInternal_5) {
    ov::Strides strides{2, 2, 2};
    ov::Strides dilations{1, 1, 1};
    ov::CoordinateDiff pads_begin{2, 2, 2};
    ov::CoordinateDiff pads_end{2, 2, 2};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{ 2, 8, 11, 12, 3 });
        auto azp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1 }, { 1 });
        auto a_sub = make_type_relaxed<ov::op::v1::Subtract>(element::TypeVector{ov::element::u8, ov::element::u8},
                                                             element::TypeVector{ov::element::f32},
                                                             ov::op::TemporaryReplaceOutputType(input, ov::element::f32).get(),
                                                             ov::op::TemporaryReplaceOutputType(azp_const, ov::element::f32).get());

        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 8, 1, 1, 3, 3, 5 }, { 1 });
        auto wzp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1 }, { 1 });
        auto w_sub = make_type_relaxed<ov::op::v1::Subtract>(element::TypeVector{ov::element::u8, ov::element::u8},
                                                             element::TypeVector{ov::element::f32},
                                                             ov::op::TemporaryReplaceOutputType(weights_const, ov::element::f32).get(),
                                                             ov::op::TemporaryReplaceOutputType(wzp_const, ov::element::f32).get());

        auto conv = make_type_relaxed<ov::op::v1::GroupConvolution>(element::TypeVector{ov::element::u8, ov::element::u8},
                                                                    element::TypeVector{ov::element::f32},
                                                                    ov::op::TemporaryReplaceOutputType(a_sub, ov::element::f32).get(),
                                                                    ov::op::TemporaryReplaceOutputType(w_sub, ov::element::f32).get(),
                                                                    strides,
                                                                    pads_begin,
                                                                    pads_end,
                                                                    dilations,
                                                                    ov::op::PadType::EXPLICIT);

        model = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{input});
        manager.register_pass<ConvertConvolutionToInternal>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{ 2, 8, 11, 12, 3 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 8, 1, 1, 3, 3, 5 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto azp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1 }, { 1 });
        auto wzp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1 }, { 1 });
        auto compensation = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 1, 8, 1, 1, 1 }, { 1 });
        auto conv = std::make_shared<ov::intel_gpu::op::Convolution>(input,
                                                                     weights_const,
                                                                     no_bias,
                                                                     azp_const,
                                                                     wzp_const,
                                                                     compensation,
                                                                     strides,
                                                                     pads_begin,
                                                                     pads_end,
                                                                     dilations,
                                                                     8,
                                                                     ov::op::PadType::EXPLICIT,
                                                                     ov::element::f32);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{conv}, ov::ParameterVector{input});
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
