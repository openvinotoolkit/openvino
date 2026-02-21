// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_fc_to_compressed.hpp"

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/fully_connected.hpp"
#include "ov_ops/fully_connected_compressed.hpp"

using namespace testing;
using namespace ov::pass;

TEST_F(TransformationTestsF, ConvertFCToCompressed1) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{10, 2048});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{5, 2048}, {1});
        auto wei_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{5, 1}, {1});
        auto wei_scale = std::make_shared<ov::op::v1::Multiply>(wei_convert, scale_const);
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});
        auto fc = std::make_shared<ov::op::internal::FullyConnected>(input, wei_scale, bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});

        const std::vector<ov::element::Type> supported_activation_types{ov::element::f32};
        const std::vector<ov::element::Type> supported_weights_types{ov::element::u8};
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>(supported_activation_types,
                                                                               supported_weights_types);
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{10, 2048});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{5, 2048}, {1});
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{5, 1}, {1});
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});
        auto fc_compressed =
            std::make_shared<ov::op::internal::FullyConnectedCompressed>(input, weights_const, bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input});
    }
}

// with zp
TEST_F(TransformationTestsF, ConvertFCToCompressed2) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{10, 2048});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{5, 2048}, {1});
        auto wei_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{5, 1}, {1});
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f32);
        auto wei_zp = std::make_shared<ov::op::v1::Subtract>(wei_convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{5, 1}, {1});
        auto wei_scale = std::make_shared<ov::op::v1::Multiply>(wei_zp, scale_const);
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});
        auto fc = std::make_shared<ov::op::internal::FullyConnected>(input, wei_scale, bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});

        const std::vector<ov::element::Type> supported_activation_types{ov::element::f32};
        const std::vector<ov::element::Type> supported_weights_types{ov::element::u8};
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>(supported_activation_types,
                                                                               supported_weights_types);
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{10, 2048});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{5, 2048}, {1});
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{5, 1}, {1});
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{5, 1}, {1});
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});
        auto fc_compressed = std::make_shared<ov::op::internal::FullyConnectedCompressed>(input,
                                                                                          weights_const,
                                                                                          bias,
                                                                                          scale_const,
                                                                                          zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input});
    }
}

// group compressed
TEST_F(TransformationTestsF, ConvertFCToCompressed3) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{10, 2048});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{5, 16, 128}, {1});
        auto wei_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{5, 16, 1}, {1});
        auto wei_scale = std::make_shared<ov::op::v1::Multiply>(wei_convert, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {-1, 2048});
        auto wei_reshape = std::make_shared<ov::op::v1::Reshape>(wei_scale, reshape_const, false);
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});

        auto fc = std::make_shared<ov::op::internal::FullyConnected>(input, wei_reshape, bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});

        const std::vector<ov::element::Type> supported_activation_types{ov::element::f32};
        const std::vector<ov::element::Type> supported_weights_types{ov::element::u8};
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>(supported_activation_types,
                                                                               supported_weights_types);
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{10, 2048});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{5, 2048}, {1});
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{5, 16}, {1});
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});
        auto fc_compressed =
            std::make_shared<ov::op::internal::FullyConnectedCompressed>(input, weights_const, bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input});
    }
}

// group compressed with single out channel
TEST_F(TransformationTestsF, ConvertFCToCompressed4) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{10, 2048});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{1, 16, 128}, {1});
        auto wei_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 16, 1}, {1});
        auto wei_scale = std::make_shared<ov::op::v1::Multiply>(wei_convert, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {-1, 2048});
        auto wei_reshape = std::make_shared<ov::op::v1::Reshape>(wei_scale, reshape_const, false);
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});

        auto fc = std::make_shared<ov::op::internal::FullyConnected>(input, wei_reshape, bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});

        const std::vector<ov::element::Type> supported_activation_types{ov::element::f32};
        const std::vector<ov::element::Type> supported_weights_types{ov::element::u8};
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>(supported_activation_types,
                                                                               supported_weights_types);
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{10, 2048});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{1, 2048}, {1});
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 16}, {1});
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});
        auto fc_compressed =
            std::make_shared<ov::op::internal::FullyConnectedCompressed>(input, weights_const, bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input});
    }
}

// u4
TEST_F(TransformationTestsF, ConvertFCToCompressed5) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{10, 2048});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{5, 2048}, {1});
        auto wei_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{5, 1}, {1});
        auto wei_scale = std::make_shared<ov::op::v1::Multiply>(wei_convert, scale_const);
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});
        auto fc = std::make_shared<ov::op::internal::FullyConnected>(input, wei_scale, bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});

        const std::vector<ov::element::Type> supported_activation_types{ov::element::f32};
        const std::vector<ov::element::Type> supported_weights_types{ov::element::u4};
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>(supported_activation_types,
                                                                               supported_weights_types);
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{10, 2048});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{5, 2048}, {1});
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{5, 1}, {1});
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});
        auto fc_compressed =
            std::make_shared<ov::op::internal::FullyConnectedCompressed>(input, weights_const, bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input});
    }
}

// with zp
TEST_F(TransformationTestsF, ConvertFCToCompressed6) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{10, 2048});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{5, 2048}, {1});
        auto wei_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{5, 1}, {1});
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f32);
        auto wei_zp = std::make_shared<ov::op::v1::Subtract>(wei_convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{5, 1}, {1});
        auto wei_scale = std::make_shared<ov::op::v1::Multiply>(wei_zp, scale_const);
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});
        auto fc = std::make_shared<ov::op::internal::FullyConnected>(input, wei_scale, bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});

        const std::vector<ov::element::Type> supported_activation_types{ov::element::f32};
        const std::vector<ov::element::Type> supported_weights_types{ov::element::u4};
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>(supported_activation_types,
                                                                               supported_weights_types);
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{10, 2048});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{5, 2048}, {1});
        auto zp_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{5, 1}, {1});
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{5, 1}, {1});
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});
        auto fc_compressed = std::make_shared<ov::op::internal::FullyConnectedCompressed>(input,
                                                                                          weights_const,
                                                                                          bias,
                                                                                          scale_const,
                                                                                          zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input});
    }
}

// group compressed
TEST_F(TransformationTestsF, ConvertFCToCompressed7) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{10, 2048});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{5, 16, 128}, {1});
        auto wei_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{5, 16, 1}, {1});
        auto wei_scale = std::make_shared<ov::op::v1::Multiply>(wei_convert, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {-1, 2048});
        auto wei_reshape = std::make_shared<ov::op::v1::Reshape>(wei_scale, reshape_const, false);
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});

        auto fc = std::make_shared<ov::op::internal::FullyConnected>(input, wei_reshape, bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});

        const std::vector<ov::element::Type> supported_activation_types{ov::element::f32};
        const std::vector<ov::element::Type> supported_weights_types{ov::element::u4};
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>(supported_activation_types,
                                                                               supported_weights_types);
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{10, 2048});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{5, 2048}, {1});
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{5, 16}, {1});
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});
        auto fc_compressed =
            std::make_shared<ov::op::internal::FullyConnectedCompressed>(input, weights_const, bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input});
    }
}

// group compressed with single out channel
TEST_F(TransformationTestsF, ConvertFCToCompressed8) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{10, 2048});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{1, 16, 128}, {1});
        auto wei_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 16, 1}, {1});
        auto wei_scale = std::make_shared<ov::op::v1::Multiply>(wei_convert, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {-1, 2048});
        auto wei_reshape = std::make_shared<ov::op::v1::Reshape>(wei_scale, reshape_const, false);
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});

        auto fc = std::make_shared<ov::op::internal::FullyConnected>(input, wei_reshape, bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});

        const std::vector<ov::element::Type> supported_activation_types{ov::element::f32};
        const std::vector<ov::element::Type> supported_weights_types{ov::element::u4};
        manager.register_pass<ConvertFullyConnectedToFullyConnectedCompressed>(supported_activation_types,
                                                                               supported_weights_types);
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{10, 2048});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{1, 2048}, {1});
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 16}, {1});
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{0});
        auto fc_compressed =
            std::make_shared<ov::op::internal::FullyConnectedCompressed>(input, weights_const, bias, scale_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{fc_compressed}, ov::ParameterVector{input});
    }
}
