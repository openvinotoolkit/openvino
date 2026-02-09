// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/common/pass/convert_batch_gather_matmul_to_compressed.hpp"

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
#include "transformations/cpu_opset/common/op/batch_gather_matmul.hpp"
#include "transformations/cpu_opset/common/op/batch_gather_matmul_compressed.hpp"

using namespace testing;
using namespace ov::pass;
using namespace ov::intel_cpu;

TEST_F(TransformationTestsF, ConvertBGMToCompressed1) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{8, 10, 2048});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{128, 5, 2048}, {1});
        auto wei_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{128, 5, 1}, {1});
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f32);
        auto wei_zp = std::make_shared<ov::op::v1::Subtract>(wei_convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{128, 5, 1}, {1});
        auto wei_scale = std::make_shared<ov::op::v1::Multiply>(wei_zp, scale_const);
        auto index = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{10, 8}, {1});
        auto bgm = std::make_shared<BatchGatherMatmul>(input, wei_scale, index);

        model = std::make_shared<ov::Model>(ov::OutputVector{bgm}, ov::ParameterVector{input});

        const std::vector<ov::element::Type> supported_activation_types{ov::element::f32};
        const std::vector<ov::element::Type> supported_weights_types{ov::element::u8};
        manager.register_pass<ConvertBatchGatherMatmulToBatchGatherMatmulCompressed>(supported_activation_types,
                                                                                     supported_weights_types);
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{8, 10, 2048});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{128, 5, 2048}, {1});
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{128, 5, 1}, {1});
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{128, 5, 1}, {1});
        auto index = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{10, 8}, {1});
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0});
        auto bgm_compressed =
            std::make_shared<BatchGatherMatmulCompressed>(input, weights_const, index, bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{bgm_compressed}, ov::ParameterVector{input});
    }
}

// group compressed
TEST_F(TransformationTestsF, ConvertBGMToCompressed2) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{8, 10, 2048});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{128, 5, 16, 128}, {1});
        auto wei_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{128, 5, 16, 1}, {1});
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f32);
        auto wei_zp = std::make_shared<ov::op::v1::Subtract>(wei_convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{128, 5, 16, 1}, {1});
        auto wei_scale = std::make_shared<ov::op::v1::Multiply>(wei_zp, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {128, 5, 2048});
        auto wei_reshape = std::make_shared<ov::op::v1::Reshape>(wei_scale, reshape_const, false);
        auto index = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{10, 8}, {1});

        auto bgm = std::make_shared<BatchGatherMatmul>(input, wei_reshape, index);

        model = std::make_shared<ov::Model>(ov::OutputVector{bgm}, ov::ParameterVector{input});

        const std::vector<ov::element::Type> supported_activation_types{ov::element::f32};
        const std::vector<ov::element::Type> supported_weights_types{ov::element::u8};
        manager.register_pass<ConvertBatchGatherMatmulToBatchGatherMatmulCompressed>(supported_activation_types,
                                                                                     supported_weights_types);
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{8, 10, 2048});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{128, 5, 2048}, {1});
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{128, 5, 16}, {1});
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{128, 5, 16}, {1});
        auto index = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{10, 8}, {1});
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0});
        auto bgm_compressed =
            std::make_shared<BatchGatherMatmulCompressed>(input, weights_const, index, bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{bgm_compressed}, ov::ParameterVector{input});
    }
}

// group compressed with single out channel
TEST_F(TransformationTestsF, ConvertBGMToCompressed3) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{8, 10, 2048});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{128, 1, 16, 128}, {1});
        auto wei_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{128, 1, 16, 1}, {1});
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f32);
        auto wei_zp = std::make_shared<ov::op::v1::Subtract>(wei_convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{128, 1, 16, 1}, {1});
        auto wei_scale = std::make_shared<ov::op::v1::Multiply>(wei_zp, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {128, 1, 2048});
        auto wei_reshape = std::make_shared<ov::op::v1::Reshape>(wei_scale, reshape_const, false);
        auto index = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{10, 8}, {0});

        auto bgm = std::make_shared<BatchGatherMatmul>(input, wei_reshape, index);

        model = std::make_shared<ov::Model>(ov::OutputVector{bgm}, ov::ParameterVector{input});

        const std::vector<ov::element::Type> supported_activation_types{ov::element::f32};
        const std::vector<ov::element::Type> supported_weights_types{ov::element::u8};
        manager.register_pass<ConvertBatchGatherMatmulToBatchGatherMatmulCompressed>(supported_activation_types,
                                                                                     supported_weights_types);
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{8, 10, 2048});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{128, 1, 2048}, {1});
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{128, 1, 16}, {1});
        auto zp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{128, 1, 16}, {1});
        auto index = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{10, 8}, {0});
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0});
        auto bgm_compressed =
            std::make_shared<BatchGatherMatmulCompressed>(input, weights_const, index, bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{bgm_compressed}, ov::ParameterVector{input});
    }
}

// u4
TEST_F(TransformationTestsF, ConvertBGMToCompressed4) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{8, 10, 2048});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{128, 5, 2048}, {1});
        auto wei_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{128, 5, 1}, {1});
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f32);
        auto wei_zp = std::make_shared<ov::op::v1::Subtract>(wei_convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{128, 5, 1}, {1});
        auto wei_scale = std::make_shared<ov::op::v1::Multiply>(wei_zp, scale_const);
        auto index = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{10, 8}, {1});
        auto bgm = std::make_shared<BatchGatherMatmul>(input, wei_scale, index);

        model = std::make_shared<ov::Model>(ov::OutputVector{bgm}, ov::ParameterVector{input});

        const std::vector<ov::element::Type> supported_activation_types{ov::element::f32};
        const std::vector<ov::element::Type> supported_weights_types{ov::element::u4};
        manager.register_pass<ConvertBatchGatherMatmulToBatchGatherMatmulCompressed>(supported_activation_types,
                                                                                     supported_weights_types);
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{8, 10, 2048});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{128, 5, 2048}, {1});
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{128, 5, 1}, {1});
        auto zp_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{128, 5, 1}, {1});
        auto index = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{10, 8}, {1});
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0});
        auto bgm_compressed =
            std::make_shared<BatchGatherMatmulCompressed>(input, weights_const, index, bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{bgm_compressed}, ov::ParameterVector{input});
    }
}

// group compressed
TEST_F(TransformationTestsF, ConvertBGMToCompressed5) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{8, 10, 2048});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{128, 5, 16, 128}, {1});
        auto wei_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{128, 5, 16, 1}, {1});
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f32);
        auto wei_zp = std::make_shared<ov::op::v1::Subtract>(wei_convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{128, 5, 16, 1}, {1});
        auto wei_scale = std::make_shared<ov::op::v1::Multiply>(wei_zp, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {128, 5, 2048});
        auto wei_reshape = std::make_shared<ov::op::v1::Reshape>(wei_scale, reshape_const, false);
        auto index = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{10, 8}, {1});

        auto bgm = std::make_shared<BatchGatherMatmul>(input, wei_reshape, index);

        model = std::make_shared<ov::Model>(ov::OutputVector{bgm}, ov::ParameterVector{input});

        const std::vector<ov::element::Type> supported_activation_types{ov::element::f32};
        const std::vector<ov::element::Type> supported_weights_types{ov::element::u4};
        manager.register_pass<ConvertBatchGatherMatmulToBatchGatherMatmulCompressed>(supported_activation_types,
                                                                                     supported_weights_types);
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{8, 10, 2048});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{128, 5, 2048}, {1});
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{128, 5, 16}, {1});
        auto zp_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{128, 5, 16}, {1});
        auto index = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{10, 8}, {1});
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0});
        auto bgm_compressed =
            std::make_shared<BatchGatherMatmulCompressed>(input, weights_const, index, bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{bgm_compressed}, ov::ParameterVector{input});
    }
}

// group compressed with single out channel
TEST_F(TransformationTestsF, ConvertBGMToCompressed6) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{8, 10, 2048});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{128, 1, 16, 128}, {1});
        auto wei_convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{128, 1, 16, 1}, {1});
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_const, ov::element::f32);
        auto wei_zp = std::make_shared<ov::op::v1::Subtract>(wei_convert, zp_convert);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{128, 1, 16, 1}, {1});
        auto wei_scale = std::make_shared<ov::op::v1::Multiply>(wei_zp, scale_const);
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {128, 1, 2048});
        auto wei_reshape = std::make_shared<ov::op::v1::Reshape>(wei_scale, reshape_const, false);
        auto index = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{10, 8}, {0});

        auto bgm = std::make_shared<BatchGatherMatmul>(input, wei_reshape, index);

        model = std::make_shared<ov::Model>(ov::OutputVector{bgm}, ov::ParameterVector{input});

        const std::vector<ov::element::Type> supported_activation_types{ov::element::f32};
        const std::vector<ov::element::Type> supported_weights_types{ov::element::u4};
        manager.register_pass<ConvertBatchGatherMatmulToBatchGatherMatmulCompressed>(supported_activation_types,
                                                                                     supported_weights_types);
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{8, 10, 2048});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{128, 1, 2048}, {1});
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{128, 1, 16}, {1});
        auto zp_const = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{128, 1, 16}, {1});
        auto index = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{10, 8}, {0});
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0});
        auto bgm_compressed =
            std::make_shared<BatchGatherMatmulCompressed>(input, weights_const, index, bias, scale_const, zp_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{bgm_compressed}, ov::ParameterVector{input});
    }
}