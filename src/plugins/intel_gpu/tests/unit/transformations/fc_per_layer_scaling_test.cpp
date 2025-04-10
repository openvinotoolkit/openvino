// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/ov_test_utils.hpp"

#include <string>
#include <memory>

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/pass/manager.hpp"

#include <transformations/utils/utils.hpp>
#include "plugin/transformations/fc_per_layer_scaling.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

TEST_F(TransformationTestsF, FullyConnectedPerLayerScalingTest1) {
    float scale_factor = 2.f;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weights_const, no_bias, scale_const, zp_const);
        auto convert = std::make_shared<ov::op::v0::Convert>(fc_compressed, ov::element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);

        model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
        manager.register_pass<FullyConnectedPerLayerScaling>(scale_factor);
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto scale_down_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 1 }, { 1.f / scale_factor });
        auto scale_down = std::make_shared<ov::op::v1::Multiply>(input, scale_down_const);
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(scale_down, weights_const, no_bias, scale_const, zp_const);
        auto scale_up_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 1 }, { scale_factor });
        auto scale_up = std::make_shared<ov::op::v1::Multiply>(fc_compressed, scale_up_const);
        auto convert = std::make_shared<ov::op::v0::Convert>(scale_up, ov::element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, FullyConnectedPerLayerScalingTest2) {
    float scale_factor = 2.f;
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{ 1, 32 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weights_const, bias, scale_const, zp_const);
        auto convert = std::make_shared<ov::op::v0::Convert>(fc_compressed, ov::element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);

        model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
        manager.register_pass<FullyConnectedPerLayerScaling>(scale_factor);
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto bias = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{ 1, 32 });
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto scale_down_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 1 }, { 1.f / scale_factor });
        auto scale_down = std::make_shared<ov::op::v1::Multiply>(input, scale_down_const);
        auto bias_scale_down_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 1 }, { 1.f / scale_factor });
        auto bias_scale_down = std::make_shared<ov::op::v1::Multiply>(bias, scale_down_const);
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(scale_down, weights_const, bias_scale_down, scale_const, zp_const);
        auto scale_up_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 1 }, { scale_factor });
        auto scale_up = std::make_shared<ov::op::v1::Multiply>(fc_compressed, scale_up_const);
        auto convert = std::make_shared<ov::op::v0::Convert>(scale_up, ov::element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);

        model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

TEST_F(TransformationTestsF, FullyConnectedPerLayerScalingTest3) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, 16 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 16 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto zp_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{ 32, 1 }, { 1 });
        auto fc_compressed = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(input, weights_const, no_bias, scale_const, zp_const);
        auto convert = std::make_shared<ov::op::v0::Convert>(fc_compressed, ov::element::f32);
        auto result = std::make_shared<ov::op::v0::Result>(convert);

        model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input});
        manager.register_pass<FullyConnectedPerLayerScaling>(1.f);
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov