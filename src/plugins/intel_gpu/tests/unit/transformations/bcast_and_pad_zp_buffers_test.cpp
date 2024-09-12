// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"

#include "openvino/core/model.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "intel_gpu/op/convolution.hpp"
#include "intel_gpu/op/placeholder.hpp"

#include "plugin/transformations/bcast_and_pad_zp_buffers.hpp"

#include <memory>

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

TEST_F(TransformationTestsF, BroadcastAndPadZeroPointBuffers_1) {
    ov::Strides strides{1, 1};
    ov::Strides dilations{1, 1};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{ 2, 8, 11, 12 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 2, 8, 4, 3, 3 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto azp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 1, 1, 1 }, { 1 });
        auto wzp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 1, 1, 1 }, { 1 });
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

        model = std::make_shared<ov::Model>(ov::NodeVector{ conv }, ov::ParameterVector{ input });
        manager.register_pass<BroadcastAndPadZeroPointBuffers>(32);
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{ 2, 8, 11, 12 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 2, 8, 4, 3, 3 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto azp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 32, 1, 1 }, { 1 });
        auto wzp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 1, 1, 1 }, { 1 });
        auto compensation = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 1, 32, 1, 1 }, { 1 });
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

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ conv }, ov::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, BroadcastAndPadZeroPointBuffers_2) {
    ov::Strides strides{1};
    ov::Strides dilations{1};
    ov::CoordinateDiff pads_begin{2};
    ov::CoordinateDiff pads_end{0};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{ 2, 3, 11 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 4, 3, 3 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto azp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 1, 1 }, { 1 });
        auto wzp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 1, 1 }, { 1 });
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

        model = std::make_shared<ov::Model>(ov::NodeVector{ conv }, ov::ParameterVector{ input });
        manager.register_pass<BroadcastAndPadZeroPointBuffers>(32);
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{ 2, 3, 11 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 4, 3, 3 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto azp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 32, 1 }, { 1 });
        auto wzp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32, 1, 1 }, { 1 });
        auto compensation = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 1, 32, 1 }, { 1 });
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

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ conv }, ov::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, BroadcastAndPadZeroPointBuffers_3) {
    ov::Strides strides{1};
    ov::Strides dilations{1};
    ov::CoordinateDiff pads_begin{2};
    ov::CoordinateDiff pads_end{0};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{ 2, 3, 11 });
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

        model = std::make_shared<ov::Model>(ov::NodeVector{ conv }, ov::ParameterVector{ input });
        manager.register_pass<BroadcastAndPadZeroPointBuffers>(32);
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{ 2, 3, 11 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 4, 3, 3 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto azp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32 }, { 1 });
        auto wzp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 32 }, { 1 });
        auto compensation = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 1, 32, 1 }, { 1 });
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

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ conv }, ov::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, BroadcastAndPadZeroPointBuffers_scalar_wzp) {
    ov::Strides strides{1, 1};
    ov::Strides dilations{1, 1};
    ov::CoordinateDiff pads_begin{0, 0};
    ov::CoordinateDiff pads_end{0, 0};
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{ 1, 8, 11, 12 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 8, 8, 3, 3 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto azp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 1, 1, 1 }, { 1 });
        auto wzp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 8, 1, 1 }, { 12 });
        auto compensation = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 1, 8, 1, 1 }, { 1 });
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

        model = std::make_shared<ov::Model>(ov::NodeVector{ conv }, ov::ParameterVector{ input });
        manager.register_pass<BroadcastAndPadZeroPointBuffers>(8, true);
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{ 1, 8, 11, 12 });
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 8, 8, 3, 3 }, { 1 });
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto azp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 8, 1, 1 }, { 1 });
        auto wzp_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{ 1, 1, 1, 1 }, { 12 });
        auto compensation = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{ 1, 8, 1, 1 }, { 1 });
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

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{ conv }, ov::ParameterVector{ input });
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
