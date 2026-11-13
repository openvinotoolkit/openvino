// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/scale_down_gated_mlp_input.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

namespace {
std::shared_ptr<ov::Node> fc(const ov::Output<ov::Node>& in, size_t n, size_t k) {
    auto weights = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{n, k}, {1});
    auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{n, 1}, {1});
    auto zp = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{n, 1}, {1});
    return std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(in, weights, std::make_shared<ov::intel_gpu::op::Placeholder>(), scale, zp);
}

std::shared_ptr<ov::Node> scalar(ov::element::Type type, float value) {
    return ov::op::v0::Constant::create(type, ov::Shape{}, {value});
}

// in -> FC(gate) -> Swish -> Multiply(., FC(up)) -> FC(down) [-> Convert(f32, kept)] -> Result
std::shared_ptr<ov::Model> gated_mlp(bool fixed, bool with_convert, bool swap) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
    std::shared_ptr<ov::Node> gate = std::make_shared<ov::op::v4::Swish>(fc(input, 64, 16));
    if (fixed)
        gate = std::make_shared<ov::op::v1::Multiply>(gate, scalar(ov::element::f16, 1.f / 8.f));
    auto up = fc(input, 64, 16);
    auto product = swap ? std::make_shared<ov::op::v1::Multiply>(up, gate) : std::make_shared<ov::op::v1::Multiply>(gate, up);
    std::shared_ptr<ov::Node> out = fc(product, 16, 64);
    if (with_convert) {
        out = std::make_shared<ov::op::v0::Convert>(out, ov::element::f32);
        ov::disable_conversion(out, ov::element::f16);
    }
    if (fixed)
        out = std::make_shared<ov::op::v1::Multiply>(out, scalar(out->get_output_element_type(0), 8.f));
    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(out)}, ov::ParameterVector{input});
}
}  // namespace

TEST_F(TransformationTestsF, ScaleDownGatedMLPInput_KeptFP32Output) {
    model = gated_mlp(false, true, false);
    manager.register_pass<ScaleDownGatedMLPInput>();
    model_ref = gated_mlp(true, true, false);
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, ScaleDownGatedMLPInput_FP16Output) {
    model = gated_mlp(false, false, false);
    manager.register_pass<ScaleDownGatedMLPInput>();
    model_ref = gated_mlp(true, false, false);
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, ScaleDownGatedMLPInput_SwappedOperands) {
    model = gated_mlp(false, true, true);
    manager.register_pass<ScaleDownGatedMLPInput>();
    model_ref = gated_mlp(true, true, true);
    comparator.enable(FunctionsComparator::CONST_VALUES);
}

TEST_F(TransformationTestsF, ScaleDownGatedMLPInput_NoSwish_NotApplied) {
    auto build = [] {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto product = std::make_shared<ov::op::v1::Multiply>(fc(input, 64, 16), fc(input, 64, 16));
        auto down = fc(product, 16, 64);
        return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(down)}, ov::ParameterVector{input});
    };
    model = build();
    manager.register_pass<ScaleDownGatedMLPInput>();
    model_ref = build();
}

TEST_F(TransformationTestsF, ScaleDownGatedMLPInput_ProductWithTwoConsumers_NotApplied) {
    auto build = [] {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 16});
        auto product = std::make_shared<ov::op::v1::Multiply>(std::make_shared<ov::op::v4::Swish>(fc(input, 64, 16)), fc(input, 64, 16));
        auto down = fc(product, 16, 64);
        return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(down), std::make_shared<ov::op::v0::Result>(product)},
                                           ov::ParameterVector{input});
    };
    model = build();
    manager.register_pass<ScaleDownGatedMLPInput>();
    model_ref = build();
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
