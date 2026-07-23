// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>

#include "common_test_utils/node_builders/constant.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov::test {

// Not a run()/reference-comparison test: the fix saturates the overflowing output to f16 range,
// which would read as a large error against the (finite, non-overflowing) fp32 reference, so we
// only assert finiteness.
class ClampFP16FCOutputOverflowCPUTest : public ::testing::Test {
protected:
    static constexpr size_t kRows = 4;
    static constexpr size_t kCols = 64;
};

TEST_F(ClampFP16FCOutputOverflowCPUTest, ResidualAddStaysFiniteUnderF16) {
    ov::Core core;
    const auto capabilities = core.get_property(ov::test::utils::DEVICE_CPU, ov::device::capabilities);
    if (std::find(capabilities.begin(), capabilities.end(), "FP16") == capabilities.end()) {
        GTEST_SKIP() << "No FP16 support";
    }

    auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, kRows, kCols});
    // magnitude chosen so accumulation over kCols exceeds f16's ~65504 range
    ov::test::utils::InputGenerateData weight_data(500, 200, 1);
    auto weight = ov::test::utils::make_constant(ov::element::f32, ov::Shape{kCols, kCols}, weight_data);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(activation, weight, false, true);

    auto residual = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, kRows, kCols});
    auto add = std::make_shared<ov::op::v1::Add>(matmul, residual);

    auto model = std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<ov::op::v0::Result>(add)},
                                              ov::ParameterVector{activation, residual},
                                              "ClampFP16FCOutputOverflow");

    ov::AnyMap config{{ov::hint::inference_precision.name(), ov::element::f16}};
    auto compiled = core.compile_model(model, ov::test::utils::DEVICE_CPU, config);
    auto infer_request = compiled.create_infer_request();

    ov::test::utils::InputGenerateData activation_data(1, 2, 1);
    auto activation_tensor =
        ov::test::utils::create_and_fill_tensor(ov::element::f32, ov::Shape{1, kRows, kCols}, activation_data);
    ov::test::utils::InputGenerateData residual_data(0, 2, 100);
    auto residual_tensor =
        ov::test::utils::create_and_fill_tensor(ov::element::f32, ov::Shape{1, kRows, kCols}, residual_data);

    infer_request.set_tensor(model->inputs()[0], activation_tensor);
    infer_request.set_tensor(model->inputs()[1], residual_tensor);
    infer_request.infer();

    auto output = infer_request.get_output_tensor(0);
    const auto* data = output.data<float>();
    size_t nan_count = 0;
    size_t inf_count = 0;
    for (size_t i = 0; i < output.get_size(); ++i) {
        if (std::isnan(data[i])) {
            ++nan_count;
        } else if (std::isinf(data[i])) {
            ++inf_count;
        }
    }
    EXPECT_EQ(nan_count, 0u) << "ClampFP16FCOutput should prevent NaN from an overflowing FC/MatMul "
                                "output feeding a residual Add";
    EXPECT_EQ(inf_count, 0u) << "ClampFP16FCOutput should saturate an overflowing FC/MatMul output "
                                "to a finite value before it reaches the residual Add";
}

}  // namespace ov::test
