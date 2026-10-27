// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/tensor.hpp"

namespace {

std::shared_ptr<ov::Model> make_fp16_camera_model() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 1, 4, 4});
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto weights = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 1, 1, 1}, {0.5f});

    ov::OutputVector branches;
    for (int64_t index = 0; index < 2; ++index) {
        auto gather_index = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {index});
        auto gather = std::make_shared<ov::op::v8::Gather>(input, gather_index, axis, 0);
        branches.push_back(
            std::make_shared<ov::op::v1::Convolution>(gather, weights, ov::Strides{1, 1}, ov::CoordinateDiff{0, 0}, ov::CoordinateDiff{0, 0}, ov::Strides{1, 1})
                ->output(0));
    }

    auto concat = std::make_shared<ov::op::v0::Concat>(branches, 3);
    return std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(concat), ov::ParameterVector{input});
}

bool has_device(const std::vector<std::string>& available_devices, const std::string& device_name) {
    return std::any_of(available_devices.begin(), available_devices.end(), [&device_name](const std::string& available_device) {
        return available_device == device_name ||
               (available_device.size() > device_name.size() && available_device.compare(0, device_name.size(), device_name) == 0 &&
                available_device[device_name.size()] == '.');
    });
}

}  // namespace

TEST(IntelGpuBatchReplicatedBranches, ValidatesFp16NumericalTolerance) {
    auto reference_model = make_fp16_camera_model();

    ov::Tensor input(ov::element::f16, ov::Shape{1, 2, 1, 4, 4});
    auto* input_values = input.data<ov::float16>();
    for (size_t i = 0; i < input.get_size(); ++i) {
        input_values[i] = static_cast<float>(i + 1) * 0.125f;
    }

    ov::Core core;
    const auto available_devices = core.get_available_devices();
    if (!has_device(available_devices, "CPU") || !has_device(available_devices, "GPU")) {
        GTEST_SKIP() << "Requires discoverable CPU and GPU plugins.";
    }

    auto cpu_model = core.compile_model(reference_model, "CPU");
    auto gpu_model = core.compile_model(reference_model, "GPU");

    auto cpu_request = cpu_model.create_infer_request();
    cpu_request.set_input_tensor(input);
    cpu_request.infer();
    const auto cpu_output = cpu_request.get_output_tensor(0);

    auto gpu_request = gpu_model.create_infer_request();
    gpu_request.set_input_tensor(input);
    gpu_request.infer();
    const auto gpu_output = gpu_request.get_output_tensor(0);

    ASSERT_EQ(cpu_output.get_size(), gpu_output.get_size());
    const auto* cpu_values = cpu_output.data<ov::float16>();
    const auto* gpu_values = gpu_output.data<ov::float16>();
    for (size_t i = 0; i < cpu_output.get_size(); ++i) {
        EXPECT_NEAR(static_cast<float>(cpu_values[i]), static_cast<float>(gpu_values[i]), 1e-2f) << "at output index " << i;
    }
}
