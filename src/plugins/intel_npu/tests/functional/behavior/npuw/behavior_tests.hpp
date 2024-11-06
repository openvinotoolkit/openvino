// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <gtest/gtest.h>

#include <memory>
#include <random>

#include "generators/model_generator.hpp"
#include "intel_npu/npuw_private_properties.hpp"
#include "mocks/mock_plugins.hpp"
#include "mocks/register_in_ov.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace ov {
namespace npuw {
namespace tests {

class BehaviorTestsNPUW : public ::testing::Test {
public:
    ov::Core core;
    std::shared_ptr<MockNpuPlugin> mock_npu_plugin;
    std::shared_ptr<MockCpuPlugin> mock_cpu_plugin;

    ov::AnyMap use_npuw_props;
    std::shared_ptr<ov::Model> model;
    ModelGenerator model_generator;

    void SetUp() override {
        mock_npu_plugin = std::make_shared<MockNpuPlugin>();
        mock_npu_plugin->create_implementation();
        mock_cpu_plugin = std::make_shared<MockCpuPlugin>();
        mock_cpu_plugin->create_implementation();
        use_npuw_props = ov::AnyMap{ov::intel_npu::use_npuw(true)};
    }

    // Make sure it is called after expectations are set!
    void register_mock_plugins_in_ov() {
        m_shared_objects.push_back(reg_plugin<MockNpuPlugin>(core, mock_npu_plugin));
        m_shared_objects.push_back(reg_plugin<MockCpuPlugin>(core, mock_cpu_plugin));
    }

private:
    std::vector<std::shared_ptr<void>> m_shared_objects;
};

template <typename T>
using uniform_distribution_common = typename std::conditional<
    std::is_floating_point<T>::value,
    std::uniform_real_distribution<T>,
    typename std::conditional<std::is_integral<T>::value, std::uniform_int_distribution<T>, void>::type>::type;
template <typename T, typename T2>

ov::Tensor make_random_tensor(ov::element::Type type,
                              ov::Shape shape,
                              T rand_min = std::numeric_limits<uint8_t>::min(),
                              T rand_max = std::numeric_limits<uint8_t>::max()) {
    size_t tensor_size = std::accumulate(shape.begin(), shape.end(), std::size_t(1), std::multiplies<size_t>());
    auto tensor = ov::Tensor(type, shape);
    auto data = tensor.data<T>();

    std::mt19937 gen(0);
    uniform_distribution_common<T2> distribution(rand_min, rand_max);
    for (size_t i = 0; i < tensor_size; i++) {
        data[i] = static_cast<T>(distribution(gen));
    }
    return tensor;
}

template <typename T>
void set_random_inputs(ov::InferRequest infer_request) {
    for (const auto& input : infer_request.get_compiled_model().inputs()) {
        // TODO: Extend template header to two types.
        //       For float16 call should be make_random_tensor<ov::float16, float>
        infer_request.set_tensor(input, make_random_tensor<T, T>(input.get_element_type(), input.get_shape()));
    }
}
}  // namespace tests
}  // namespace npuw
}  // namespace ov
