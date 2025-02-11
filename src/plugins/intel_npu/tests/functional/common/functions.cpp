// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functions.h"
#include "common/npu_test_env_cfg.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

std::shared_ptr<ov::Model> buildSingleLayerSoftMaxNetwork() {
    ov::Shape inputShape = {1, 3, 4, 3};
    ov::element::Type model_type = ov::element::f32;
    size_t axis = 1;

    const ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(model_type, ov::Shape({inputShape}))};
    params.at(0)->set_friendly_name("Parameter");

    const auto softMax = std::make_shared<ov::op::v1::Softmax>(params.at(0), axis);
    softMax->set_friendly_name("softMax");

    const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(softMax)};
    results.at(0)->set_friendly_name("Result");

    auto ov_model = std::make_shared<ov::Model>(results, params, "softMax");

    return ov_model;
}

std::shared_ptr<ov::Model> createModelWithLargeSize() {
    auto data = std::make_shared<ov::opset11::Parameter>(ov::element::f16, ov::Shape{4000, 4000});
    auto mul_constant = ov::opset11::Constant::create(ov::element::f16, ov::Shape{1}, {1.5});
    auto mul = std::make_shared<ov::opset11::Multiply>(data, mul_constant);
    auto add_constant = ov::opset11::Constant::create(ov::element::f16, ov::Shape{1}, {0.5});
    auto add = std::make_shared<ov::opset11::Add>(mul, add_constant);
    // Just a sample model here, large iteration to make the model large
    for (int i = 0; i < 1000; i++) {
        add = std::make_shared<ov::opset11::Add>(add, add_constant);
    }
    auto res = std::make_shared<ov::opset11::Result>(add);

    /// Create the OpenVINO model
    return std::make_shared<ov::Model>(ov::ResultVector{std::move(res)}, ov::ParameterVector{std::move(data)});
}

const std::string PlatformEnvironment::PLATFORM = []() -> std::string {
    const auto& var = ov::test::utils::NpuTestEnvConfig::getInstance().IE_NPU_TESTS_PLATFORM;
    if (!var.empty()) {
        return var;
    } else {
        std::cerr << "Environment variable is not set: IE_NPU_TESTS_PLATFORM! Exiting..." << std::endl;
        exit(-1);
    }
}();
