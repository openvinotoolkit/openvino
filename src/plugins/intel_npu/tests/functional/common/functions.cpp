//
// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functions.h"
#include "common/vpu_test_env_cfg.hpp"
#include "openvino/op/softmax.hpp"
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

const std::string PlatformEnvironment::PLATFORM = []() -> std::string {
    const auto& var = ov::test::utils::VpuTestEnvConfig::getInstance().IE_NPU_TESTS_PLATFORM;
    if (!var.empty()) {
        return var;
    } else {
        std::cerr << "Environment variable is not set: IE_NPU_TESTS_PLATFORM! Exiting..." << std::endl;
        exit(-1);
    }
}();
