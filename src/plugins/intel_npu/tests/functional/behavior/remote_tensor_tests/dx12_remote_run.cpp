// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dx12_remote_run.hpp"

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/config/options.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"

#ifdef _WIN32

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> remoteConfigs = {{ov::log::level(ov::log::Level::WARNING)}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         DX12RemoteRunTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(remoteConfigs)),
                         DX12RemoteRunTests::getTestCaseName);

namespace {

std::shared_ptr<ov::Model> getDynamicRemoteRunFunction() {
    const std::vector<size_t> inputShape = {1, 10, 12};
    const ov::element::Type_t ngPrc = ov::element::Type_t::f32;

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape({inputShape}))};
    params.front()->get_output_tensor(0).set_names({"Parameter_1"});

    auto relu = std::make_shared<ov::op::v0::Relu>(params[0]);
    relu->get_output_tensor(0).set_names({"Relu_2"});

    return std::make_shared<ov::Model>(relu, params, "SimpleActivation");
}

auto dynamicRemoteConfigs = []() {
    return std::vector<ov::AnyMap>{{{"NPU_COMPILER_TYPE", "PLUGIN"}, {"NPU_COMPILATION_MODE", "ReferenceSW"}}};
};

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    smoke_BehaviorTests,
    DX12RemoteRunDynamicTests,
    ::testing::Combine(::testing::Values(getDynamicRemoteRunFunction()),
                       ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{
                           {{1, 10, 12}, {1, 10, 12}},
                           {{1, 18, 15}, {1, 18, 15}}}),
                       ::testing::Values(ov::test::utils::DEVICE_NPU),
                       ::testing::ValuesIn(dynamicRemoteConfigs())),
    ov::test::utils::appendPlatformTypeTestName<OVInferRequestDynamicTests>);

#endif
