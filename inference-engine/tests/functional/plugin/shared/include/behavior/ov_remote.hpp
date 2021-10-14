// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/test_common.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/parameter.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/executable_network.hpp"
#include "openvino/op/parameter.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"

namespace ov {
namespace test {

using RemoteTensorParams = std::tuple<element::Type,        // element type
                                      std::string,          // target device
                                      runtime::ConfigMap,   // config
                                      std::pair<runtime::ParamMap, runtime::ParamMap>>; // remote context and tensor parameters

class OVRemoteTest : public testing::WithParamInterface<RemoteTensorParams>,
                     public CommonTestUtils::TestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<RemoteTensorParams> obj);
protected:
    void SetUp() override;
    void TearDown() override;

    element::Type element_type;
    std::string target_device;
    runtime::ConfigMap config;
    runtime::ParamMap context_parameters;
    runtime::ParamMap tensor_parameters;
    std::shared_ptr<Function> function;
    runtime::Core core = *PluginCache::get().core();
    runtime::ExecutableNetwork exec_network;
    runtime::InferRequest infer_request;
    std::shared_ptr<op::v0::Parameter> input;
};
}  // namespace test
}  // namespace ov
