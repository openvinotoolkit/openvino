// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <common_test_utils/test_constants.hpp>
#include <common_test_utils/test_assertions.hpp>
#include <memory>

#include "gmock_plugin.hpp"
#include "openvino/runtime/core.hpp"
#include "plugin.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_icompiled_model.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_icore.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_iplugin.hpp"
#include "unit_test_utils/mocks/openvino/runtime/mock_isync_infer_request.hpp"

using namespace ::testing;

using namespace ov::mock_auto_plugin;

#define EXPECT_THROW_WITH_MESSAGE(stmt, etype, whatstring)              \
    EXPECT_THROW(                                                       \
        try { stmt; } catch (const etype& ex) {                         \
            EXPECT_THAT(std::string(ex.what()), HasSubstr(whatstring)); \
            throw;                                                      \
        },                                                              \
        etype)

// define a matcher to check if perf hint expects
MATCHER_P(ComparePerfHint, perfHint, "Check if perf hint expects.") {
    ov::Any arg_perfHint = "No PERFORMANCE_HINT";
    auto itor = arg.find(ov::hint::performance_mode.name());
    if (itor != arg.end()) {
        arg_perfHint = itor->second;
    }

    return perfHint == arg_perfHint.as<std::string>();
}

#define RETURN_MOCK_VALUE(value)  \
    InvokeWithoutArgs([value]() { \
        return ov::Any(value);    \
    })

//  getMetric will return a fake ov::Any, gmock will call ostreamer << ov::Any
//  it will cause core dump, so add this special implemented
namespace testing {
namespace internal {
template <>
void PrintTo<ov::Any>(const ov::Any& a, std::ostream* os);
}
}  // namespace testing

#define ENABLE_LOG_IN_MOCK()                                                     \
    ON_CALL(*(HLogger), print(_)).WillByDefault([&](std::stringstream& stream) { \
        std::cout << stream.str() << std::endl;                                  \
    });

namespace ov {
namespace mock_auto_plugin {
namespace tests {

enum MODELTYPE {
    STATIC = 0,
    DYNAMIC = 1,
};
class BaseTest {
public:
    std::shared_ptr<const ov::Model> model;
    std::shared_ptr<ov::Model> model_can_batch;
    std::shared_ptr<NiceMock<ov::MockIPlugin>> mock_plugin_cpu;
    std::shared_ptr<NiceMock<ov::MockIPlugin>> mock_plugin_gpu;
    std::shared_ptr<NiceMock<MockAutoPlugin>> plugin;
    // mock exeNetwork helper
    ov::SoPtr<ov::MockICompiledModel> mockExeNetwork;
    std::shared_ptr<ov::MockICompiledModel> mockIExeNet;
    // mock exeNetwork actual
    ov::SoPtr<ov::MockICompiledModel> mockExeNetworkActual;
    std::shared_ptr<ov::MockICompiledModel> mockIExeNetActual;
    // config for Auto device
    ov::AnyMap config;
    std::vector<DeviceInformation> metaDevices;
    std::shared_ptr<ov::mock_auto_plugin::MockISyncInferRequest> inferReqInternal;
    std::shared_ptr<ov::mock_auto_plugin::MockISyncInferRequest> inferReqInternalActual;

    ov::Any optimalNum;
    virtual ~BaseTest();
    BaseTest(const MODELTYPE modelType = MODELTYPE::STATIC);

protected:
    std::shared_ptr<ov::Model> create_model();
    std::shared_ptr<ov::Model> create_dynamic_output_model();
};
// for auto unit tests which can covered by mock core, or need to test with gmock icore
class AutoTest : public BaseTest {
public:
    std::shared_ptr<NiceMock<ov::MockICore>> core;
    AutoTest(const MODELTYPE modelType = MODELTYPE::STATIC);
    ~AutoTest();
};
}  // namespace tests
}  // namespace mock_auto_plugin

ACTION_P(Throw, what) {
    OPENVINO_THROW(what);
}
}  // namespace ov
