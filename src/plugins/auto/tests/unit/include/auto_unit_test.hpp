// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include "plugin.hpp"
#include "openvino/runtime/core.hpp"
#include "gmock_plugin.hpp"
#include "mock_common.hpp"
#include <common_test_utils/test_constants.hpp>
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp"

using ::testing::MatcherCast;
using ::testing::AllOf;
using ::testing::Throw;
using ::testing::Matches;
using ::testing::_;
using ::testing::StrEq;
using ::testing::StrNe;
using ::testing::Return;
using ::testing::Property;
using ::testing::Eq;
using ::testing::ReturnRef;
using ::testing::AtLeast;
using ::testing::AnyNumber;
using ::testing::InvokeWithoutArgs;
using ::testing::HasSubstr;
using ::testing::NiceMock;

using namespace ov::mock_auto_plugin;

#define EXPECT_THROW_WITH_MESSAGE(stmt, etype, whatstring) EXPECT_THROW( \
        try { \
            stmt; \
        } catch (const etype& ex) { \
            EXPECT_THAT(std::string(ex.what()), HasSubstr(whatstring)); \
            throw; \
        } \
    , etype)

// define a matcher to check if perf hint expects
MATCHER_P(ComparePerfHint, perfHint, "Check if perf hint expects.") {
    ov::Any arg_perfHint = "No PERFORMANCE_HINT";
    auto itor = arg.find(ov::hint::performance_mode.name());
    if (itor != arg.end()) {
        arg_perfHint = itor->second;
    }

    return perfHint == arg_perfHint.as<std::string>();
}
namespace ov {
namespace mock_auto_plugin {
namespace tests {


class BaseTest {
public:
    std::shared_ptr<ov::Model>                      model;
    std::shared_ptr<NiceMock<MockPluginBase>> mock_plugin_cpu;
    std::shared_ptr<NiceMock<MockPluginBase>> mock_plugin_gpu;
    std::shared_ptr<NiceMock<MockAutoPlugin>>       plugin;
    //mock exeNetwork helper
    ov::SoPtr<ov::MockCompiledModel>  mockExeNetwork;
    std::shared_ptr<ov::MockCompiledModel>   mockIExeNet;
    //mock exeNetwork actual
    ov::SoPtr<ov::MockCompiledModel>  mockExeNetworkActual;
    std::shared_ptr<ov::MockCompiledModel>   mockIExeNetActual;
    // config for Auto device
    ov::AnyMap              config;
    std::vector<DeviceInformation>                metaDevices;
    std::shared_ptr<ov::MockSyncInferRequest>     inferReqInternal;
    std::shared_ptr<ov::MockSyncInferRequest>     inferReqInternalActual;

    ov::Any optimalNum;
    virtual ~BaseTest();
    BaseTest();

protected:
    std::shared_ptr<ov::Model> create_model();
};
// for auto unit tests which can covered by mock core, or need to test with gmock icore
class AutoTest : public BaseTest {
public:
    std::shared_ptr<NiceMock<MockICore >>           core;
    AutoTest();
    ~AutoTest();
};

// for unit tests which requires real core, batch support or remote context
// mock plugin name: MOCK_CPU,MOCK_HARDWARE
// please extend as needed

class AutoTestWithRealCore : public BaseTest {
public:
    AutoTestWithRealCore();
    ~AutoTestWithRealCore() = default;
    ov::Core core;

protected:
    void register_plugin_simple(ov::Core& core, const std::string& device_name, const ov::AnyMap& properties);
    void register_plugin_support_batch_and_context(ov::Core& core, const std::string& device_name, const ov::AnyMap& properties);
    std::vector<std::shared_ptr<ov::IRemoteContext>> m_mock_contexts;
    std::shared_ptr<void> m_so;
    std::shared_ptr<ov::ICompiledModel> compiled_model;
    void reg_plugin(ov::Core& core,
                    std::shared_ptr<ov::IPlugin> plugin,
                    const std::string& device_name,
                    const ov::AnyMap& properties);
};
}  // namespace tests
}  // namespace mock_auto_plugin
}  // namespace ov