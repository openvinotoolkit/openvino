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
namespace ov {
namespace mock_auto_plugin {
namespace tests {

class AutoTest {
public:
    std::shared_ptr<ov::Model>                      model;
    std::shared_ptr<NiceMock<MockICore >>           core;
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
    virtual ~AutoTest();
    AutoTest();

protected:
    std::shared_ptr<ov::Model> create_model();
};

}  // namespace tests
}  // namespace mock_auto_plugin
}  // namespace ov