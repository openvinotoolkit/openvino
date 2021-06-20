// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief header file for MockIExecutableNetwork
 * \file mock_iexecutable_network.hpp
 */
#pragma once

#include <gmock/gmock.h>

#include <map>
#include <string>
#include <vector>

#include "ie_iexecutable_network.hpp"

using namespace InferenceEngine;

IE_SUPPRESS_DEPRECATED_START

class MockIExecutableNetwork : public IExecutableNetwork {
public:
    MOCK_QUALIFIED_METHOD2(GetOutputsInfo, const noexcept, StatusCode(ConstOutputsDataMap  &, ResponseDesc *));
    MOCK_QUALIFIED_METHOD2(GetInputsInfo, const noexcept, StatusCode(ConstInputsDataMap &, ResponseDesc *));
    MOCK_QUALIFIED_METHOD2(CreateInferRequest, noexcept, StatusCode(IInferRequest::Ptr &, ResponseDesc*));
    MOCK_QUALIFIED_METHOD2(Export, noexcept, StatusCode(const std::string &, ResponseDesc*));
    MOCK_QUALIFIED_METHOD2(Export, noexcept, StatusCode(std::ostream &, ResponseDesc *));
    MOCK_QUALIFIED_METHOD2(GetExecGraphInfo, noexcept, StatusCode(ICNNNetwork::Ptr &, ResponseDesc*));
    MOCK_QUALIFIED_METHOD2(SetConfig, noexcept, StatusCode(const std::map<std::string, Parameter> &config, ResponseDesc *resp));
    MOCK_QUALIFIED_METHOD3(GetConfig, const noexcept, StatusCode(const std::string &name, Parameter &result, ResponseDesc *resp));
    MOCK_QUALIFIED_METHOD3(GetMetric, const noexcept, StatusCode(const std::string &name, Parameter &result, ResponseDesc *resp));
    MOCK_QUALIFIED_METHOD2(GetContext, const noexcept, StatusCode(RemoteContext::Ptr &pContext, ResponseDesc *resp));
};

IE_SUPPRESS_DEPRECATED_END
