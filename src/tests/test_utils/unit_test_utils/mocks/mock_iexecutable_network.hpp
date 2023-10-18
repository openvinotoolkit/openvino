// Copyright (C) 2018-2023 Intel Corporation
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
    MOCK_METHOD(StatusCode, GetOutputsInfo, (ConstOutputsDataMap&, ResponseDesc*), (const, noexcept));
    MOCK_METHOD(StatusCode, GetInputsInfo, (ConstInputsDataMap&, ResponseDesc*), (const, noexcept));
    MOCK_METHOD(StatusCode, CreateInferRequest, (IInferRequest::Ptr&, ResponseDesc*), (noexcept));
    MOCK_METHOD(StatusCode, Export, (const std::string&, ResponseDesc*), (noexcept));
    MOCK_METHOD(StatusCode, Export, (std::ostream&, ResponseDesc*), (noexcept));
    MOCK_METHOD(StatusCode, GetExecGraphInfo, (ICNNNetwork::Ptr&, ResponseDesc*), (noexcept));
    MOCK_METHOD(StatusCode,
                SetConfig,
                ((const std::map<std::string, Parameter>& config), ResponseDesc* resp),
                (noexcept));
    MOCK_METHOD(StatusCode,
                GetConfig,
                (const std::string& name, Parameter& result, ResponseDesc* resp),
                (const, noexcept));
    MOCK_METHOD(StatusCode,
                GetMetric,
                (const std::string& name, Parameter& result, ResponseDesc* resp),
                (const, noexcept));
    MOCK_METHOD(StatusCode, GetContext, (RemoteContext::Ptr & pContext, ResponseDesc* resp), (const, noexcept));
};

IE_SUPPRESS_DEPRECATED_END
