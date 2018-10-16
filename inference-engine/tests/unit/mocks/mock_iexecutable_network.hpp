// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief header file for MockIExecutableNetwork
 * \file mock_iexecutable_network.hpp
 */
#pragma once

#include "ie_iexecutable_network.hpp"
#include <gmock/gmock-generated-function-mockers.h>

using namespace InferenceEngine;

class MockIExecutableNetwork : public IExecutableNetwork {
public:
    MOCK_QUALIFIED_METHOD2(GetOutputsInfo, const noexcept, StatusCode (ConstOutputsDataMap  &, ResponseDesc *));
    MOCK_QUALIFIED_METHOD2(GetInputsInfo, const noexcept, StatusCode (ConstInputsDataMap &, ResponseDesc *));
    MOCK_QUALIFIED_METHOD2(CreateInferRequest, noexcept, StatusCode(IInferRequest::Ptr &, ResponseDesc*));
    MOCK_QUALIFIED_METHOD2(Export, noexcept, StatusCode(const std::string &, ResponseDesc*));
    MOCK_QUALIFIED_METHOD2(GetMappedTopology, noexcept, StatusCode(std::map<std::string, std::vector<PrimitiveInfo::Ptr>> &, ResponseDesc*));
    MOCK_QUALIFIED_METHOD0(Release, noexcept, void ());
    MOCK_QUALIFIED_METHOD3(QueryState, noexcept, StatusCode(IMemoryState::Ptr &, size_t  , ResponseDesc*));
};
