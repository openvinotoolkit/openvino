// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include "cpp_interfaces/interface/mock_iinfer_request_internal.hpp"
#include "cpp_interfaces/impl/mock_infer_request_internal.hpp"

#include "ie_plugin.hpp"
#include "ie_input_info.hpp"
#include "ie_icnn_network.hpp"
#include "ie_iexecutable_network.hpp"
#include <cpp_interfaces/impl/ie_executable_network_internal.hpp>
#include <cpp_interfaces/impl/ie_infer_request_internal.hpp>

using namespace InferenceEngine;

class MockIExecutableNetworkInternal : public IExecutableNetworkInternal {
public:
    MOCK_CONST_METHOD0(GetOutputsInfo, ConstOutputsDataMap ());
    MOCK_CONST_METHOD0(GetInputsInfo, ConstInputsDataMap ());
    MOCK_METHOD1(CreateInferRequest, void(IInferRequest::Ptr &));
    MOCK_METHOD1(Export, void(const std::string &));
    MOCK_METHOD1(GetMappedTopology, void(std::map<std::string, std::vector<PrimitiveInfo::Ptr>> &));
    MOCK_METHOD0(QueryState, std::vector<IMemoryStateInternal::Ptr>());
    MOCK_METHOD1(GetExecGraphInfo, void(ICNNNetwork::Ptr &));

    MOCK_METHOD2(SetConfig, void(const std::map<std::string, Parameter> &config, ResponseDesc *resp));
    MOCK_CONST_METHOD3(GetConfig, void(const std::string &name, Parameter &result, ResponseDesc *resp));
    MOCK_CONST_METHOD3(GetMetric, void(const std::string &name, Parameter &result, ResponseDesc *resp));
};
