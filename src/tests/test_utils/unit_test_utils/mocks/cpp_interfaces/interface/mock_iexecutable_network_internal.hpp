// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>

#include <cpp_interfaces/interface/ie_iexecutable_network_internal.hpp>
#include <map>
#include <string>
#include <vector>

#include "ie_input_info.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinfer_request_internal.hpp"

using namespace InferenceEngine;

class MockIExecutableNetworkInternal : public IExecutableNetworkInternal {
public:
    MOCK_CONST_METHOD0(GetOutputsInfo, ConstOutputsDataMap(void));
    MOCK_CONST_METHOD0(GetInputsInfo, ConstInputsDataMap(void));
    MOCK_METHOD0(CreateInferRequest, IInferRequestInternal::Ptr(void));
    MOCK_METHOD1(Export, void(const std::string&));
    void Export(std::ostream&) override{};
    MOCK_METHOD0(GetExecGraphInfo, std::shared_ptr<ngraph::Function>(void));

    MOCK_METHOD1(SetConfig, void(const std::map<std::string, Parameter>& config));
    MOCK_CONST_METHOD1(GetConfig, Parameter(const std::string& name));
    MOCK_CONST_METHOD1(GetMetric, Parameter(const std::string& name));
    MOCK_CONST_METHOD0(GetContext, std::shared_ptr<RemoteContext>(void));
    void WrapOstreamExport(std::ostream& networkModel) {
        IExecutableNetworkInternal::Export(networkModel);
    }
};
