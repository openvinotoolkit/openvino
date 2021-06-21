// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>

#include <gmock/gmock.h>

#include "ie_input_info.hpp"
#include <cpp_interfaces/interface/ie_iexecutable_network_internal.hpp>

#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinfer_request_internal.hpp"

using namespace InferenceEngine;

class MockIExecutableNetworkInternal : public IExecutableNetworkInternal {
public:
    MOCK_METHOD(ConstOutputsDataMap, GetOutputsInfo, ());
    MOCK_METHOD(ConstInputsDataMap, GetInputsInfo, ());
    MOCK_METHOD(IInferRequestInternal::Ptr, CreateInferRequest, ());
    MOCK_METHOD(void, Export, (const std::string &));
    void Export(std::ostream &) override {};
    MOCK_METHOD(std::vector<IVariableStateInternal::Ptr>, QueryState, ());
    MOCK_METHOD(CNNNetwork, GetExecGraphInfo, ());

    MOCK_METHOD(void, SetConfig, ((const std::map<std::string, Parameter> &)));
    MOCK_METHOD(Parameter, GetConfig, (const std::string &name));
    MOCK_METHOD(Parameter, GetMetric, (const std::string &name));
    MOCK_METHOD(RemoteContext::Ptr, GetContext, ());
    void WrapOstreamExport(std::ostream& networkModel) {
        IExecutableNetworkInternal::Export(networkModel);
    }
};
