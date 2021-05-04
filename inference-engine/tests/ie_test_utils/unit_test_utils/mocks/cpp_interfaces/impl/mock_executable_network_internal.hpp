// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>

#include "ie_input_info.hpp"
#include "cpp/ie_cnn_network.h"

#include <cpp_interfaces/impl/ie_executable_network_internal.hpp>

#include <gmock/gmock.h>

#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinfer_request_internal.hpp"

using namespace InferenceEngine;

class MockExecutableNetworkInternal : public ExecutableNetworkInternal {
public:
    MOCK_METHOD1(setNetworkInputs, void(InputsDataMap));
    MOCK_METHOD1(setNetworkOutputs, void(OutputsDataMap));
    MOCK_METHOD0(CreateInferRequest, IInferRequestInternal::Ptr(void));
    MOCK_METHOD1(Export, void(const std::string &));
    MOCK_METHOD0(GetExecGraphInfo, CNNNetwork(void));
    void WrapOstreamExport(std::ostream& networkModel) {
        ExecutableNetworkInternal::Export(networkModel);
    }
    const std::string exportString = "MockExecutableNetworkInternal";
    void ExportImpl(std::ostream& networkModel) override {
        networkModel << exportString << std::endl;
    }
    MOCK_METHOD2(CreateInferRequestImpl, IInferRequestInternal::Ptr(InputsDataMap, OutputsDataMap));
};
