// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "ie_plugin.hpp"
#include "ie_input_info.hpp"
#include "ie_icnn_network.hpp"
#include "ie_iexecutable_network.hpp"

#include <cpp_interfaces/impl/ie_executable_network_internal.hpp>
#include <cpp_interfaces/impl/ie_infer_request_internal.hpp>

#include <gmock/gmock.h>
#include "cpp_interfaces/interface/mock_iinfer_request_internal.hpp"
#include "cpp_interfaces/impl/mock_infer_request_internal.hpp"

using namespace InferenceEngine;

class MockExecutableNetworkInternal : public ExecutableNetworkInternal {
public:
    MOCK_METHOD1(setNetworkInputs, void(InputsDataMap));
    MOCK_METHOD1(setNetworkOutputs, void(OutputsDataMap));
    MOCK_METHOD1(CreateInferRequest, void(IInferRequest::Ptr &));
    MOCK_METHOD1(Export, void(const std::string &));
    MOCK_METHOD1(GetMappedTopology, void(std::map<std::string, std::vector<PrimitiveInfo::Ptr>> &));

};

