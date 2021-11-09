// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace SubgraphTestsDefinitions {

using AutoInferConsistencyParamsTuple = typename std::tuple<
        size_t,                             // infer numbers
        std::string,                        // model file
        std::string,                        // device name
        std::string,                        // base device
        std::map<std::string, std::string>,  // auto_configuration
        std::map<std::string, std::string>  // base_cpu_configuration
>;


class AutoInferConsistency : public testing::WithParamInterface<AutoInferConsistencyParamsTuple>,
                                 virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ParamType>& obj);

protected:
    void SetUp() override;
    size_t getInferCount() const { return _inferCount;}
    void loadBaseNet();
    void inferBaseNet();
    void GenerateInputs() override;
    void Infer() override;
    std::vector<InferenceEngine::Blob::Ptr> getBaseNetOutputs();
    size_t _inferCount;
    InferenceEngine::ExecutableNetwork _baseExecNet;
    InferenceEngine::InferRequest      _baseInferRequest;
    std::map<std::string, std::string> _baseConfig;
    std::string                        _baseDevice;
};

}  // namespace SubgraphTestsDefinitions
