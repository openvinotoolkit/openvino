// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"



namespace BlobTestsDefinitions {
using nghraphSubgraphFuncType = std::function<std::shared_ptr<ngraph::Function>(std::vector<size_t>, ngraph::element::Type)>;
using makeTestBlobFuncType = std::function<InferenceEngine::Blob::Ptr(InferenceEngine::Blob::Ptr, InferenceEngine::ExecutableNetwork&)>;
typedef std::tuple<
    ngraph::element::Type,                                   // input type
    std::pair<nghraphSubgraphFuncType, std::vector<size_t>>, // subgraph function with input size vector
    makeTestBlobFuncType,                                    // function resposible for creating tested blob type
    std::function<void()>,                                   // teardown function
    size_t,                                                  // infer request number
    size_t,                                                  // batch size
    bool,                                                    // run async
    std::string,                                             // device name
    std::map<std::string, std::string>                       // additional config
    > BlobTestBaseParams;

class BlobTestBase :
    public testing::WithParamInterface<BlobTestBaseParams>,
    public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<BlobTestBaseParams> obj);

protected:
    size_t inferCount;
    bool asyncExecution;

    std::vector<std::vector<InferenceEngine::Blob::Ptr>> refInputBlobs;
    std::vector<std::vector<std::vector<uint8_t>>> referenceOutputs;
    std::vector<std::vector<InferenceEngine::Blob::Ptr>> actualOutputs;
    makeTestBlobFuncType makeBlobFn;
    std::function<void()> teardownFn;

    void SetUp() override;
    void Run() override;
    void Infer() override;
    void Validate() override;
};
}
