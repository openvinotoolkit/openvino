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
using InferenceEngine::Blob;
using nghraphSubgraphFuncType = std::function<std::shared_ptr<ngraph::Function>(std::vector<size_t>, ngraph::element::Type)>;
using networkPreprocessFuncType = std::function<void(InferenceEngine::CNNNetwork&)>;
using generateInputFuncType = std::function<Blob::Ptr(const InferenceEngine::InputInfo)>;
using generateReferenceFuncType = std::function<std::vector<std::vector<uint8_t>>(InferenceEngine::CNNNetwork&, std::vector<Blob::Ptr>&)>;
using makeTestBlobFuncType = std::function<Blob::Ptr(Blob::Ptr, InferenceEngine::ExecutableNetwork&)>;
using teardownFuncType = std::function<void()>;

typedef std::tuple<
    ngraph::element::Type,                                   // input type
    std::pair<nghraphSubgraphFuncType, std::vector<size_t>>, // subgraph function with input size vector
    std::tuple<networkPreprocessFuncType,                    // function responsible for network preprocessing
               generateInputFuncType,                        // function responsible for generating input
               generateReferenceFuncType,                    // function responsible for generating reference output
               makeTestBlobFuncType,                         // function responsible for creating tested blob type
               teardownFuncType>,                            // teardown function
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

    networkPreprocessFuncType preprocessFn;
    generateInputFuncType generateInputFn;
    generateReferenceFuncType generateReferenceFn;
    makeTestBlobFuncType makeBlobFn;
    teardownFuncType teardownFn;

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;
    std::vector<std::vector<uint8_t>> CalculateReference();
    InferenceEngine::Blob::Ptr PrepareTestBlob(InferenceEngine::Blob::Ptr inputBlob);

    void SetUp() override;
    void Run() override;
    void Infer() override;
    void Validate() override;
};
} // namespace BlobTestsDefinitions
