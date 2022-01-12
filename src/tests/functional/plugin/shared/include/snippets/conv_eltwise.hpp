// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        InferenceEngine::Precision,  // Network Precision
        InferenceEngine::SizeVector, // Input Shape #0
        InferenceEngine::SizeVector, // Input Shape #1
        std::shared_ptr<ov::Node>,   // The first binary eltwise op after the Convolution
        size_t,                      // Expected num nodes
        size_t,                      // Expected num subgraphs
        std::string                  // Target Device
> multiInputParams;

class ConvEltwise : public testing::WithParamInterface<LayerTestsDefinitions::multiInputParams>,
                    virtual public ov::test::SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<LayerTestsDefinitions::multiInputParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
