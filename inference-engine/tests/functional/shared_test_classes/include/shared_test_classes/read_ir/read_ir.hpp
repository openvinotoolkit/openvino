// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace LayerTestsDefinitions {

using ReadIRParams = std::tuple<
        std::string,                         // IR path
        std::string,                         // Target Device
        std::map<std::string, std::string>>; // Plugin Config

class ReadIRTest : public testing::WithParamInterface<ReadIRParams>,
                   virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReadIRParams> &obj);

protected:
    void SetUp() override;
//    void generate_inputs() override;
//    void Compare(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> &expected,
//                 const std::vector<InferenceEngine::Blob::Ptr> &actual) override;
//    std::vector<InferenceEngine::Blob::Ptr> GetOutputs() override;

private:
    std::string pathToModel;
    std::string sourceModel;
    std::vector<std::pair<std::string, size_t>> ocuranceInModels;
};
} // namespace LayerTestsDefinitions
