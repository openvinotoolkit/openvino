// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/single_layer/proposal.hpp>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/single_layer/psroi_pooling.hpp"
#include "shared_test_classes/single_layer/roi_pooling.hpp"
#include "shared_test_classes/single_layer/roi_align.hpp"

namespace LayerTestsDefinitions {

using ReadIRParams = std::tuple<
        std::string,                         // IR path
        std::string,                         // Target Device
        std::map<std::string, std::string>>; // Plugin Config

class ReadIRTest : public testing::WithParamInterface<ReadIRParams>,
                   virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReadIRParams> &obj);
    void QueryNetwork() override;

protected:
    void SetUp() override;
    void GenerateInputs() override;
    void Compare(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> &expected,
                 const std::vector<InferenceEngine::Blob::Ptr> &actual) override;
    std::vector<InferenceEngine::Blob::Ptr> GetOutputs() override;

private:
    std::string pathToModel;
    std::string sourceModel;
    std::vector<std::pair<std::string, size_t>> ocuranceInModels;
};
} // namespace LayerTestsDefinitions
