// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace LayerTestsDefinitions {
typedef std::tuple<
    std::vector<size_t>,                 // Input Shapes
    std::vector<int64_t>,                // Axes
    std::vector<int64_t>,                // Dim
    std::vector<int64_t>                 // Offset
> cropParams;

using crop4DParamsTuple = typename std::tuple<
    cropParams,                        // Crop Parameters
    InferenceEngine::Precision,        // Network precision
    std::string,                       // Device name
    std::map<std::string, std::string> // Configuration
>;

class Crop4DLayerTest : public testing::WithParamInterface<crop4DParamsTuple>,
    virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<crop4DParamsTuple> &obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override;

    void Run() override;

protected:
    void SetUp() override;

private:
    std::shared_ptr<ngraph::Function> GenerateNgraphFriendlyModel();
};

}  // namespace LayerTestsDefinitions
