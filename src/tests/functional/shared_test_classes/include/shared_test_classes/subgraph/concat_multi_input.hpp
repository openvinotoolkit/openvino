// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"

namespace SubgraphTestsDefinitions {

typedef std::tuple<
        std::vector<std::vector<size_t>>,   // Input shapes
        InferenceEngine::Precision,         // Network Precision
        std::string,                        // Target Device
        std::map<std::string, std::string>  // Config
> concatMultiParams;

class ConcatMultiInput : public testing::WithParamInterface<concatMultiParams>,
    virtual public LayerTestsUtils::LayerTestsCommon {
private:
    std::vector<size_t> paramSize;
    ngraph::element::Type ngPrc;
    std::vector<std::vector<size_t>> inputShapes;

public:
    void GenerateStridedSliceModel();
    void GenerateConstOnlyModel();
    void GenerateMemoryModel();
    static std::string getTestCaseName(const testing::TestParamInfo<concatMultiParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace SubgraphTestsDefinitions
