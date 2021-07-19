// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUBGRAPH_MULTIPLE_INPUT_HPP
#define SUBGRAPH_MULTIPLE_INPUT_HPP

#include "common_test_utils/test_common.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include <ie_core.hpp>

namespace SubgraphTestsDefinitions {
typedef std::tuple<
    std::string,                        // Target device name
    InferenceEngine::Precision,         // Network precision
    size_t,                             // Input size
    std::map<std::string, std::string>  // Configuration
> multipleInputParams;

class MultipleInputTest : public LayerTestsUtils::LayerTestsCommon,
    public testing::WithParamInterface<multipleInputParams> {
protected:
    void SetUp() override;
public:
    static std::string getTestCaseName(const testing::TestParamInfo<multipleInputParams> &obj);
};
} // namespace SubgraphTestsDefinitions

#endif // SUBGRAPH_MULTIPLE_INPUT_HPP
