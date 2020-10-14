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

namespace LayerTestsDefinitions {

typedef std::tuple<
            InferenceEngine::Precision,         // Network precision
            std::vector<size_t>,                // Input shapes
            std::string                        // Device name
> paramResultParams;

class ParameterResultSubgraphTest : public testing::WithParamInterface<paramResultParams>,
                                    virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<paramResultParams> obj);
protected:
    void SetUp() override;
};
}  // namespace LayerTestsDefinitions

