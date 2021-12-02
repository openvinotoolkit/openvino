// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "../base/layer_test_utils.hpp"
#include "../base/ov_subgraph.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/preprocess/preprocess_builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace SubgraphTestsDefinitions {

using preprocessParamsTuple = std::tuple<
        ov::builder::preprocess::preprocess_func,  // Function with preprocessing
        std::string>;                              // Device name

class PrePostProcessTest : public testing::WithParamInterface<preprocessParamsTuple>,
                           virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<preprocessParamsTuple> &obj);

protected:
    void SetUp() override;
};

/////////////////////////////////////////////////////

using preprocessParamsTupleDynamic = std::tuple<
        ov::builder::preprocess::preprocess_func_dynamic,  // Function with preprocessing
        std::string>;                                      // Device name

class PrePostProcessTestDynamic : public testing::WithParamInterface<preprocessParamsTupleDynamic>,
                           virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<preprocessParamsTupleDynamic> &obj);

protected:
    void SetUp() override;
};

}  // namespace SubgraphTestsDefinitions
