// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace LayerTestsDefinitions {

using softMaxLayerTestParams = std::tuple<
        ngraph::element::Type_t,                                    // netPrecision
//        ngraph::element::Type,                                    // Input precision
//        ngraph::element::Type,                                    // Output precision
//        InferenceEngine::Layout,                                       // Input layout
//        InferenceEngine::Layout,                                       // Output layout
        std::pair<ov::PartialShape, std::vector<ov::Shape>>,   // Dynamic shape + Target static shapes
        size_t,                                                        // axis
        std::string,                                                   // targetDevice
        std::map<std::string, std::string>                             // config
>;

class SoftMaxLayerTest : public testing::WithParamInterface<softMaxLayerTestParams>,
                         virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<softMaxLayerTestParams>& obj);

protected:
    void SetUp() override;
};
}  // namespace LayerTestsDefinitions
