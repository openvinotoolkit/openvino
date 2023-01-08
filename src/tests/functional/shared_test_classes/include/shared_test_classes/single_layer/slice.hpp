// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace LayerTestsDefinitions {

struct Slice8SpecificParams {
        std::vector<ov::test::InputShape> shapes;
        std::vector<int64_t> start;
        std::vector<int64_t> stop;
        std::vector<int64_t> step;
        std::vector<int64_t> axes;
};

using Slice8Params = std::tuple<
        Slice8SpecificParams,              // Slice-8 specific parameters
        ov::test::ElementType,             // Net precision
        ov::test::ElementType,             // Input precision
        ov::test::ElementType,             // Output precision
        InferenceEngine::Layout,           // Input layout
        InferenceEngine::Layout,           // Output layout
        std::string,                       // Device name
        std::map<std::string, std::string> // Additional network configuration
>;

class Slice8LayerTest : public testing::WithParamInterface<Slice8Params>,
                        virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<Slice8Params> &obj);

protected:
    void SetUp() override;
};
}  // namespace LayerTestsDefinitions
