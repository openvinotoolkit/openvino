// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
namespace subgraph {

using bucketizeParamsTuple = std::tuple<InputShape,    // Data shape
                                        InputShape,    // Buckets shape
                                        bool,          // Right edge of interval
                                        ElementType,   // Data input precision
                                        ElementType,   // Buckets input precision
                                        ElementType,   // Output precision
                                        TargetDevice>;  // Device name

class BucketizeLayerTest : public testing::WithParamInterface<bucketizeParamsTuple>,
                           virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<bucketizeParamsTuple>& obj);
    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override;

protected:
    void SetUp() override;
};

}  // namespace subgraph
}  // namespace test
}  // namespace ov
