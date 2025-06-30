// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

typedef std::tuple<ov::element::Type,  // Input type
                   ov::element::Type,  // Output type
                   ov::Shape,          // Input shapes
                   std::string         // Device name
                   >
    ShapeOfParams;

class ReluShapeOfSubgraphTest : public testing::WithParamInterface<ov::test::ShapeOfParams>,
                                virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ov::test::ShapeOfParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
