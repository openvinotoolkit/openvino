// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
typedef std::tuple<
        std::string,              // Equation
        std::vector<InputShape>   // Input shapes
> EinsumEquationWithInput;

typedef std::tuple<
        ov::element::Type,         // Model type
        EinsumEquationWithInput,   // Equation with corresponding input shapes
        std::string                // Device name
> EinsumLayerTestParamsSet;

class EinsumLayerTest : public testing::WithParamInterface<EinsumLayerTestParamsSet>,
                        virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<EinsumLayerTestParamsSet>& obj);
protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
