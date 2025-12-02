// Copyright (C) 2018-2025 Intel Corporation
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
        std::vector<int>,                  // logits length
        std::vector<std::vector<int>>,     // labels
        std::vector<int>,                  // labels length
        int,                               // blank index
        bool,                              // preprocessCollapseRepeated
        bool,                              // ctcMergeRepeated
        bool                               // Unique
> CTCLossParamsSubset;

typedef std::tuple<
        CTCLossParamsSubset,
        std::vector<InputShape>,  // Input shapes
        ov::element::Type,        // Float point precision
        ov::element::Type,        // Integer precision
        std::string               // Device name
> CTCLossParams;

class CTCLossLayerTest : public testing::WithParamInterface<CTCLossParams>,
                        virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CTCLossParams> &obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
