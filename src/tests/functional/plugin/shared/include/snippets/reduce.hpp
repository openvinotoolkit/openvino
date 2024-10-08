// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"
#include "common_test_utils/test_enums.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<InputShape,                      // Input shape
                   ov::test::utils::ReductionType,  // Reduce type
                   std::vector<int>,                // Reduction axes
                   bool,                            // Keep dims
                   size_t,                          // Expected num nodes
                   size_t,                          // Expected num subgraphs
                   std::string                      // Target device
> ReduceParams;

class Reduce : public testing::WithParamInterface<ov::test::snippets::ReduceParams>,
               virtual public SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::ReduceParams> obj);

protected:
    void SetUp() override;
};

} // namespace snippets
} // namespace test
} // namespace ov