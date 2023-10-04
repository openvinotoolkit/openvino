// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/preprocess/preprocess_builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

using preprocessParamsTuple = std::tuple<ov::builder::preprocess::preprocess_func,  // Function with preprocessing
                                         std::string>;                              // Device name

class PrePostProcessTest : public testing::WithParamInterface<preprocessParamsTuple>,
                           virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<preprocessParamsTuple>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
