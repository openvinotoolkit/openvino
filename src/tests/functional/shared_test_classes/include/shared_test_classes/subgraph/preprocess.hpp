// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/subgraph_builders/preprocess_builders.hpp"
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

using postprocessParamsTuple = std::tuple<ov::builder::preprocess::postprocess_func,  // Function with postprocessing
                                          std::string>;                               // Device name

class PostProcessTest : public testing::WithParamInterface<postprocessParamsTuple>,
                        virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<postprocessParamsTuple>& obj);

protected:
    void SetUp() override;

public:
    std::string func_name;
};

}  // namespace test
}  // namespace ov

namespace SubgraphTestsDefinitions {

using ov::test::PrePostProcessTest;
using ov::test::preprocessParamsTuple;

}  // namespace SubgraphTestsDefinitions
