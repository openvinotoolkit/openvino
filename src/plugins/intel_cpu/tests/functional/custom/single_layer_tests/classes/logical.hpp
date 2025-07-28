// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "utils/cpu_test_utils.hpp"
#include "gtest/gtest.h"
#include "common_test_utils/test_enums.hpp"

namespace ov {
namespace test {

using LogicalLayerCPUTestParamSet =
    std::tuple<std::vector<ov::test::InputShape>,  // Input shapes
               utils::LogicalTypes,                // Logical type
               ov::test::utils::InputLayerType,    // Second input type
               ov::element::Type,                  // Infer precision
               bool>;                              // Enforce Snippets

class LogicalLayerCPUTest : public testing::WithParamInterface<LogicalLayerCPUTestParamSet>,
                            virtual public ov::test::SubgraphBaseTest,
                            public CPUTestUtils::CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LogicalLayerCPUTestParamSet> &obj);

protected:
    void SetUp() override;

private:
    std::string getPrimitiveType(const utils::LogicalTypes& logical_type) const;
};


namespace logical {

const std::vector<std::vector<InputShape>>& inUnaryShapes();
const std::vector<std::vector<InputShape>>& inBinaryShapes();
const std::vector<ov::test::utils::InputLayerType>& secondInTypes();

const std::vector<utils::LogicalTypes>& logicalUnaryTypes();
const std::vector<utils::LogicalTypes>& logicalUnaryTypesSnippets();

const std::vector<utils::LogicalTypes>& logicalBinaryTypes();
const std::vector<utils::LogicalTypes>& logicalBinaryTypesSnippets();

const std::vector<ov::element::Type> inferPrc();

}  // namespace logical
}  // namespace test
}  // namespace ov
