// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

using parameterResultParams = std::tuple<ov::test::InputShape,  // Input shape
                                         std::string>;          // Device name

class ParameterResultSubgraphTest : public testing::WithParamInterface<parameterResultParams>,
                                    virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<parameterResultParams>& obj);

protected:
    void SetUp() override;
    std::shared_ptr<ov::Model> createModel(const ov::PartialShape& shape);
};

}  // namespace test
}  // namespace ov
