// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ov_models/builders.hpp"

namespace SubgraphTestsDefinitions {

using parameterResultParams = std::tuple<ov::test::InputShape, // Input shape
                                         std::string>;         // Device name

class ParameterResultSubgraphTestBase : public testing::WithParamInterface<parameterResultParams> {
    public:
        static std::string getTestCaseName(const testing::TestParamInfo<parameterResultParams>& obj);
    protected:
        std::shared_ptr<ov::Model> createModel(const ov::PartialShape& shape);
};

class ParameterResultSubgraphTestLegacyApi : public ParameterResultSubgraphTestBase,
                                             virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override;
};

class ParameterResultSubgraphTest : public ParameterResultSubgraphTestBase,
                                    virtual public ov::test::SubgraphBaseTest {
protected:
    void SetUp() override;
};

}  // namespace SubgraphTestsDefinitions
