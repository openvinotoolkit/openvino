// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"
#include "snippets_helpers.hpp"
#include "subgraph_mlp.hpp"

namespace ov {
namespace test {
namespace snippets {

using MLPParams = std::tuple<
    std::pair<InputShape, std::vector<Shape>>, // InputShape + Weights shape
    MLPFunction::WeightFormat,                 // Weight format
    ov::test::utils::ActivationTypes,          // Activation function type
    ov::element::Type,                         // Inference precision
    size_t,                                    // Expected num nodes
    size_t,                                    // Expected num subgraphs
    std::string,                               // Target Device
    ov::AnyMap                                 // Config
>;

class MLP : public testing::WithParamInterface<MLPParams>, virtual public SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::MLPParams> obj);
protected:
    void SetUp() override;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
