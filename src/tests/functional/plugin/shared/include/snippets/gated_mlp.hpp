// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"
#include "snippets_helpers.hpp"
#include "subgraph_gated_mlp.hpp"

namespace ov {
namespace test {
namespace snippets {

using GatedMLPParams = std::tuple<
    std::pair<InputShape, std::vector<Shape>>, // InputShape + Weights shape
    GatedMLPFunction::WeightFormat,            // Weight format
    ov::test::utils::ActivationTypes,          // Activation function type
    ov::element::Type,                         // Inference precision
    size_t,                                    // Expected num nodes
    size_t,                                    // Expected num subgraphs
    std::string,                               // Target Device
    ov::AnyMap                                 // Config
>;

class GatedMLP : public testing::WithParamInterface<GatedMLPParams>,
                 virtual public SnippetsTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ov::test::snippets::GatedMLPParams>& obj);
protected:
    void SetUp() override;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
