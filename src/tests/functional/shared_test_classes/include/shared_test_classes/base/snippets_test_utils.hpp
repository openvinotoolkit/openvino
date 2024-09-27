// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include "ov_subgraph.hpp"

namespace ov {
namespace test {
namespace snippets {
using ov::test::operator<<;
class SnippetsTestsCommon : virtual public ov::test::SubgraphBaseTest {
protected:
    void validateNumSubgraphs();

    void validateOriginalLayersNamesByType(const std::string& layerType, const std::string& originalLayersNames);

    void setInferenceType(ov::element::Type type);

    // Expected num nodes and subgraphs in exec graphs depends on the plugin
    // pipeline, tokenization callback for example. Therefore, they have to be provided manually.
    size_t ref_num_nodes = 0;
    size_t ref_num_subgraphs = 0;
};
}  // namespace snippets
}  // namespace test
}  // namespace ov
