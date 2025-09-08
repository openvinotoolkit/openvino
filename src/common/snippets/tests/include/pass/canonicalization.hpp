// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lowering_utils.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets_helpers.hpp"
#include "snippets/shape_types.hpp"
#include "snippets/pass/canonicalization.hpp"

namespace ov {
namespace test {
namespace snippets {

class CanonicalizationTests : public TransformationTestsF {
public:
    using VectorDims = ov::snippets::VectorDims;
    using Layout = std::vector<size_t>;
    virtual void run();

protected:
    std::vector<VectorDims> m_input_shapes;
    std::vector<Layout> m_input_layouts;
    void prepare_functions(const std::vector<VectorDims>& shapes);
};

}  // namespace snippets
}  // namespace test
}  // namespace ov