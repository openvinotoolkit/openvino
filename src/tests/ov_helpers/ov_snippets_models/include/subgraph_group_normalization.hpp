// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "./snippets_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {

class GroupNormalizationFunction : public SnippetsFunctionBase {
public:
    explicit GroupNormalizationFunction(const std::vector<PartialShape>& inputShapes, const size_t& numGroup, const float& eps)
        : SnippetsFunctionBase(inputShapes), num_groups(numGroup), epsilon(eps)  {
        OPENVINO_ASSERT(input_shapes.size() == 3, "Got invalid number of input shapes");
    }

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initLowered() const override;

private:
    size_t num_groups;
    float epsilon;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
