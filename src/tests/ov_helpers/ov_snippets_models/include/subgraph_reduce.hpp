// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/test_enums.hpp"
#include "./snippets_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {
class ReduceFunction : public SnippetsFunctionBase {
public:
    explicit ReduceFunction(const std::vector<PartialShape>& inputShapes,
                            ov::test::utils::ReductionType reduce_type,
                            const std::vector<int>& axes,
                            bool keep_dims)
        : SnippetsFunctionBase(inputShapes),
          reduce_type(reduce_type),
          axes(axes),
          keep_dims(keep_dims) {
        OPENVINO_ASSERT(input_shapes.size() == 1, "Got invalid number of input shapes");
    }

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;

    ov::test::utils::ReductionType reduce_type;
    std::vector<int> axes;
    bool keep_dims;
};
}  // namespace snippets
}  // namespace test
}  // namespace ov
