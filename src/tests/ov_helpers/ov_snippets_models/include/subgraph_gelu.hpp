// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "snippets_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {

class CodegenGeluFunction : public SnippetsFunctionBase {
public:
    CodegenGeluFunction(const std::vector<ov::PartialShape>& inputShapes,
                        ov::element::Type_t precision = ov::element::f32,
                        bool useSubgraph = false);

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;

private:
    bool m_use_subgraph;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov

