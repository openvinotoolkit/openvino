// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "snippets_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {

class BitwiseFunction : public SnippetsFunctionBase {
public:
    BitwiseFunction(const std::vector<PartialShape>& inputShapes, ov::element::Type_t precision);

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
