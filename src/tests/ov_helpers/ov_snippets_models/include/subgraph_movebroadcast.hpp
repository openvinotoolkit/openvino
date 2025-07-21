// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "snippets_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {

class MoveBroadcastFunction : public SnippetsFunctionBase {
public:
    MoveBroadcastFunction(const std::vector<ov::PartialShape>& inputShapes,
                         ov::element::Type_t precision = ov::element::f32);

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov