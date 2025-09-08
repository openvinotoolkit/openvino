// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/core/model.hpp"
#include "snippets_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {

/**
 * @class SubgraphRollMatMulRollFunction
 * @brief SubgraphRollMatMulRollFunction instance returns original function.
 *
 * Input arguments are used to create function in getOriginal method only.
 * Dont call getReference and getLowered methods, they are not implemented and throw std::runtime_error exception.
 */
class SubgraphRollMatMulRollFunction : public SnippetsFunctionBase {
public:
    SubgraphRollMatMulRollFunction(const std::vector<ov::PartialShape>& input_shapes, const element::Type input_type);

protected:
    std::shared_ptr<Model> initOriginal() const override;
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
