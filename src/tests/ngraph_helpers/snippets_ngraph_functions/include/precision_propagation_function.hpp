// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include <low_precision/layer_transformation.hpp>

namespace ov {
namespace test {
namespace snippets {

class PrecisionPropagationFunction {
public:
    static std::shared_ptr<ngraph::Function> get(
        const ngraph::element::Type precision1,
        const ngraph::PartialShape& inputShape1,
        const ngraph::element::Type precision2,
        const ngraph::PartialShape& inputShape2);
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
