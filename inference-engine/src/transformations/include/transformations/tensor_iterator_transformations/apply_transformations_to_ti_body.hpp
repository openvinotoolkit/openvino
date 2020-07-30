// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ApplyTransformationsToTIBody;

}  // namespace pass
}  // namespace ngraph

/*
 * Description:
 * TODO:fill
 *
 * Usage:
 * TODO:fill
 *
 * Callback example:
 * TODO: fill
 *
 */

class ngraph::pass::ApplyTransformationsToTIBody: public ngraph::pass::GraphRewrite {
public:
    explicit ApplyTransformationsToTIBody(ngraph::pass::Manager& manager) : GraphRewrite() {
        apply_transformations_to_ti_body(manager);
    }

private:
    void apply_transformations_to_ti_body(ngraph::pass::Manager& manager);
};
