// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API PassParam;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::PassParam {
public:
    using param_callback = std::function<bool(const std::shared_ptr<const ::ngraph::Node>)>;

    explicit PassParam(const param_callback & callback = getDefaultCallback()) : transformation_callback(callback) {}

    void setCallback(const param_callback & callback) {
        transformation_callback = callback;
    }

    static param_callback getDefaultCallback() {
        return [](const std::shared_ptr<const Node> &) -> bool {
            return false;
        };
    }

protected:
    param_callback transformation_callback;
};
