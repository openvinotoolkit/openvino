// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

#include <vector>
#include <memory>

namespace vpu {

using Transformations = std::unordered_map<ngraph::NodeTypeInfo, std::function<void(std::shared_ptr<ngraph::Node>)>>;

class DynamicToStaticShape: public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("DynamicToStaticShape", "0");
    explicit DynamicToStaticShape(const Transformations& specificTransformations = {});
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;

    // Keep this method for backward compatibility with other plugins
    void transform(std::shared_ptr<ngraph::Function> function) { run_on_model(function); }
private:
    Transformations transformations;
};

}  // namespace vpu
