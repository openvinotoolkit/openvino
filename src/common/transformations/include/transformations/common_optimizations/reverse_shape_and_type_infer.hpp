// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ReverseShapeAndTypeInfer;

}  // namespace pass
}  // namespace ov

class ov::pass::ReverseShapeAndTypeInfer : public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("ReverseShapeAndTypeInfer", "0");
    bool run_on_model(const std::shared_ptr<ngraph::Function>& f) override;
};
