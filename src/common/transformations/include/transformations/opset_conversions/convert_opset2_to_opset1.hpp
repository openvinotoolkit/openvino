// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertOpSet2ToOpSet1;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertOpSet2ToOpSet1 : public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("ConvertOpSet2ToOpSet1", "0");
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;
};
