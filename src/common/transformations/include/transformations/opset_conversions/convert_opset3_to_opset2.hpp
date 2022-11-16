// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertOpSet3ToOpSet2;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertOpSet3ToOpSet2 : public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("ConvertOpSet3ToOpSet2", "0");
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;
};

namespace ngraph {
namespace pass {
using ov::pass::ConvertOpSet3ToOpSet2;
}  // namespace pass
}  // namespace ngraph
