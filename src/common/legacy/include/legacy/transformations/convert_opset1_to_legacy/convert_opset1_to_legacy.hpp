// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>


namespace ngraph {
namespace pass {

class ConvertOpSet1ToLegacy;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertOpSet1ToLegacy: public ngraph::pass::FunctionPass {
public:
    OPENVINO_RTTI("ConvertOpSet1ToLegacy", "0");
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;
};
