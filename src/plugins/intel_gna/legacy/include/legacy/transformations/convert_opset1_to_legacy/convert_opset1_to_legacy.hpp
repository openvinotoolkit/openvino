// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <vector>

namespace ngraph {
namespace pass {

class ConvertOpSet1ToLegacy;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertOpSet1ToLegacy : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ConvertOpSet1ToLegacy", "0");
    bool run_on_model(const std::shared_ptr<ngraph::Function>& m) override;
};
