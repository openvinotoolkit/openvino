// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>


namespace ov {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ConvertOpSet1ToLegacy);

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertOpSet1ToLegacy: public ov::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_function(std::shared_ptr<ov::Function> f) override;
};
