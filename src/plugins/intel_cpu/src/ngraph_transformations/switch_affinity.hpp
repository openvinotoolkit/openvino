// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace MKLDNNPlugin {
class SwitchAffinity: public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SwitchAffinity(const bool share_constants = true);
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

private:
    bool share_constants;
};
}  // namespace MKLDNNPlugin
