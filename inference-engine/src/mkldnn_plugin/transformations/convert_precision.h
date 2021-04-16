// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations/convert_precision.hpp>

namespace MKLDNNPlugin {
namespace pass {

class ConvertPrecision : public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    bool run_on_function(std::shared_ptr<ngraph::Function> f) final;

    static const std::vector<std::pair<ngraph::element::Type, ngraph::element::Type>> list;
};

}  // namespace pass
}  // namespace MKLDNNPlugin
