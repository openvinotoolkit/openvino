// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/markup.hpp"

#include <memory>

#include <snippets/itt.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

bool Markup::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    RUN_ON_FUNCTION_SCOPE(Markup);

    for (const auto& op : f->get_ops()) {
        if (is_type<ngraph::opset1::Convolution>(op)) {
            auto& rt = op->get_rt_info();
            rt["LayoutDependent"] = true;
        }
    }

    return false;
}

} // namespace pass
} // namespace snippets
} // namespace ngraph
