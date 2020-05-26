// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <string>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ConvertOneHotToOneHotIE);

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertOneHotToOneHotIE: public ngraph::pass::GraphRewrite {
public:
    ConvertOneHotToOneHotIE() : GraphRewrite(), is_f16(false) {
        convert_one_hot();
    }

    bool run_on_function(std::shared_ptr<ngraph::Function> f) final;

private:
    void convert_one_hot();
    bool is_f16;
};
