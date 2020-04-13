// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>
#include "transformations/utils/pass_param.hpp"

namespace ngraph {
namespace pass {

    class INFERENCE_ENGINE_API_CLASS(ConvertSpaceToBatch);

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertSpaceToBatch: public ngraph::pass::GraphRewrite, public ngraph::pass::PassParam  {
public:
    ConvertSpaceToBatch() : GraphRewrite(), PassParam() {
        // convert_space_to_batch();
        convert_space_to_batch_by_elements();
    }

private:
    void convert_space_to_batch();
    void convert_space_to_batch_by_elements();
};
