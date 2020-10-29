// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/ops.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include "transformations/utils/pass_param.hpp"

namespace ngraph {
namespace pass {

    class INFERENCE_ENGINE_API_CLASS(ConvertBatchToSpace);

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertBatchToSpace: public ngraph::pass::GraphRewrite, public ngraph::pass::PassParam  {
public:
    ConvertBatchToSpace() : GraphRewrite(), PassParam() {
        // convert_batch_to_space();
        convert_batch_to_space_ie_side();
    }

private:
    void convert_batch_to_space();
    void convert_batch_to_space_ie_side();
};
