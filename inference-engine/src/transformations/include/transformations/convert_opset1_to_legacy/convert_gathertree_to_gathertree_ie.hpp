// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

#include <ngraph/op/gather_tree.hpp>
#include <ngraph_ops/gather_tree_ie.hpp>

namespace ngraph {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ConvertGatherTreeToGatherTreeIE);

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertGatherTreeToGatherTreeIE: public ngraph::pass::GraphRewrite {
public:
    ConvertGatherTreeToGatherTreeIE() : GraphRewrite() {
        convert();
    }

private:
    void convert();
};
