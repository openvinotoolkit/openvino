// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <string>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph_ops/pad_ie.hpp>

#include "ngraph/op/lrn.hpp"
#include "ngraph/op/constant.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertPadToPadIE;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertPadToPadIE: public ngraph::pass::GraphRewrite {
public:
    ConvertPadToPadIE() : GraphRewrite() {
        convert_pad();
    }

private:
    void convert_pad();
};
