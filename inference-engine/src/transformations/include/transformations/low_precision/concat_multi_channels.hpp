// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ngraph/ngraph.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

using namespace std;


namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConcatMultiChannelsTransformation: public ngraph::pass::GraphRewrite {
public:
    ConcatMultiChannelsTransformation();
};

}
}
