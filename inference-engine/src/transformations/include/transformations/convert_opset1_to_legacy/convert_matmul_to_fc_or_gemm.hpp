// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <sstream>
#include <vector>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertMatMulToFCorGemm;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertMatMulToFCorGemm: public ngraph::pass::MatcherPass {
public:
    ConvertMatMulToFCorGemm();
};
