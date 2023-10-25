// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>

#include <algorithm>
#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace ngraph {
namespace pass {

class ConvertMatMulToFCorGemm;
class ConvertMatMulToFC;
class ConvertMatMulToGemm;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertMatMulToFC : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertMatMulToFC", "0");
    ConvertMatMulToFC();
};

class ngraph::pass::ConvertMatMulToGemm : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertMatMulToGemm", "0");
    ConvertMatMulToGemm();
};

class ngraph::pass::ConvertMatMulToFCorGemm : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("ConvertMatMulToFCorGemm", "0");
    ConvertMatMulToFCorGemm() {
        add_matcher<ngraph::pass::ConvertMatMulToFC>();
        add_matcher<ngraph::pass::ConvertMatMulToGemm>();
    }
};
