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

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ConvertMatMulToFCorGemm);

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertMatMulToFCorGemm: public ngraph::pass::GraphRewrite {
public:
    ConvertMatMulToFCorGemm(): GraphRewrite() {
        convert_matmul();
    }

private:
    void convert_matmul();
};
