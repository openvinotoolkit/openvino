// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class ConvertMulOrAddFinally;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertMulOrAddFinally: public ngraph::pass::GraphRewrite {
public:
    OPENVINO_RTTI("ConvertMulOrAddFinally", "0");
    ConvertMulOrAddFinally();
};
