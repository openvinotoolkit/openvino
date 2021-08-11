// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ConvertMulOrAddFinally);

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertMulOrAddFinally: public ov::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertMulOrAddFinally();
};
