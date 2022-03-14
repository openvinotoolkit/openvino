// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertGather0D;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertGather0D decomposes v1::Gather operation into v0::Unsqueeze + v1::Gather + v0::Squeeze pattern when
 * gather indices is scalar
 */
class ngraph::pass::ConvertGather0D : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertGather0D", "0");
    ConvertGather0D();
};
