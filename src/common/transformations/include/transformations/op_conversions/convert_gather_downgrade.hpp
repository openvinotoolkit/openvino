// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertGather7ToGather1;
class TRANSFORMATIONS_API ConvertGather8ToGather7;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertGather7ToGather1 converts v7::Gather into v1::Gather.
 */
class ngraph::pass::ConvertGather7ToGather1 : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertGather7ToGather1", "0");
    ConvertGather7ToGather1();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertGather8ToGather7 converts v8::Gather into v7::Gather.
 */
class ngraph::pass::ConvertGather8ToGather7 : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertGather8ToGather7", "0");
    ConvertGather8ToGather7();
};
