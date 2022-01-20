// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

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
    NGRAPH_RTTI_DECLARATION;
    ConvertGather7ToGather1();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertGather8ToGather7 converts v8::Gather into v7::Gather.
 */
class ngraph::pass::ConvertGather8ToGather7 : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertGather8ToGather7();
};
