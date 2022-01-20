// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertGather1ToGather7;

class TRANSFORMATIONS_API ConvertGather7ToGather8;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertGather1ToGather7 converts v1::Gather into v7::Gather.
 */
class ngraph::pass::ConvertGather1ToGather7 : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertGather1ToGather7();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertGather7ToGather8 converts v7::Gather into v8::Gather.
 */
class ngraph::pass::ConvertGather7ToGather8 : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertGather7ToGather8();
};
