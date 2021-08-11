// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertGather7ToGather1;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertGather7ToGather1 converts v7::Gather into v1::Gather.
 */
class ov::pass::ConvertGather7ToGather1 : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertGather7ToGather1();
};
