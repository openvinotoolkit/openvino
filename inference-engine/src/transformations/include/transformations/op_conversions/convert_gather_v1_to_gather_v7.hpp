// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertGather1ToGather7;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertGather1ToGather7 converts v1::Gather into v7::Gather.
 */
class ov::pass::ConvertGather1ToGather7 : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertGather1ToGather7();
};
