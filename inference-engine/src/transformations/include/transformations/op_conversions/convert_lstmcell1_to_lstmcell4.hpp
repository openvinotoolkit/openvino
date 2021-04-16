// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <utility>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertLSTMCell1ToLSTMCell4;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertLSTMCell1ToLSTMCell4 covert v0::LSTMCell into v4::LSTMCell.
 */
class ngraph::pass::ConvertLSTMCell1ToLSTMCell4: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertLSTMCell1ToLSTMCell4();
};
