// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API SimplifyCTCGreedyDecoder;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief SimplifyCTCGreedyDecoder covert v6:CTCGreedyDecoderSeqLen into v0::CTCGreedyDecoder.
 */
class ngraph::pass::SimplifyCTCGreedyDecoder: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SimplifyCTCGreedyDecoder();
};
