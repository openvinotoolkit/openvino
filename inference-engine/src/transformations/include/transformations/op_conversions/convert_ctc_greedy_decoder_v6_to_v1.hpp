// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertCTCGreedyDecoderV6ToV1;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief ConvertCTCGreedyDecoderV6ToV1 covert v6:CTCGreedyDecoderSeqLen into v0::CTCGreedyDecoder.
 */
class ngraph::pass::ConvertCTCGreedyDecoderV6ToV1: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertCTCGreedyDecoderV6ToV1();
};
