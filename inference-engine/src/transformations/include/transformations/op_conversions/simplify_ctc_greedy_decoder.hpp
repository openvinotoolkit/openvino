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
 * @brief SimplifyCTCGreedyDecoder converts v6:CTCGreedyDecoderSeqLen into v0::CTCGreedyDecoder.
 *
 *            data[N, T, C]    seq_len[N]
 *                   \          /
 *            CTCGreedyDecoderSeqLen
 *
 * will be converted to
 *
 *           data[T, N, C]   seq_mask[T, N]
 *                    \         /
 *                  CTCGreedyDecoder
 *                    /         \
 *       class_index[N, T]    seq_len[N]
 *
 * Also the transformation has limitation. It not support blank_index input of CTCGreedyDecoderSeqLen.
 * Transformation support only case with default value when blank index = C-1.
 */
class ngraph::pass::SimplifyCTCGreedyDecoder: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SimplifyCTCGreedyDecoder();
};
