// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API SimplifyCTCGreedyDecoderSeqLenWithoutBlankIndex;
class TRANSFORMATIONS_API SimplifyCTCGreedyDecoderSeqLenWithBlankIndex;
class TRANSFORMATIONS_API SimplifyCTCGreedyDecoderSeqLen;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief SimplifyCTCGreedyDecoderSeqLenWithoutBlankIndex converts v6:CTCGreedyDecoderSeqLen without BlankIndex parameter
 * into v0::CTCGreedyDecoder.
 */
class ngraph::pass::SimplifyCTCGreedyDecoderSeqLenWithoutBlankIndex: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SimplifyCTCGreedyDecoderSeqLenWithoutBlankIndex();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief SimplifyCTCGreedyDecoderSeqLenWithBlankIndex converts v6:CTCGreedyDecoderSeqLen with BlankIndex parameter
 * into v0::CTCGreedyDecoder.
 * The transformation works only for case when the blank_index input == C-1, where C is the number of classes.
 */
class ngraph::pass::SimplifyCTCGreedyDecoderSeqLenWithBlankIndex: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SimplifyCTCGreedyDecoderSeqLenWithBlankIndex();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief SimplifyCTCGreedyDecoder replaces various sub-graphs with a CTCGreedyDecoderSeqLen op.
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
 * The transformation works only for case when the blank_index input == C-1, where C is the number of classes.
 */
class ngraph::pass::SimplifyCTCGreedyDecoderSeqLen: public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    SimplifyCTCGreedyDecoderSeqLen() {
        add_matcher<ngraph::pass::SimplifyCTCGreedyDecoderSeqLenWithBlankIndex>();
        add_matcher<ngraph::pass::SimplifyCTCGreedyDecoderSeqLenWithoutBlankIndex>();
    }

    static ngraph::matcher_pass_callback simplify_ctc_greedy_decoder_seq_len();
};