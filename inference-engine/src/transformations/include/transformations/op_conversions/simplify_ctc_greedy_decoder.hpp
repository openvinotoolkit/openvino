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
 * Here are steps that transformation performs:
 * 1. Convert seq_len to seq_mask
 *    Receive time and batch dimensions:
 *        T = Gather(ShapeOf(data), Const(0));
 *        N = Gather(ShapeOf(data), Const(1));
 *    Generate 1D tensor:
 *        range1T = Range(Const(1), Const(T), Const(1));
 *    Generate 2D tensor:
 *        upper_bounds = Broadcast(seq_len, [T, N]);
 *    Compute boolean sequence mask:
 *        bool_seq_mask = GreaterEqual(upper_bounds, range1T);
 *    Generate resulted seq mask:
 *        seq_mask = Select(bool_seq_mask, Conts(1.f), Const(0.f));
 * 2. Create CTCGreedyDecoder with original merge_repeated attribute and connect data and resulted seq_mask
 *        decoder = CTCGreedyDecoder(data, seq_mask, ctc_merge_repeated);
 * 3. Normalize output from CTCGreedyDecoder() = output_f and create other output with output_seq_len
 *        Receive the first output with correct classes_index_type:
 *            output_i = Convert(output_f; destination_type=classes_index_type);
 *        Get to know where equal -1:
 *            where_equal_minus1 = Equal(output_i, Const(-1));
 *        Compute output seq mask:
 *            output_seq_mask = Select(where_equal_minus1, Const(0), Const(1));
 *        Receive the second output
 *            output_seq_len = ReduceSum(output_seq_mask, axis=1);
 *        Receive the second output with correct seq_len_type:
 *            output_seq_len_i = Convert(output_seq_len; destination_type=seq_len_type);
 */
class ngraph::pass::SimplifyCTCGreedyDecoder: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    SimplifyCTCGreedyDecoder();
};
