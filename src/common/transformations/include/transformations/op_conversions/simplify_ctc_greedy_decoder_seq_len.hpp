// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API SimplifyCTCGreedyDecoderSeqLen;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
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
 * The transformation works only for case when the blank_index input == C-1, where C is the number of classes.
 */
class ov::pass::SimplifyCTCGreedyDecoderSeqLen : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SimplifyCTCGreedyDecoderSeqLen");
    SimplifyCTCGreedyDecoderSeqLen();
};
