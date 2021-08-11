// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>
#include <string>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class INFERENCE_ENGINE_API_CLASS(ConvertLSTMSequenceMatcher);
class INFERENCE_ENGINE_API_CLASS(ConvertGRUSequenceMatcher);
class INFERENCE_ENGINE_API_CLASS(ConvertRNNSequenceMatcher);

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief Converts LSTMSequence to legacy LSTMSequenceIE.
 * SequenceIE op doesn't use seq_length input and num_direction (direction) attribute.
 * We squeeze num_direction dimension for all corresponding inputs and unsqueeze them after the SequenceIE op.
 */

class ov::pass::ConvertLSTMSequenceMatcher : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertLSTMSequenceMatcher();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Converts GRUSequence to legacy GRUSequenceIE.
 * SequenceIE op doesn't use seq_length input and num_direction (direction) attribute.
 * We squeeze num_direction dimension for all corresponding inputs and unsqueeze them after the SequenceIE op.
 */

class ov::pass::ConvertGRUSequenceMatcher : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertGRUSequenceMatcher();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief Converts RNNSequence to legacy RNNSequenceIE.
 * SequenceIE op doesn't use seq_length input and num_direction (direction) attribute.
 * We squeeze num_direction dimension for all corresponding inputs and unsqueeze them after the SequenceIE op.
 */

class ov::pass::ConvertRNNSequenceMatcher : public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertRNNSequenceMatcher();
};
