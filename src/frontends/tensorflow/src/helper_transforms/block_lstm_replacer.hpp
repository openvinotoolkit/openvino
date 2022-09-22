// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "openvino/frontend/tensorflow/visibility.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {
namespace pass {

// This transformation handles BlockLSTM with just one output, concatenation of all the intermediate
// output values of the hidden.
class TENSORFLOW_API BlockLSTMToLSTMSequenceOneOutput : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::BlockLSTMToLSTMSequenceOneOutput");
    BlockLSTMToLSTMSequenceOneOutput();
};

}  // namespace pass
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
