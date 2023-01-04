// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

/*
 * Description:
 *     ConvertFqRnnToQuantizedRnn detects RNN / LSTM / GRU_RNN operations
 *     with FQ operations on the inputs and forms a new TypeRelaxed operation
 *     with quantization parameters as runtime parameters of the operation.
 *     @todo add ascii graph examples
 */

namespace ov {
namespace intel_cpu {

class ConvertFqRnnToQuantizedRnn: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertFqRnnToQuantizedRnn", "0");
    ConvertFqRnnToQuantizedRnn();
};

}   // namespace intel_cpu
}   // namespace ov
