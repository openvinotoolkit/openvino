// Copyright (C) 2020-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/op/util/arithmetic_reductions_keep_dims.hpp"
#include "openvino/op/util/logical_reduction_keep_dims.hpp"

/*
 * Description:
 *     ConvertReduceNoKeepDimsBase detects Reduce operations with keepDims = false.
 *     Such Reduce operation is replaced with Reduce operation with keepDims = true and Squeeze
 *     which removes undesired dimensions.
 *
 * Before:
 * 
 *    +--------------+    +-----------------+
 *    |    Data      |    |   Axes tensor   |
 *    +-----------+--+    +-+---------------+
 *                |         |
 *        +---------------------------+
 *        | Reduce (keepDims = false) |
 *        +---------------------------+
 * 
 * After:
 * 
 *    +--------------+    +-----------------+
 *    |    Data      |    |   Axes tensor   |
 *    +-----------+--+    +-+------------+--+
 *                |         |            |
 *        +---------------------------+  |
 *        | Reduce (keepDims = true)  |  |
 *        +-----------------------+---+  |
 *                                |      |
 *                       +--------v------v-+
 *                       |     Squeeze     |
 *                       +-----------------+
 * 
 */

namespace ov {
namespace intel_cpu {

class ConvertReduceNoKeepDimsBase: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertReduceNoKeepDims", "0");
    template <class T>
    ov::matcher_pass_callback convert_reduce();
};

template <typename ReductionType>
class ConvertReduction: public ConvertReduceNoKeepDimsBase {
public:
    OPENVINO_RTTI("ConvertReduction", "0");
    ConvertReduction();
};


class ConvertReduceNoKeepDims: public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("ConvertReduceNoKeepDims", "0");
    ConvertReduceNoKeepDims() {
        add_matcher<ConvertReduction<ov::op::util::LogicalReductionKeepDims>>();
        add_matcher<ConvertReduction<ov::op::util::ArithmeticReductionKeepDims>>();
    }
};

}   // namespace intel_cpu
}   // namespace ov
