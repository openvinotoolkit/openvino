// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

/*
 * Description:
 *     ConvertReduceMultiAxisBase detects Reduce operations that do not support
 *     multi-axis input by ACL: Min, Max, Sum, Prod. Multi-axis Reduce operation
 *     is replaced with a sequence of single-axe Reduce operations.
 *
 * Before:
 * 
 * +--------------+    +-------------------+
 * |    Data      |    | Axes tensor [A,B] |
 * +-----------+--+    +-+-----------------+
 *             |         |
 *        +----v---------v----+
 *        |      Reduce       |
 *        +---------+---------+
 *                  |
 *           +------v------+
 *           |   Result    |
 *           +-------------+
 * 
 * After:
 * 
 * +-------------+   +---------------+
 * |   Data      |   | Axes scalar A |
 * +---------+---+   +----+----------+
 *           |            |
 *         +-v------------v--+    +-----------------+
 *         |    Reduce       |    | Axes scalar B   |
 *         +--------------+--+    +---+-------------+
 *                        |           |
 *                      +-v-----------v---+
 *                      |     Reduce      |
 *                      +-------+---------+
 *                              |
 *                      +-------v---------+
 *                      |     Result      |
 *                      +-----------------+
 * 
 */

namespace ov {
namespace intel_cpu {

class ConvertReduceMultiAxisBase: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertReduceMultiAxisBase", "0");
    template <class T>
    ngraph::matcher_pass_callback convert_reduce();
};

class ConvertReduceProd: public ConvertReduceMultiAxisBase {
public:
    OPENVINO_RTTI("ConvertReduceProd", "0");
    ConvertReduceProd();
};

class ConvertReduceMin: public ConvertReduceMultiAxisBase {
public:
    OPENVINO_RTTI("ConvertReduceMin", "0");
    ConvertReduceMin();
};

class ConvertReduceMax: public ConvertReduceMultiAxisBase {
public:
    OPENVINO_RTTI("ConvertReduceMax", "0");
    ConvertReduceMax();
};

class ConvertReduceSum: public ConvertReduceMultiAxisBase {
public:
    OPENVINO_RTTI("ConvertReduceSum", "0");
    ConvertReduceSum();
};

class ConvertReduceMultiAxis: public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("ConvertReduceMultiAxis", "0");
    ConvertReduceMultiAxis() {
        add_matcher<ConvertReduceProd>();
        add_matcher<ConvertReduceMin>();
        add_matcher<ConvertReduceMax>();
        add_matcher<ConvertReduceSum>();
    }
};

}   // namespace intel_cpu
}   // namespace ov
