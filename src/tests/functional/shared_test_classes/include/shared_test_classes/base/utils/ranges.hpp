// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/op/proposal.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/mod.hpp"
#include "ngraph/op/floor_mod.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/erf.hpp"
#include "ngraph/op/non_max_suppression.hpp"
#include "ngraph/op/reduce_l1.hpp"
#include "ngraph/op/reduce_l2.hpp"
#include "ngraph/op/reduce_sum.hpp"
#include "ngraph/op/reduce_prod.hpp"
#include "ngraph/op/reduce_mean.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/min.hpp"

#include "openvino/op/dft.hpp"
#include "openvino/op/idft.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/op/logical_not.hpp"

#include <map>
#include <vector>

namespace ov {
namespace test {
namespace utils {

// todo: remove w/a to generate correct constant data (replace parameter to const) in conformance with defined range
struct ConstRanges {
    static double max, min;
    static bool is_defined;

    static void set(double _min, double _max) {
        min = _min;
        max = _max;
        is_defined = true;
    }

    static void reset() {
        min = std::numeric_limits<double>::max();
        max = std::numeric_limits<double>::min();
        is_defined = false;
    }
};

struct InputGenerateData {
    double_t start_from;
    uint32_t range;
    int32_t resolution;
    int seed;

    InputGenerateData(double_t _start_from = 0, uint32_t _range = 10, int32_t _resolution = 1, int _seed = 1)
            : start_from(_start_from), range(_range), resolution(_resolution), seed(_seed) {
        if (ConstRanges::is_defined) {
            auto min_orig = start_from;
            auto max_orig = start_from + range * resolution;
            auto min_ref = ConstRanges::min;
            auto max_ref = ConstRanges::max;
            if (min_orig < min_ref || min_orig == 0)
                start_from = min_ref;
            range = (max_orig > max_ref || max_orig == 10 ? max_ref : max_orig - start_from) - start_from;
        }
    }
};

static std::map<ov::NodeTypeInfo, std::vector<std::vector<InputGenerateData>>> inputRanges = {
        // NodeTypeInfo: {IntRanges{}, RealRanges{}} (Ranges are used by generate<ov::Node>)
        { ov::op::v0::Erf::get_type_info_static(), {{{-3, 6}}, {{-3, 6, 10}}} },
        { ov::op::v1::Divide::get_type_info_static(), {{{101, 100}}, {{2, 2, 128}}} },
        { ov::op::v1::FloorMod::get_type_info_static(), {{{2, 4}}, {{2, 2, 128}}} },
        { ov::op::v1::Mod::get_type_info_static(), {{{2, 4}}, {{2, 2, 128}}} },
        { ov::op::v1::ReduceMax::get_type_info_static(), {{{0, 5}}, {{-5, 5, 1000}}} },
        { ov::op::v1::ReduceMean::get_type_info_static(), {{{0, 5}}, {{0, 5, 1000}}} },
        { ov::op::v1::ReduceMin::get_type_info_static(), {{{0, 5}}, {{0, 5, 1000}}} },
        { ov::op::v1::ReduceProd::get_type_info_static(), {{{0, 5}}, {{0, 5, 1000}}} },
        { ov::op::v1::ReduceSum::get_type_info_static(), {{{0, 5}}, {{0, 5, 1000}}} },
        { ov::op::v1::ReduceSum::get_type_info_static(), {{{0, 5}}, {{0, 5, 1000}}} },
        { ov::op::v1::ReduceSum::get_type_info_static(), {{{0, 5}}, {{0, 5, 1000}}} },
        { ov::op::v1::Power::get_type_info_static(), {{{2, 4}}, {{2, 2, 128}}} },
        { ov::op::v4::Proposal::get_type_info_static(), {{{0, 1, 1000, 8234231}}, {{0, 1, 1000, 8234231}}} },
        { ov::op::v4::ReduceL1::get_type_info_static(), {{{0, 5}}, {{0, 5, 1000}}} },
        { ov::op::v4::ReduceL2::get_type_info_static(), {{{0, 5}}, {{0, 5, 1000}}} },
        { ov::op::v7::DFT::get_type_info_static(), {{{0, 1}}, {{0, 1, 1000000}}} },
        { ov::op::v1::LogicalAnd::get_type_info_static(), {{{0, 2}}, {{0, 2, 1}}} },
        { ov::op::v1::LogicalOr::get_type_info_static(), {{{0, 2}}, {{0, 2, 1}}} },
        { ov::op::v1::LogicalNot::get_type_info_static(), {{{0, 2}}, {{0, 2, 1}}} },
        { ov::op::v1::LogicalXor::get_type_info_static(), {{{0, 2}}, {{0, 2, 1}}} },
};

} // namespace utils
} // namespace test
} // namespace ov
