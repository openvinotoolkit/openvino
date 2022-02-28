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

namespace ov {
namespace test {
namespace utils {

struct InputGenerateData {
    uint32_t range;
    int32_t start_from;
    int32_t resolution;
    int seed;

    InputGenerateData(uint32_t _range = 10, int32_t _start_from = 0, int32_t _resolution = 1, int _seed = 1)
            : range(_range), start_from(_start_from), resolution(_resolution), seed(_seed) {}
};

static std::map<ov::NodeTypeInfo, std::vector<std::vector<InputGenerateData>>> inputRanges = {
        // NodeTypeInfo: {IntRanges{}, RealRanges{}} (Ranges are used by generate<ov::Node>)
        { ov::op::v0::Erf::get_type_info_static(), {{{6, -3}}, {{6, -3, 10}}} },
        { ov::op::v1::Divide::get_type_info_static(), {{{100, 101}}, {{2, 2, 128}}} },
        { ov::op::v1::FloorMod::get_type_info_static(), {{{4, 2}}, {{2, 2, 128}}} },
        { ov::op::v1::Mod::get_type_info_static(), {{{4, 2}}, {{2, 2, 128}}} },
        { ov::op::v1::ReduceMax::get_type_info_static(), {{{5, 0}}, {{5, -5, 1000}}} },
        { ov::op::v1::ReduceMean::get_type_info_static(), {{{5, 0}}, {{5, 0, 1000}}} },
        { ov::op::v1::ReduceMin::get_type_info_static(), {{{5, 0}}, {{5, 0, 1000}}} },
        { ov::op::v1::ReduceProd::get_type_info_static(), {{{5, 0}}, {{5, 0, 1000}}} },
        { ov::op::v1::ReduceSum::get_type_info_static(), {{{5, 0}}, {{5, 0, 1000}}} },
        { ov::op::v1::ReduceSum::get_type_info_static(), {{{5, 0}}, {{5, 0, 1000}}} },
        { ov::op::v1::ReduceSum::get_type_info_static(), {{{5, 0}}, {{5, 0, 1000}}} },
        { ov::op::v1::Power::get_type_info_static(), {{{4, 2}}, {{2, 2, 128}}} },
        { ov::op::v4::Proposal::get_type_info_static(), {{{1, 0, 1000, 8234231}}, {{1, 0, 1000, 8234231}}} },
        { ov::op::v4::ReduceL1::get_type_info_static(), {{{5, 0}}, {{5, 0, 1000}}} },
        { ov::op::v4::ReduceL2::get_type_info_static(), {{{5, 0}}, {{5, 0, 1000}}} },
};

} // namespace utils
} // namespace test
} // namespace ov
