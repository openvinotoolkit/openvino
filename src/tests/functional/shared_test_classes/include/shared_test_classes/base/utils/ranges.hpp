// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"

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
        // NodeTypeInfo: {IntRanges{}, RealRanges{}}
        { ov::op::v0::Erf::type_info, {{{6, -3}}, {{6, -3, 10}} }},
        { ov::op::v1::Divide::type_info, {{{100, 101}}, {{2, 2, 128}} }},
        { ov::op::v1::FloorMod::type_info, {{{4, 2}}, {{2, 2, 128}} }},
        { ov::op::v1::Mod::type_info, {{{4, 2}}, {{2, 2, 128}} }},
        { ov::op::v1::Power::type_info, {{{4, 2}}, {{2, 2, 128}} }},
};

} // namespace utils
} // namespace test
} // namespace ov
