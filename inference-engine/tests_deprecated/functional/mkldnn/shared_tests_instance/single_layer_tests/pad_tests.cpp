// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pad_tests.hpp"

PLUGING_CASE(CPU, PadTFTests, 1, { 3, 4 }, in, { 2, 2 }, { 1, 3 }, "constant", 0.f, { 6, 9 },  ref_constant);
PLUGING_CASE(CPU, PadTFTests, 2, { 3, 4 }, in, { 2, 2 }, { 1, 3 },     "edge", 0.f, { 6, 9 },      ref_edge);
PLUGING_CASE(CPU, PadTFTests, 3, { 3, 4 }, in, { 2, 2 }, { 1, 3 },  "reflect", 0.f, { 6, 9 },   ref_reflect);
PLUGING_CASE(CPU, PadTFTests, 4, { 3, 4 }, in, { 2, 2 }, { 1, 3 },"symmetric", 0.f, { 6, 9 }, ref_symmetric);
