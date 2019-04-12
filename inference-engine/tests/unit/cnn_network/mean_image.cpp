// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include "mean_image.h"

template <>
const std::vector<short> MeanImage<short>::getValue() { return { 1000, 1001, 1002, 1003, 1004, 1005 }; }
template <>
const std::vector<float> MeanImage<float>::getValue() { return  { 10.10f, 11.11f, 12.12f, 13.13f, 14.14f, 15.15f }; }
template <>
const std::vector<uint8_t> MeanImage<uint8_t>::getValue() { return { 10, 11, 12, 13, 14, 15 }; }