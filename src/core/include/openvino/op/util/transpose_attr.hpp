// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace ov {
namespace op {

/** \brief Transpose operator inputs.*/
enum TransposeIn : size_t { ARG, ORDER, IN_COUNT };

/** \brief Transpose operator outputs.*/
enum TransposeOut : size_t { ARG_T, OUT_COUNT };

}  // namespace op
}  // namespace ov
