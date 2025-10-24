/*******************************************************************************
* Copyright (c) 2022-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/// @file
/// C++ API

#pragma once

#include "kernel/conv/common.hpp"

namespace gpu::xetla::kernel {

/// @addtogroup xetla_conv
/// @{

/// @brief conv_fwd functor.
/// The basic functionality of xetla conv is to calculate the convolution.
///
/// @tparam dispatch_policy Is the dispatch algorithm of the conv implementation.
/// @tparam brconv_t Is the brconv functor.
/// @tparam epilogue_t Is the epilogue functor.
template <typename dispatch_policy, typename brconv_t, typename epilogue_t>
class conv_fwd_t {};

/// @} xetla_conv

} // namespace gpu::xetla::kernel
