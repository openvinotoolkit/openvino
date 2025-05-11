/*******************************************************************************
* Copyright (c) 2022-2023 Intel Corporation
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

#include "kernel/gemm/common.hpp"

namespace gpu::xetla::kernel {

/// @addtogroup xetla_gemm_universal
/// @{

/// @brief GEMM_UNIVERSAL functor.
/// The basic functionality of xetla GEMM_UNIVERSAL is to calculate the \f$C = A \times B\f$.
///
/// For advanced usages, xetla GEMM_UNIVERSAL can also support:
/// - \f$C = tileOp(A \times B)\f$ by configuring the tile_op_t in epilogue.
/// @tparam dispatch_policy Is the dispatch algorithm of the GEMM_UNIVERSAL implementation.
/// @tparam gemm_t Is the gemm functor.
/// @tparam epilogue_t Is the epilogue functor.
template <typename dispatch_policy, typename gemm_t, typename epilogue_t,
        class enable = void>
class gemm_universal_t {};

/// @} xetla_gemm_universal

} // namespace gpu::xetla::kernel
