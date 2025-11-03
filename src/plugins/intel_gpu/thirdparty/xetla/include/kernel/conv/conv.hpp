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

/// @defgroup xetla_conv XeTLA conv
/// This is a convolution API built on top of xetla group level API to provide a more convenient way
/// to compose a convoultion kernel.

#include "kernel/conv/api.hpp"
#include "kernel/conv/common.hpp"
#include "kernel/conv/dispatch_policy.hpp"
#include "kernel/conv/impl/slicing_fwd_xe.hpp"
#include "kernel/gemm/common.hpp" // since conv use the same checks from namespace details
#include "kernel/gemm/dispatch_policy.hpp" //do we need the separate for conv or just use from gemm_universal?
