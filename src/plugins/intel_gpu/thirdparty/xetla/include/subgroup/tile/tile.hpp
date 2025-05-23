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

#include "subgroup/tile/api.hpp"
#include "subgroup/tile/chained_tile_op.hpp"
#include "subgroup/tile/common.hpp"
#include "subgroup/tile/impl/fma_xe.hpp"
#include "subgroup/tile/impl/load_xe.hpp"
#include "subgroup/tile/impl/mma_xe.hpp"
#include "subgroup/tile/impl/op_function.hpp"
#include "subgroup/tile/impl/payload_xe.hpp"
#include "subgroup/tile/impl/prefetch_xe.hpp"
#include "subgroup/tile/impl/reduction.hpp"
#include "subgroup/tile/impl/store_xe.hpp"
#include "subgroup/tile/impl/tile_op_functor.hpp"
