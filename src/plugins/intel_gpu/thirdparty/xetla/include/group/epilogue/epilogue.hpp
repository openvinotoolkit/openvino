
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

/// @defgroup xetla_epilogue XeTLA EPILOGUE
/// This is a epilogue API to compute matC = tile_op(matAcc).

#include "group/epilogue/api.hpp"
#include "group/epilogue/common.hpp"
#include "group/epilogue/epilogue_policy.hpp"
#include "group/epilogue/impl/tile_op_xe.hpp"
#include "group/tile_shape.hpp"
