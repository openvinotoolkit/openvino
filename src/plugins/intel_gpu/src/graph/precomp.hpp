// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pch/precomp_core.hpp"

#include "primitive_inst.h"
#include "program_node.h"
#include "pass_manager.h"
#include "primitive_type_base.h"
#include "data_inst.h"
#include "impls/ocl/primitive_base.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/graph/topology.hpp"

