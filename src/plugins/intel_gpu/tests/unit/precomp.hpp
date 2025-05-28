// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pch/precomp_core.hpp"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "test_utils/test_utils.h"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/runtime/execution_config.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/graph/network.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include "primitive_inst.h"
