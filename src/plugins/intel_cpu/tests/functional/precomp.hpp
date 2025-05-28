// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pch/precomp_core.hpp"

#include <regex>

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "gmock/gmock-matchers.h"

#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/op.hpp"
#include "openvino/openvino.hpp"
