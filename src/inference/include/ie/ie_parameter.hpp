// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for the Parameter class
 * @file ie_parameter.hpp
 */
#pragma once

#ifndef IN_OV_COMPONENT
#    warning("The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#endif

#include <algorithm>
#include <cctype>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <typeinfo>
#include <utility>
#include <vector>

#include "ie_blob.h"
#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"

namespace InferenceEngine {

/**
 * @brief Alias for type that can store any value
 */
using Parameter = ov::Any;
using ParamMap = ov::AnyMap;

}  // namespace InferenceEngine
