// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for the Parameter class
 * @file ie_parameter.hpp
 */
#pragma once

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
#include "openvino/runtime/parameter.hpp"

namespace InferenceEngine {

using ov::runtime::Parameter;
using ov::runtime::ParamMap;

#ifdef __ANDROID__
extern template struct INFERENCE_ENGINE_API_CLASS(InferenceEngine::Parameter::RealData<InferenceEngine::Blob::Ptr>);
#endif

}  // namespace InferenceEngine
