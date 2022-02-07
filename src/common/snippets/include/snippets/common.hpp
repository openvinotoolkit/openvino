// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie/ie_common.h>
#include <openvino/util/log.hpp>

/**
 * @def SNIPPETS_THROW
 * @brief A macro used for snippet transformations
 * to throw specified exception with a description.
 */
#define SNIPPETS_THROW(...) IE_THROW(__VA_ARGS__) << "[SNIPPETS] "

/**
 * @def SNIPPETS_DEBUG
 * @brief A macro used for snippet transformations to log debug info.
 */
#define SNIPPETS_DEBUG OPENVINO_DEBUG << "[SNIPPETS] "
