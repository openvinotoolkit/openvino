// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/visibility.hpp"

#if !defined(HAS_FILESYSTEM) && !defined(HAS_EXP_FILESYSTEM)
#    error "Neither #include <filesystem> nor #include <experimental/filesystem> is available."
#elif defined(HAS_FILESYSTEM)
#    include <filesystem>
namespace std_fs = std::filesystem;
#elif defined(HAS_EXP_FILESYSTEM)
#    include <experimental/filesystem>
namespace std_fs = std::experimental::filesystem;
#endif
