// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include <list>
#include <algorithm>

#include "openvino/opsets/opset.hpp"

#include "common_test_utils/file_utils.hpp"
#include "functional_test_utils/node_utils.hpp"

namespace ov {
namespace test {
namespace conformance {
extern const char* targetDevice;
extern const char *targetPluginName;
extern const char* refCachePath;

extern std::vector<std::string> IRFolderPaths;
extern std::vector<std::string> disabledTests;

enum ShapeMode {
    DYNAMIC,
    STATIC,
    BOTH
};

extern ShapeMode shapeMode;

}  // namespace conformance
}  // namespace test
}  // namespace ov
