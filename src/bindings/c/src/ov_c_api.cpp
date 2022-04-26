// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iterator>
#include <string>
#include <utility>
#include <cassert>
#include <map>
#include <vector>
#include <set>
#include <algorithm>
#include <chrono>
#include <tuple>
#include <memory>
#include <streambuf>
#include <istream>

#include "c_api/ov_c_api.h"
#include "openvino/openvino.hpp"

/**
 * @struct ov_core
 * @brief This struct represents OpenVINO 2.0 Core entity.
 */
struct ov_core {
    ov::Core object;
};

