// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/extension.hpp"

#include <iostream>

#include "openvino/core/except.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

using namespace ov;

ov::Extension::~Extension() = default;
ov::BaseOpExtension::~BaseOpExtension() = default;
