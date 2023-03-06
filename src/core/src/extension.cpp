// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/extension.hpp"

#include <iostream>

#include "openvino/core/except.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"
#include "so_extension.hpp"

ov::Extension::~Extension() = default;
ov::BaseOpExtension::~BaseOpExtension() = default;

ov::detail::SOExtension::~SOExtension() {
    m_ext = {};
}
