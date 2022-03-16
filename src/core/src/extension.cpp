// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/extension.hpp"

#include <iostream>
#include <unordered_map>

#include "openvino/core/evaluate_extension.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

ov::Extension::~Extension() = default;
ov::BaseOpExtension::~BaseOpExtension() = default;

std::vector<ov::Extension::Ptr>& ov::get_extensions_for_type(const ov::DiscreteTypeInfo& type) {
    static std::unordered_map<ov::DiscreteTypeInfo, std::vector<ov::Extension::Ptr>> all_extensions;
    return all_extensions[type];
}
