// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/graph_iterator.hpp"

using namespace ov::frontend;

std::map<std::string, std::string> GraphIterator::get_input_names_map() const {
    return {};
}

std::map<std::string, std::string> GraphIterator::get_output_names_map() const {
    return {};
}
