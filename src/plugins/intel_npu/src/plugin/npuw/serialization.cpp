// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "serialization.hpp"

void ov::npuw::s11n::write(std::ostream& stream, const std::string& var) {
    auto var_size = var.size();
    stream.write(reinterpret_cast<char*>(&var_size), sizeof var_size);
    stream.write(var.c_str(), var.size());
}

void ov::npuw::s11n::read(std::istream& stream, std::string& var) {
    std::size_t var_size = 0;
    stream.read(reinterpret_cast<char*>(&var_size), sizeof var_size);
    stream.read(var.data(), var_size);
}
