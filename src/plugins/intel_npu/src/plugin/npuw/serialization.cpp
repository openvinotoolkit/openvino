// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "serialization.hpp"

#include "spatial.hpp"

void ov::npuw::s11n::write(std::ostream& stream, const std::string& var) {
    auto var_size = var.size();
    stream.write(reinterpret_cast<const char*>(&var_size), sizeof var_size);
    stream.write(var.c_str(), var.size());
}

void ov::npuw::s11n::write(std::ostream& stream, const bool& var) {
    stream.write(reinterpret_cast<const char*>(&var), sizeof var);
}

void ov::npuw::s11n::write(std::ostream& stream, const ov::npuw::compiled::Spatial& var) {
    // FIXME: add to overloads
    ov::npuw::s11n::write(stream, var.params.size());
    for (const auto& p : var.params) {
        ov::npuw::s11n::write(stream, p.idx);
        ov::npuw::s11n::write(stream, p.dim);
    }
    ov::npuw::s11n::write(stream, var.range);
    ov::npuw::s11n::write(stream, var.nway);
    ov::npuw::s11n::write(stream, var.out_dim);
    ov::npuw::s11n::write(stream, var.nway_iters);
    ov::npuw::s11n::write(stream, var.tail_size);
}

void ov::npuw::s11n::write(std::ostream& stream, const ov::Tensor& var) {}

void ov::npuw::s11n::read(std::istream& stream, std::string& var) {
    std::size_t var_size = 0;
    stream.read(reinterpret_cast<char*>(&var_size), sizeof var_size);
    stream.read(var.data(), var_size);
}

void ov::npuw::s11n::read(std::istream& stream, bool& var) {
    stream.read(reinterpret_cast<char*>(&var), sizeof var);
}

void ov::npuw::s11n::read(std::istream& stream, ov::npuw::compiled::Spatial& var) {
    // FIXME: add to overloads
    ov::npuw::compiled::Spatial spat;
    std::size_t params_size = 0;
    ov::npuw::s11n::read(stream, params_size);
    for (std::size_t i = 0; i < params_size; ++i) {
        ov::npuw::compiled::Spatial::Param p;
        ov::npuw::s11n::read(stream, p.idx);
        ov::npuw::s11n::read(stream, p.dim);
        spat.params.push_back(p);
    }
    ov::npuw::s11n::read(stream, spat.range);
    ov::npuw::s11n::read(stream, spat.nway);
    ov::npuw::s11n::read(stream, spat.out_dim);
    ov::npuw::s11n::read(stream, spat.nway_iters);
    ov::npuw::s11n::read(stream, spat.tail_size);
}

void ov::npuw::s11n::read(std::istream& stream, ov::Tensor& var) {}
