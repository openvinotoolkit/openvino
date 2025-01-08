// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "serialization.hpp"

#include "intel_npu/config/config.hpp"
#include "spatial.hpp"

void ov::npuw::s11n::write(std::ostream& stream, const std::string& var) {
    auto var_size = var.size();
    stream.write(reinterpret_cast<const char*>(&var_size), sizeof var_size);
    stream.write(&var[0], var.size());
}

void ov::npuw::s11n::write(std::ostream& stream, const bool& var) {
    stream.write(reinterpret_cast<const char*>(&var), sizeof var);
}

void ov::npuw::s11n::write(std::ostream& stream, const ov::npuw::compiled::Spatial& var) {
    using ov::npuw::s11n::write;
    // FIXME: add to overloads
    write(stream, var.params.size());
    for (const auto& p : var.params) {
        write(stream, p.idx);
        write(stream, p.dim);
    }
    write(stream, var.range);
    write(stream, var.nway);
    write(stream, var.out_dim);
    write(stream, var.nway_iters);
    write(stream, var.tail_size);
}

void ov::npuw::s11n::write(std::ostream& stream, const ov::Tensor& var) {
    using ov::npuw::s11n::write;

    NPUW_ASSERT(var.is_continuous());
    auto type_str = var.get_element_type().to_string();
    write(stream, type_str);
    write(stream, var.get_shape());
    if (type_str != "i4" && type_str != "u4") {
        write(stream, var.get_strides());
    }
    write(stream, var.get_byte_size());
    stream.write(reinterpret_cast<const char*>(var.data()), var.get_byte_size());
}

void ov::npuw::s11n::write(std::ostream& stream, const ::intel_npu::Config& var) {
    write(stream, var.toString());
}

void ov::npuw::s11n::read(std::istream& stream, std::string& var) {
    std::size_t var_size = 0;
    stream.read(reinterpret_cast<char*>(&var_size), sizeof var_size);
    var.resize(var_size);
    stream.read(&var[0], var_size);
}

void ov::npuw::s11n::read(std::istream& stream, bool& var) {
    stream.read(reinterpret_cast<char*>(&var), sizeof var);
}

void ov::npuw::s11n::read(std::istream& stream, ov::npuw::compiled::Spatial& var) {
    using ov::npuw::s11n::read;
    // FIXME: add to overloads
    ov::npuw::compiled::Spatial spat;
    std::size_t params_size = 0;
    read(stream, params_size);
    for (std::size_t i = 0; i < params_size; ++i) {
        ov::npuw::compiled::Spatial::Param p;
        read(stream, p.idx);
        read(stream, p.dim);
        spat.params.push_back(p);
    }
    read(stream, spat.range);
    read(stream, spat.nway);
    read(stream, spat.out_dim);
    read(stream, spat.nway_iters);
    read(stream, spat.tail_size);
}

void ov::npuw::s11n::read(std::istream& stream, ov::Tensor& var) {
    std::string type_str;
    read(stream, type_str);
    ov::element::Type type(type_str);

    ov::Shape shape;
    read(stream, shape);

    ov::Strides strides;
    if (type_str != "i4" && type_str != "u4") {
        read(stream, strides);
    }

    std::size_t byte_size = 0;
    read(stream, byte_size);

    std::vector<char> vec(byte_size, 0);
    stream.read(reinterpret_cast<char*>(&vec[0]), byte_size);

    // Need to get ownership over data, thus creating a temporary tensor over temporary data first
    auto tmp = ov::Tensor(type, shape, vec.data(), strides);
    var = ov::Tensor(type, shape);
    tmp.copy_to(var);
}

void ov::npuw::s11n::read(std::istream& stream, ::intel_npu::Config& var) {
    std::string str_cfg;
    read(stream, str_cfg);

    // Parse cfg from string
    std::map<std::string, std::string> config;

    size_t pos = 0;
    std::string token, key, value;
    while ((pos = str_cfg.find(' ')) != std::string::npos) {
        token = str_cfg.substr(0, pos);
        auto pos_eq = token.find('=');
        key = token.substr(0, pos_eq);
        value = token.substr(pos_eq + 2, token.size() - pos_eq - 3);
        config[key] = value;
        str_cfg.erase(0, pos + 1);
    }

    // Process tail
    auto pos_eq = str_cfg.find('=');
    key = str_cfg.substr(0, pos_eq);
    value = str_cfg.substr(pos_eq + 2, str_cfg.size() - pos_eq - 3);
    config[key] = value;

    var.update(config);
}
