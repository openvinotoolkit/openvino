// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <functional>
#include <ostream>
#include <memory>
#include <string>

#include "openvino/core/model.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace intel_cpu {

class ModelSerializer {
public:
    typedef std::function<std::string(const std::string&)> cache_encrypt;
    ModelSerializer(std::ostream& ostream, cache_encrypt encrypt_fn = {});
    void operator<<(const std::shared_ptr<ov::Model>& model);

private:
    std::ostream& _ostream;
    cache_encrypt _cache_encrypt;
};

class ModelDeserializer {
public:
    typedef std::function<std::shared_ptr<ov::Model>(const std::string&, const ov::Tensor&)> model_builder;
    typedef std::function<std::string(const std::string&)> cache_decrypt;
    ModelDeserializer(std::istream& istream, model_builder fn, cache_decrypt decrypt_fn = {});
    void operator>>(std::shared_ptr<ov::Model>& model);

private:
    std::istream& _istream;
    model_builder _model_builder;
    cache_decrypt _cache_decrypt;
};

}   // namespace intel_cpu
}   // namespace ov
