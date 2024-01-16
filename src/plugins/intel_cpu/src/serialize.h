// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <functional>
#include <iostream>

#include "cpp/ie_cnn_network.h"

namespace ov {
namespace intel_cpu {

class ModelSerializer {
public:
    ModelSerializer(std::ostream& ostream);
    void operator<<(const std::shared_ptr<ov::Model>& model);

private:
    std::ostream& _ostream;
};

class ModelDeserializer {
public:
    typedef std::function<std::shared_ptr<ov::Model>(const std::string&, const ov::Tensor&)> model_builder;
    ModelDeserializer(std::istream& istream, model_builder fn);
    void operator>>(std::shared_ptr<ov::Model>& model);

private:
    std::istream& _istream;
    model_builder _model_builder;
};

}   // namespace intel_cpu
}   // namespace ov
