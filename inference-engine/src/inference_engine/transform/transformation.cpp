// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transform/transformation.hpp>
#include <string>

namespace InferenceEngine {
namespace Transform {

std::string Transformation::getName() const {
    return name;
}

void Transformation::setName(const std::string& name) {
    this->name = name;
}

}  // namespace Transform
}  // namespace InferenceEngine
