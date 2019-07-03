// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transform/transform_network.hpp>
#include <string>
#include <vector>
#include <map>

namespace InferenceEngine {
namespace Transform {

class Transformation {
    std::string name;
public:
    std::string getName() const;
    void setName(const std::string& name);
    virtual ~Transformation() = default;
    virtual void execute(Network& network) = 0;
};

}  // namespace Transform
}  // namespace InferenceEngine
