// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/openvino.hpp>
#include <string>

namespace transformation_sample {

class ModelTransformer {
public:
    virtual ~ModelTransformer() = default;
    virtual void transform(std::shared_ptr<ov::Model> model) const = 0;
};

}  // namespace transformation_sample
