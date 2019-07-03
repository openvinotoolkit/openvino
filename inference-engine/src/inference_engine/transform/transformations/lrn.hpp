// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <transform/transformation.hpp>

namespace InferenceEngine {
namespace Transform {

class TransformationLRN: public Transformation {
public:
    TransformationLRN();
    void execute(Network& network) override;
};

}  // namespace Transform
}  // namespace InferenceEngine
