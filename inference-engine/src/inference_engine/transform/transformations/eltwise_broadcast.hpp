// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <transform/transformation.hpp>

namespace InferenceEngine {
namespace Transform {

class TransformationEltwiseBroadcast: public Transformation {
public:
    TransformationEltwiseBroadcast();
    void execute(Network& network) override;
};

}  // namespace Transform
}  // namespace InferenceEngine
