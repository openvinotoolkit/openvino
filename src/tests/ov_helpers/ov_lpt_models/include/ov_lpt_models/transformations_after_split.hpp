// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "openvino/core/model.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class TransformationsAfterSplitFunction {
public:
    static std::shared_ptr<ov::Model> get(const std::string transformationName);

    static std::shared_ptr<ov::Node> getLayerByTransformationName(const std::string transformationName,
                                                                  const ov::Output<ov::Node> parent);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
