// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "openvino/core/model.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class TransformationsAfterSplitFunction {
public:
    static std::shared_ptr<ov::Model> get(const std::string transformationName);

    static std::shared_ptr<Node> getLayerByTransformationName(
        const std::string transformationName,
        const Output<Node> parent);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
