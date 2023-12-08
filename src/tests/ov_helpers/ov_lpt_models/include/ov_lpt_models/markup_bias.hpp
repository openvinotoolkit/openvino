// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/core/model.hpp>

#include "common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class MarkupBiasFunction {
public:
    static std::shared_ptr<ov::Model> get(const ov::element::Type& precision,
                                          const ov::PartialShape& input_shape,
                                          const ov::PartialShape& add_shape,
                                          const std::string& operation_type,
                                          const bool extra_multipy);
};
}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
