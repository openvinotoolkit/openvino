// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "openvino/core/core.hpp"
#include "openvino/core/node.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "shape_inference/static_shape.hpp"
#include "tensor_data_accessor.hpp"

namespace ov::intel_cpu {

class IStaticShapeInfer : public IShapeInfer {
public:
    using IShapeInfer::infer;

    /**
     * @brief Do shape inference.
     *
     * @param input_shapes     Input shapes vector of static shape reference adapter.
     * @param tensor_accessor  Accessor to CPU constant data specific for operator.
     * @return Optionally return vector of static shape adapters holding CPU dimensions.
     */
    virtual std::optional<std::vector<StaticShape>> infer(const std::vector<StaticShapeRef>& input_shapes,
                                                          const ov::ITensorAccessor& tensor_accessor) = 0;

    virtual const std::vector<int64_t>& get_input_ranks() = 0;
};

std::shared_ptr<IStaticShapeInfer> make_shape_inference(std::shared_ptr<ov::Node> op);
}  // namespace ov::intel_cpu
