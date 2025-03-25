// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <gtest/gtest.h>

#include "openvino/op/ops.hpp"
#include "openvino/op/parameter.hpp"
#include "shape_inference/shape_inference.hpp"
#include "shape_inference/static_shape.hpp"

namespace ov {

namespace intel_cpu {

using StaticShapeVector = std::vector<ov::intel_cpu::StaticShape>;

std::vector<StaticShapeRef> make_static_shape_refs(const StaticShapeVector& shapes);

StaticShapeVector shape_inference(ov::Node* op,
                                  const StaticShapeVector& input_shapes,
                                  const std::unordered_map<size_t, Tensor>& constant_data = {});

template <class TOp>
class OpStaticShapeInferenceTest : public testing::Test {
protected:
    using op_type = TOp;

    StaticShapeVector input_shapes, output_shapes;
    ov::intel_cpu::StaticShape exp_shape;
    std::shared_ptr<TOp> op;

    template <class... Args>
    std::shared_ptr<TOp> make_op(Args&&... args) {
        return std::make_shared<TOp>(std::forward<Args>(args)...);
    }
};

}  // namespace intel_cpu
}  // namespace ov
