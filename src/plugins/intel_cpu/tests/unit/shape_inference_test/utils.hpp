// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/op/ops.hpp>
#include <openvino/op/parameter.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

#include "utils/shape_inference/static_shape.hpp"

#pragma once

namespace ov {
namespace intel_cpu {
template <class TIface = IShapeInferCommon, class TTensorPtr = HostTensorPtr>
void shape_inference(ov::Node* op,
                     const std::vector<StaticShape>& input_shapes,
                     std::vector<StaticShape>& output_shapes,
                     const std::map<size_t, TTensorPtr>& constant_data = {}) {
    const auto shape_infer = make_shape_inference<TIface>(op->shared_from_this());
    auto result = shape_infer->infer(input_shapes, constant_data);
    OPENVINO_ASSERT(ShapeInferStatus::success == result.status, "shape inference result has unexpected status");
    output_shapes = std::move(result.shapes);
}
}  // namespace intel_cpu
}  // namespace ov

struct TestTensor {
    std::shared_ptr<ngraph::runtime::HostTensor> tensor;
    ov::intel_cpu::StaticShape static_shape;

    template <typename T>
    TestTensor(std::initializer_list<T> values) : TestTensor(ov::intel_cpu::StaticShape({values.size()}), values) {}

    template <typename T>
    TestTensor(T scalar) : TestTensor(ov::intel_cpu::StaticShape({}), {scalar}) {}

    TestTensor(ov::intel_cpu::StaticShape shape) : static_shape(shape) {}

    template <typename T>
    TestTensor(ov::intel_cpu::StaticShape shape, std::initializer_list<T> values) {
        static_shape = shape;

        ov::Shape s;
        for (auto dim : shape)
            s.push_back(dim.get_length());

        if (values.size() > 0) {
            tensor = std::make_shared<ngraph::runtime::HostTensor>(ov::element::from<T>(), s);
            T* ptr = tensor->get_data_ptr<T>();
            int i = 0;
            for (auto& v : values)
                ptr[i++] = v;
        }
    }
};

// TestTensor can be constructed from initializer_list<T>/int64_t/Shape/Shape+initializer_list
// so each element of inputs can be:
//      {1,2,3,4}                   tensor of shape [4] and values (1,2,3,4)
//      2                           tensor of scalar with value 2
//      Shape{2,2}                  tensor of shape [2,2] and value unknown
//      {Shape{2,2}, {1,2,3,4}}     tensor of shape [2,2] and values (1,2,3,4)
inline void check_static_shape(ov::Node* op,
                               std::initializer_list<TestTensor> inputs,
                               std::initializer_list<ov::intel_cpu::StaticShape> expect_shapes) {
    std::vector<ov::intel_cpu::StaticShape> output_shapes;
    std::vector<ov::intel_cpu::StaticShape> input_shapes;
    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> constData;

    int index = 0;
    std::for_each(inputs.begin(), inputs.end(), [&](TestTensor t) {
        input_shapes.push_back(t.static_shape);
        if (t.tensor)
            constData[index] = t.tensor;
        index++;
    });

    output_shapes.resize(expect_shapes.size(), ov::intel_cpu::StaticShape{});

    shape_inference(op, input_shapes, output_shapes, constData);

    EXPECT_EQ(output_shapes.size(), expect_shapes.size());
    int id = 0;
    for (auto& shape : expect_shapes) {
        EXPECT_EQ(output_shapes[id], shape);
        id++;
    }
}

inline void check_output_shape(ov::Node* op, std::initializer_list<ov::PartialShape> expect_shapes) {
    int id = 0;
    EXPECT_EQ(op->outputs().size(), expect_shapes.size());
    for (auto& shape : expect_shapes) {
        EXPECT_EQ(op->get_output_partial_shape(id), shape);
        id++;
    }
}

using ShapeVector = std::vector<ov::intel_cpu::StaticShape>;

template <class TOp>
class OpStaticShapeInferenceTest : public testing::Test {
protected:
    using op_type = TOp;

    ShapeVector input_shapes, output_shapes;
    ov::intel_cpu::StaticShape exp_shape;
    std::shared_ptr<TOp> op;

    template <class... Args>
    std::shared_ptr<TOp> make_op(Args&&... args) {
        return std::make_shared<TOp>(std::forward<Args>(args)...);
    }
};
