// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "lowered/pass/buffer_allocation.hpp"
#include "openvino/opsets/opset1.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/op/load.hpp"
#include "snippets/op/powerstatic.hpp"
#include "snippets/op/reduce.hpp"
#include "snippets/op/result.hpp"
#include "snippets/op/store.hpp"
#include "transformations/snippets/common/shape_inference.hpp"

namespace ov {
namespace test {
namespace snippets {

class BufferAllocationCPUTest : public BufferAllocationTest {
protected:
    std::shared_ptr<ov::snippets::IShapeInferSnippetsFactory> GetShapeInferFactory() const override {
        return std::make_shared<ov::snippets::CPUShapeInferSnippetsFactory>();
    }
};

template <typename Parent>
class MHAFP32BufferAllocationCPUTest : public Parent {
protected:
    std::shared_ptr<ov::Model> GetModel(const std::vector<ov::PartialShape>& shapes) const override {
        const auto subtensor_scalar = std::vector<size_t>{1};
        const auto subtensor_power = std::vector<size_t>{1, ov::snippets::utils::get_full_dim_value()};
        const auto subtensor_full = std::vector<size_t>(2, ov::snippets::utils::get_full_dim_value());

        // Dims are selected in order to have blocking loops by each dim
        OPENVINO_ASSERT(shapes.size() == 3, "Incorrect count of input shapes");
        const auto parameter0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shapes[0]);
        const auto parameter1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shapes[1]);
        const auto parameter2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shapes[2]);

        const auto order = std::vector<size_t>{0, 2, 3, 1};
        const auto load_reshape = std::make_shared<ov::snippets::op::LoadReorder>(parameter1, 1, 0, order);
        const auto store = std::make_shared<ov::snippets::op::Store>(load_reshape);
        const auto relu0 = std::make_shared<ov::op::v0::Relu>(store);
        const auto matmul0 = create_matmul(parameter0, relu0);

        const auto relu1 = std::make_shared<ov::op::v0::Relu>(matmul0);

        // Decomposed Softmax
        const auto reduce_max = std::make_shared<ov::snippets::op::ReduceMax>(relu1, 3);
        ov::snippets::op::ReduceBase::compute_and_set_reduce_subtensors(reduce_max);
        const auto subtract = std::make_shared<ov::op::v1::Subtract>(relu1, reduce_max);
        const auto exp = std::make_shared<ov::op::v0::Exp>(subtract);

        const auto reduce_sum = std::make_shared<ov::snippets::op::ReduceSum>(exp, 3);
        ov::snippets::op::ReduceBase::compute_and_set_reduce_subtensors(reduce_sum);
        const auto power = std::make_shared<ov::snippets::op::PowerStatic>(reduce_sum, -1.f);
        const auto multiply = std::make_shared<ov::op::v1::Multiply>(exp, power);

        const auto matmul1 = create_matmul(multiply, parameter2);
        const auto relu2 = std::make_shared<ov::op::v0::Relu>(matmul1);

        const auto body = std::make_shared<ov::Model>(std::make_shared<ov::snippets::op::Result>(relu2),
                                                      ov::ParameterVector{parameter0, parameter1, parameter2});

        this->MarkOp(load_reshape, {subtensor_scalar}, {subtensor_scalar});
        this->MarkOp(store, {subtensor_scalar}, {subtensor_scalar});
        this->MarkOp(power, {subtensor_power}, {subtensor_power});
        this->MarkOp(matmul0, {subtensor_full, subtensor_full}, {subtensor_full});
        this->MarkOp(matmul1, {subtensor_full, subtensor_full}, {subtensor_full});

        ov::snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(load_reshape->input(0))->set_layout(order);

        return body;
    }

    virtual std::shared_ptr<ov::Node> create_matmul(const std::shared_ptr<ov::Node>& lhs,
                                                    const std::shared_ptr<ov::Node>& rhs) const = 0;
};

namespace BufferAllocationCPUTest_Instances {

inline const std::vector<ov::PartialShape> static_shapes = {
    {1, 12, 1024, 1024},
    {1, 128, 12, 1024},
    {1, 12, 128, 256},
};

inline const std::vector<ov::PartialShape> dynamic_shapes = {
    {-1, -1, -1, -1},
    {-1, -1, -1, -1},
    {-1, -1, -1, -1},
};

}  // namespace BufferAllocationCPUTest_Instances

}  // namespace snippets
}  // namespace test
}  // namespace ov
