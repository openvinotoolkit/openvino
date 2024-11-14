// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "snippets/shape_inference/shape_inference.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {
namespace snippets {
namespace op {

/**
 * @interface Buffer
 * @brief This is a class for memory storage.
 *        The buffers can have source (called as "IntermediateMemory") and can be without source (called as "NewMemory").
 *        First one contains memory which was stored by source -> these buffers propagate output shape and element type from parents to output.
 *        Second one has passed `element_type` and `shape` by user - these attributes describe independent empty memory.
 *        The both behaviors are implemented via the corresponding classes which are derived from the class "Buffer::BaseImpl".
 *        It allows user to work with only the class "op::Buffer" - all needed logic is implemented in the field `m_impl`.
 * @ingroup snippets
 */
class Buffer : public ov::op::Op {
public:
    OPENVINO_OP("Buffer", "SnippetsOpset");
    Buffer() = default;
    Buffer(const ov::Output<ov::Node>& arg);
    Buffer(const OutputVector& arguments);
    Buffer(const ov::Shape& shape, ov::element::Type element_type = ov::element::u8);

    bool visit_attributes(AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    size_t get_allocation_size() const { return m_impl->get_allocation_size(); }

    class ShapeInfer : public IShapeInferSnippets {
        std::shared_ptr<IShapeInferSnippets> m_impl_shape_infer {nullptr};
    public:
        explicit ShapeInfer(const std::shared_ptr<ov::Node>& n);
        Result infer(const std::vector<VectorDimsRef>& input_shapes) override;
    };

private:
    // Base class for implementations of Buffer
    class BaseImpl {
    public:
        BaseImpl() = default;
        virtual ~BaseImpl() = default;
        virtual size_t get_allocation_size() const = 0;
        virtual std::shared_ptr<BaseImpl> clone() const = 0;
        virtual void validate_and_infer_types(Buffer* buffer) const = 0;
        virtual bool visit_attributes(AttributeVisitor& visitor) = 0;
        virtual std::shared_ptr<IShapeInferSnippets> get_shape_infer() const = 0;
    };

    // IntermediateMemoryImpl represents intermediate memory.
    // The buffers with this implementation must have source (parents)
    class IntermediateMemoryImpl : public BaseImpl {
    public:
        IntermediateMemoryImpl() = default;

        size_t get_allocation_size() const override { return utils::get_dynamic_value<size_t>(); }
        std::shared_ptr<BaseImpl> clone() const override;
        void validate_and_infer_types(Buffer* buffer) const override;
        bool visit_attributes(AttributeVisitor& visitor) override { return true; }
        std::shared_ptr<IShapeInferSnippets> get_shape_infer() const override { return std::make_shared<ShapeInfer>(); }
    private:
        class ShapeInfer : public IShapeInferSnippets {
        public:
            Result infer(const std::vector<VectorDimsRef>& input_shapes) override;
        };
    };

    // NewMemoryImpl represents a new empty memory for allocation with specified shape and element type.
    // The buffers with this implementation mustn't have source (parents)
    class NewMemoryImpl : public BaseImpl {
    public:
        NewMemoryImpl(const ov::Shape& shape, ov::element::Type element_type);

        size_t get_allocation_size() const override;
        std::shared_ptr<BaseImpl> clone() const override;
        void validate_and_infer_types(Buffer* buffer) const override;
        bool visit_attributes(AttributeVisitor& visitor) override;
        std::shared_ptr<IShapeInferSnippets> get_shape_infer() const override { return std::make_shared<ShapeInfer>(m_shape); }
    private:
        class ShapeInfer : public IShapeInferSnippets {
            ov::Shape m_shape;
        public:
            explicit ShapeInfer(ov::Shape shape);
            Result infer(const std::vector<VectorDimsRef>& input_shapes) override;
        };

        ov::Shape m_shape;
        ov::element::Type m_element_type = ov::element::u8;  // u8 - default 1 byte
    };

    // This constructor is used only in clone_with_new_inputs
    Buffer(const OutputVector& arguments, std::shared_ptr<BaseImpl> impl);

    const std::shared_ptr<BaseImpl> m_impl {nullptr};
};


} // namespace op
} // namespace snippets
} // namespace ov
