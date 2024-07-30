// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "internal_operation.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

// Internal operation for TensorList that represents a initial state of tensor list container
class TensorList : public InternalOperation {
public:
    OPENVINO_OP("TensorList", "ov::frontend::tensorflow", InternalOperation);

    TensorList(const ov::Output<ov::Node>& num_elements,
               const ov::Rank& element_rank,
               const element::Type& element_dtype,
               const std::shared_ptr<DecoderBase>& decoder = std::make_shared<DecoderFake>())
        : InternalOperation(decoder, OutputVector{num_elements}, 1, "TensorList"),
          m_num_elements(num_elements),
          m_element_rank(element_rank),
          m_element_dtype(element_dtype) {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        if (m_element_rank.is_static()) {
            auto element_rank = m_element_rank.get_length();
            auto output_shape = ov::PartialShape::dynamic(element_rank + 1);
            set_output_type(0, m_element_dtype, output_shape);
        }

        set_output_type(0, m_element_dtype, ov::PartialShape::dynamic());
    }

    ov::element::Type get_element_type() const {
        return m_element_dtype;
    }

    void set_element_type(const ov::element::Type& element_dtype) {
        m_element_dtype = element_dtype;
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        FRONT_END_OP_CONVERSION_CHECK(inputs.size() == 1,
                                      "[TensorFlow Frontend] internal error: TensorList expects no inputs");
        auto tensor_list_node = std::make_shared<TensorList>(inputs[0], m_element_rank, m_element_dtype, m_decoder);
        tensor_list_node->set_attrs(get_attrs());
        return tensor_list_node;
    }

    ov::Rank get_element_rank() const {
        return m_element_rank;
    }

    void set_element_rank(const ov::Rank& element_rank) {
        m_element_rank = element_rank;
    }

    ov::Output<ov::Node> get_num_elements() const {
        return m_num_elements;
    }

private:
    ov::Output<ov::Node> m_num_elements;
    ov::Rank m_element_rank;
    ov::element::Type m_element_dtype;
};

// Internal operation for TensorListGetItem
// it gets an element (Tensor) in tensor list by index
class TensorListGetItem : public InternalOperation {
public:
    OPENVINO_OP("TensorListGetItem", "ov::frontend::tensorflow", InternalOperation);

    TensorListGetItem(const Output<Node>& input_handle,
                      const Output<Node>& index,
                      const Output<Node>& element_shape,
                      const ov::element::Type& element_type,
                      const std::shared_ptr<DecoderBase>& decoder = std::make_shared<DecoderFake>())
        : InternalOperation(decoder, OutputVector{input_handle, index, element_shape}, 1, "TensorListGetItem"),
          m_element_type(element_type) {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        // deduce an element (Tensor) shape
        ov::PartialShape comp_element_shape = ov::PartialShape::dynamic();
        if (const auto& const_element_shape =
                ov::as_type_ptr<ov::op::v0::Constant>(input_value(2).get_node_shared_ptr())) {
            auto element_shape_value = const_element_shape->get_vector<int32_t>();
            comp_element_shape = ov::PartialShape::dynamic(static_cast<int64_t>(element_shape_value.size()));
            for (size_t idx = 0; idx < element_shape_value.size(); ++idx) {
                comp_element_shape[idx] = (element_shape_value[idx] >= 0)
                                              ? static_cast<int64_t>(element_shape_value[idx])
                                              : ov::Dimension::dynamic();
            }
        } else if (input_value(0).get_partial_shape().rank().is_static()) {
            // the second try to deduce element shape if it is still of dynamic rank
            auto tensor_list_rank = input_value(0).get_partial_shape().rank().get_length();
            OPENVINO_ASSERT(
                tensor_list_rank > 0,
                "[TensorFlow Frontend] internal error or inconsistent model: tensor list rank must be greater than 0");
            // exclude tensor dimension (or batch)
            comp_element_shape = ov::PartialShape::dynamic(tensor_list_rank - 1);
            for (int64_t idx = 1; idx < tensor_list_rank; ++idx) {
                comp_element_shape[idx - 1] = input_value(0).get_partial_shape()[idx];
            }
        }

        // deduce an element (Tensor) type
        if (m_element_type.is_dynamic() && input_value(0).get_element_type().is_static()) {
            m_element_type = input_value(0).get_element_type();
        }

        set_output_type(0, m_element_type, comp_element_shape);
    }

    ov::element::Type get_element_type() const {
        return m_element_type;
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        FRONT_END_OP_CONVERSION_CHECK(inputs.size() == 3,
                                      "[TensorFlow Frontend] internal error: TensorListGetItem expects three inputs");
        auto tensor_list_get_item =
            std::make_shared<TensorListGetItem>(inputs[0], inputs[1], inputs[2], m_element_type, m_decoder);
        tensor_list_get_item->set_attrs(get_attrs());
        return tensor_list_get_item;
    }

private:
    ov::element::Type m_element_type;
};

// Internal operation for TensorListSetItem
// it inserts tensor to tensor list by index
class TensorListSetItem : public InternalOperation {
public:
    OPENVINO_OP("TensorListSetItem", "ov::frontend::tensorflow", InternalOperation);

    TensorListSetItem(const Output<Node>& input_handle,
                      const Output<Node>& index,
                      const Output<Node>& item,
                      const std::shared_ptr<DecoderBase>& decoder = std::make_shared<DecoderFake>())
        : InternalOperation(decoder, OutputVector{input_handle, index, item}, 1, "TensorListSetItem") {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        // deduce a type of elements in tensor list
        ov::element::Type element_type = ov::element::dynamic;
        if (input_value(0).get_element_type().is_static()) {
            element_type = input_value(0).get_element_type();
        } else if (input_value(2).get_element_type().is_static()) {
            element_type = input_value(2).get_element_type();
        }

        // deduce a shape of tensor list [num_tensors, <tensor shape>]
        ov::PartialShape tensor_list_shape = ov::PartialShape::dynamic();
        if (input_value(2).get_partial_shape().rank().is_static()) {
            auto element_rank = input_value(2).get_partial_shape().rank().get_length();
            tensor_list_shape = ov::PartialShape::dynamic(element_rank + 1);
            for (int64_t idx = 0; idx < element_rank; ++idx) {
                tensor_list_shape[idx + 1] = input_value(2).get_partial_shape()[idx];
            }
        }

        set_output_type(0, element_type, tensor_list_shape);
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        FRONT_END_OP_CONVERSION_CHECK(inputs.size() == 3,
                                      "[TensorFlow Frontend] internal error: TensorListSetItem expects three inputs");
        auto tensor_list_set_item = std::make_shared<TensorListSetItem>(inputs[0], inputs[1], inputs[2], m_decoder);
        tensor_list_set_item->set_attrs(get_attrs());
        return tensor_list_set_item;
    }
};

// Internal operation for TensorListPushBack
// it inserts tensor to the tail of the list
class TensorListPushBack : public InternalOperation {
public:
    OPENVINO_OP("TensorListPushBack", "ov::frontend::tensorflow", InternalOperation);

    TensorListPushBack(const Output<Node>& input_handle,
                       const Output<Node>& tensor,
                       const std::shared_ptr<DecoderBase>& decoder = std::make_shared<DecoderFake>())
        : InternalOperation(decoder, OutputVector{input_handle, tensor}, 1, "TensorListPushBack") {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        // deduce a type of elements in tensor list
        ov::element::Type element_type = ov::element::dynamic;
        if (input_value(0).get_element_type().is_static()) {
            element_type = input_value(0).get_element_type();
        } else if (input_value(1).get_element_type().is_static()) {
            element_type = input_value(1).get_element_type();
        }

        // deduce a shape of tensor list [num_tensors, <tensor shape>]
        ov::PartialShape tensor_list_shape = ov::PartialShape::dynamic();
        if (input_value(1).get_partial_shape().rank().is_static()) {
            auto element_rank = input_value(1).get_partial_shape().rank().get_length();
            tensor_list_shape = ov::PartialShape::dynamic(element_rank + 1);
            for (int64_t idx = 0; idx < element_rank; ++idx) {
                tensor_list_shape[idx + 1] = input_value(1).get_partial_shape()[idx];
            }
        }

        set_output_type(0, element_type, tensor_list_shape);
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        FRONT_END_OP_CONVERSION_CHECK(inputs.size() == 2,
                                      "[TensorFlow Frontend] internal error: TensorListPushBack expects two inputs");
        auto tensor_list_push_back = std::make_shared<TensorListPushBack>(inputs[0], inputs[1], m_decoder);
        tensor_list_push_back->set_attrs(get_attrs());
        return tensor_list_push_back;
    }
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
