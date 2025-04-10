// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "helper_ops/internal_operation.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

class Iterator : public InternalOperation {
public:
    OPENVINO_OP("Iterator", "ov::frontend::tensorflow", InternalOperation);

    Iterator(const std::string& shared_name,
             const std::string& container,
             const std::vector<ov::element::Type>& output_types,
             const std::vector<ov::PartialShape>& output_shapes,
             const std::shared_ptr<DecoderBase>& decoder = nullptr)
        : InternalOperation(decoder, OutputVector{}, 1, "Iterator"),
          m_shared_name(shared_name),
          m_container(container),
          m_output_types(output_types),
          m_output_shapes(output_shapes) {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, ov::element::dynamic, ov::PartialShape::dynamic());
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        auto iterator_node =
            std::make_shared<Iterator>(m_shared_name, m_container, m_output_types, m_output_shapes, m_decoder);
        iterator_node->set_attrs(get_attrs());
        return iterator_node;
    }

private:
    const std::string m_shared_name;
    const std::string m_container;
    const std::vector<ov::element::Type> m_output_types;
    const std::vector<ov::PartialShape> m_output_shapes;
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
