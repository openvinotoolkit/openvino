// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "helper_ops/internal_operation.hpp"
#include "openvino/frontend/decoder.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

class SparseFillEmptyRows : public ov::frontend::tensorflow::InternalOperation {
public:
    OPENVINO_OP("SparseFillEmptyRows", "ov::frontend::tensorflow::util", ov::frontend::tensorflow::InternalOperation);

    SparseFillEmptyRows(const Output<Node>& indices,
                        const Output<Node>& values,
                        const Output<Node>& dense_shape,
                        const Output<Node>& default_value,
                        const std::shared_ptr<DecoderBase>& decoder = nullptr)
        : ov::frontend::tensorflow::InternalOperation(decoder,
                                                      OutputVector{indices, values, dense_shape, default_value},
                                                      4,
                                                      "SparseFillEmptyRows") {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        // SparseFillEmptyRows fills empty rows in the input 2-D SparseTensor with a default value.
        // Inputs:
        // 0) indices: 2D. the indices of the sparse tensor
        // 1) values: 1D. the values of the sparse tensor
        // 2) dense_shape: 1D. the shape of the sparse tensor
        // 3) default_value: 0D. default value to insert into location [row, 0, ..., 0] for rows missing from the input
        //    sparse tensor. output indices: 2-D. the indices of the filled sparse tensor
        // Outputs:
        // 0) output_indices
        // 1) output_values, 1D the values of the filled sparse tensor
        // 2) empty_row_indicator, 1D whether the dense row was missing in the input sparse tensor
        // 3) reverse_index_map, 1D a map from the input indices to the output indices
        ov::PartialShape output_indices_shape({ov::Dimension::dynamic(), 2});
        ov::PartialShape output_values_shape({ov::Dimension::dynamic()});
        ov::PartialShape empty_row_indicator_shape({ov::Dimension::dynamic()});
        ov::PartialShape reverse_index_map_shape({ov::Dimension::dynamic()});

        set_output_type(0, get_input_element_type(0), output_indices_shape);
        set_output_type(1, get_input_element_type(1), output_values_shape);
        set_output_type(2, ov::element::boolean, empty_row_indicator_shape);
        set_output_type(3, get_input_element_type(0), reverse_index_map_shape);
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        FRONT_END_OP_CONVERSION_CHECK(inputs.size() == 4,
                                      "[TensorFlow Frontend] internal error: SparseFillEmptyRows expects 4 inputs");
        auto sparse_fill_empty_rows_node =
            std::make_shared<SparseFillEmptyRows>(inputs[0], inputs[1], inputs[2], inputs[3], m_decoder);
        sparse_fill_empty_rows_node->set_attrs(get_attrs());
        return sparse_fill_empty_rows_node;
    }
};
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
