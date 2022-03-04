// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/gather_nd_base.hpp"

#include <ngraph/validation_util.hpp>

#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/squeeze.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/shape.hpp"

using namespace std;

BWDCMP_RTTI_DEFINITION(ov::op::util::GatherNDBase);

ov::op::util::GatherNDBase::GatherNDBase(const Output<Node>& data, const Output<Node>& indices, const size_t batch_dims)
    : Op({data, indices}),
      m_batch_dims(batch_dims) {
    constructor_validate_and_infer_types();
}

void ov::op::util::GatherNDBase::validate_inputs_and_infer_shape() {
    // check types of input tensors
    const auto& data_type = get_input_element_type(0);
    const auto& indices_type = get_input_element_type(1);

    NODE_VALIDATION_CHECK(this,
                          indices_type.is_integral_number(),
                          "The indices type is expected to be an integer type. Got: ",
                          indices_type);

    // check ranks of input tensors
    const auto& data_pshape = get_input_partial_shape(0);
    const auto& indices_pshape = get_input_partial_shape(1);

    if (data_pshape.rank().is_static()) {
        NODE_VALIDATION_CHECK(this, data_pshape.rank().get_length() > 0, "Data rank must be at least 1.");

        NODE_VALIDATION_CHECK(this,
                              data_pshape.rank().get_length() > static_cast<int64_t>(m_batch_dims),
                              "Number of batch dimensions must not exceed a rank of data.");
    }

    if (indices_pshape.rank().is_static()) {
        NODE_VALIDATION_CHECK(this, indices_pshape.rank().get_length() > 0, "Indices rank must be at least 1.");

        NODE_VALIDATION_CHECK(this,
                              indices_pshape.rank().get_length() > static_cast<int64_t>(m_batch_dims),
                              "Number of batch dimensions must not exceed a rank of indices.");
    }

    if (data_pshape.rank().is_static() && indices_pshape.rank().is_static()) {
        // check that batch dimensions of data and indices are the same
        for (size_t batch_dim = 0; batch_dim < m_batch_dims; batch_dim++) {
            if (data_pshape[batch_dim].is_static() && indices_pshape[batch_dim].is_static()) {
                NODE_VALIDATION_CHECK(this,
                                      data_pshape[batch_dim].get_length() == indices_pshape[batch_dim].get_length(),
                                      "Batch dimensions of data and indices must be the same.");
            }
        }

        if (indices_pshape[indices_pshape.rank().get_length() - 1].is_static()) {
            NODE_VALIDATION_CHECK(
                this,
                static_cast<int64_t>(indices_pshape[indices_pshape.rank().get_length() - 1].get_length() +
                                     m_batch_dims) <= data_pshape.rank().get_length(),
                "Length of a tuple with indices must not exceed a rank of data tensor "
                "excluding "
                "batch dimensions.");
        }
    }

    // set output shape
    set_output_size(1);
    if (data_pshape.rank().is_static() && indices_pshape.rank().is_static() &&
        indices_pshape[indices_pshape.rank().get_length() - 1].is_static()) {
        auto indices_tuple_length = indices_pshape[indices_pshape.rank().get_length() - 1].get_length();
        int64_t slice_length = data_pshape.rank().get_length() - indices_tuple_length - m_batch_dims;
        int64_t output_indices_length = indices_pshape.rank().get_length() - m_batch_dims - 1;
        auto output_rank = output_indices_length + slice_length;
        size_t delta_output_rank = 0;
        delta_output_rank = m_batch_dims;
        std::vector<Dimension> output_shape(output_rank + delta_output_rank);
        for (size_t dim = 0; dim < m_batch_dims; dim++) {
            output_shape[dim] = 1;
            if (data_pshape[dim].is_static()) {
                output_shape[dim] = data_pshape[dim].get_length();
            } else if (indices_pshape[dim].is_static()) {
                output_shape[dim] = indices_pshape[dim].get_length();
            } else {
                output_shape[dim] = Dimension::dynamic();
                break;
            }
        }
        for (int64_t dim = 0; dim < output_indices_length; dim++) {
            output_shape[dim + delta_output_rank] = indices_pshape[dim + m_batch_dims];
        }
        for (int64_t dim = 0; dim < slice_length; dim++) {
            output_shape[output_indices_length + dim + delta_output_rank] =
                data_pshape[m_batch_dims + indices_tuple_length + dim];
        }
        set_output_type(0, data_type, ov::PartialShape(output_shape));
    } else {
        set_output_type(0, data_type, ov::PartialShape::dynamic());
    }
}

bool ov::op::util::GatherNDBase::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("batch_dims", m_batch_dims);
    return true;
}
