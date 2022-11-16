// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/gather_nd.hpp"

#include "itt.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

// ------------------------------ V5 ------------------------------

BWDCMP_RTTI_DEFINITION(op::v5::GatherND);

op::v5::GatherND::GatherND(const Output<Node>& data, const Output<Node>& indices, const size_t batch_dims)
    : GatherNDBase(data, indices, batch_dims) {
    constructor_validate_and_infer_types();
}

void op::v5::GatherND::validate_and_infer_types() {
    OV_OP_SCOPE(v5_GatherND_validate_and_infer_types);
    validate_inputs_and_infer_shape();

    // If we have m_batch_dims > 1 we need to fuse batch dimensions of output
    if (m_batch_dims > 1) {
        const auto& output_pshape = get_output_partial_shape(0);
        const auto& data_type = get_input_element_type(0);

        if (output_pshape.rank().is_static()) {
            const auto& out_size = output_pshape.size();
            std::vector<Dimension> output_shape(out_size - m_batch_dims + 1);
            output_shape[0] = 1;
            for (size_t dim = 0; dim < m_batch_dims; dim++) {
                if (output_pshape[dim].is_static()) {
                    output_shape[0] *= output_pshape[dim].get_length();
                } else {
                    output_shape[0] = Dimension::dynamic();
                    break;
                }
            }
            size_t ind = 1;
            for (size_t dim = m_batch_dims; dim < out_size; dim++) {
                if (output_pshape[dim].is_static()) {
                    output_shape[ind] = output_pshape[dim].get_length();
                } else {
                    output_shape[ind] = Dimension::dynamic();
                }
                ind++;
            }

            set_output_type(0, data_type, ov::PartialShape(output_shape));
        }
    }
}

shared_ptr<Node> op::v5::GatherND::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v5_GatherND_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v5::GatherND>(new_args.at(0), new_args.at(1), m_batch_dims);
}

// ------------------------------ V8 ------------------------------
BWDCMP_RTTI_DEFINITION(op::v8::GatherND);

op::v8::GatherND::GatherND(const Output<Node>& data, const Output<Node>& indices, const size_t batch_dims)
    : GatherNDBase(data, indices, batch_dims) {
    constructor_validate_and_infer_types();
}

void op::v8::GatherND::validate_and_infer_types() {
    OV_OP_SCOPE(v8_GatherND_validate_and_infer_types);
    validate_inputs_and_infer_shape();
}

shared_ptr<Node> op::v8::GatherND::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v8_GatherND_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v8::GatherND>(new_args.at(0), new_args.at(1), m_batch_dims);
}
