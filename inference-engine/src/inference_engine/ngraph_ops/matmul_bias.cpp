// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "matmul_bias.hpp"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"

#include <details/ie_exception.hpp>


using namespace std;
using namespace ngraph;

shared_ptr<Node> op::MatmulBias::copy_with_new_args(const NodeVector& new_args) const {
    if (new_args.size() != 2 && new_args.size() != 3) {
        THROW_IE_EXCEPTION <<"Incorrect number of new arguments";
    }

    return make_shared<MatmulBias>(new_args.at(0),
                                   new_args.at(1),
                                   new_args.size() == 3 ? new_args.at(2) : nullptr,
                                   m_shape_w,
                                   m_shape_x,
                                   m_transpose_w,
                                   m_transpose_x,
                                   m_broadcast_axes);
}

op::MatmulBias::MatmulBias(shared_ptr<Node> W,
                           shared_ptr<Node> x,
                           shared_ptr<Node> b,
                           Shape shape_w,
                           Shape shape_x,
                           bool transpose_w,
                           bool transpose_x,
                           AxisSet axes)
    : Op("MatmulBias",
         check_single_output_args(b == nullptr ? vector<shared_ptr<Node>>{W, x}
                                               : vector<shared_ptr<Node>>{W, x, b}))
    , m_shape_w(shape_w)
    , m_shape_x(shape_x)
    , m_transpose_w(transpose_w)
    , m_transpose_x(transpose_x)
    , m_broadcast_axes(axes) {
    constructor_validate_and_infer_types();
}

void op::MatmulBias::validate_and_infer_types() {
    auto et = get_input_element_type(0);
    bool have_b = get_input_size() > 2;
    if (m_broadcast_axes.size() == 0 && have_b) {
        THROW_IE_EXCEPTION << "Bias but no broadcast axes";
    }

    if (!have_b && m_broadcast_axes.size() != 0) {
        THROW_IE_EXCEPTION << "Broadcast axes but no bias";
    }

    if (m_broadcast_axes.size() > 2) {
        THROW_IE_EXCEPTION << "Broadcasting to > 2D tensor";
    }

    if (m_shape_w.size() != 2) {
        THROW_IE_EXCEPTION << "W.shape.rank != 2 while creating MatmulBias";
    }

    if (m_shape_x.size() != 2) {
        THROW_IE_EXCEPTION << "x.shape.rank != 2 while creating MatmulBias";
    }

    size_t dot_dimension_w = (m_transpose_w) ? 0 : 1;
    size_t dot_dimension_x = (m_transpose_x) ? 1 : 0;

    if (m_shape_w.at(dot_dimension_w) != m_shape_x.at(dot_dimension_x)) {
        THROW_IE_EXCEPTION << "product dimensions are not equal while creating MatmulBias";
    }

    Shape dot_shape{m_shape_w.at(1 - dot_dimension_w), m_shape_x.at(1 - dot_dimension_x)};

    set_output_type(0, et, dot_shape);
}
