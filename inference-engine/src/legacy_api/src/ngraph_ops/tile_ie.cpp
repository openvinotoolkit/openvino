// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/ngraph_ops/tile_ie.hpp"

#include <algorithm>
#include <memory>

#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::TileIE::type_info;

op::TileIE::TileIE(const Output<ngraph::Node>& data1, const int64_t axis, const int64_t tiles)
    : Op({data1}), axis(axis), tiles(tiles) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::TileIE::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<TileIE>(new_args.at(0), axis, tiles);
}

void op::TileIE::validate_and_infer_types() {
    const auto & input_pshape = get_input_partial_shape(0);
    auto output_pshape = PartialShape::dynamic();
    if (input_pshape.rank().is_static()) {
        const auto & rank = input_pshape.rank().get_length();
        NODE_VALIDATION_CHECK(this,
                              axis >= 0 && axis < rank,
                              "Axis: ", axis, " must be >= 0 and less than ", rank, "(input rank)");
        output_pshape = input_pshape;
        if (output_pshape[axis].is_static()) {
            output_pshape[axis] *= tiles;
        }
    }

    set_output_type(0, get_input_element_type(0), output_pshape);
}

bool op::TileIE::visit_attributes(AttributeVisitor& visitor){
    visitor.on_attribute("axis", axis);
    visitor.on_attribute("tiles", tiles);
    return true;
}
