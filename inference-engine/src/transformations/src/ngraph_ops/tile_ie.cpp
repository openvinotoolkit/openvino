// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/tile_ie.hpp"

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

std::shared_ptr<Node> op::TileIE::copy_with_new_args(const NodeVector& new_args) const {
    if (new_args.size() != 1) {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<TileIE>(new_args.at(0), axis, tiles);
}

void op::TileIE::validate_and_infer_types() {
    auto input_shape = get_input_partial_shape(0).to_shape();

    ngraph::Shape output_shape(input_shape);
    output_shape[axis] *= tiles;

    set_output_type(0, get_input_element_type(0), PartialShape(output_shape));
}
