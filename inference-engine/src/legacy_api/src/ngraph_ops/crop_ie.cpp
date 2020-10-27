// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/ngraph_ops/crop_ie.hpp"

#include <algorithm>
#include <memory>
#include <vector>

#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::CropIE::type_info;

op::CropIE::CropIE(const Output<Node>& data, std::vector<int64_t> axes, std::vector<int64_t> dim,
                   std::vector<int64_t> offset)
    : Op({data}), axes(axes), dim(dim), offset(offset) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::CropIE::clone_with_new_inputs(const OutputVector& new_args) const {
    if (new_args.size() != 1) {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<CropIE>(new_args.at(0), axes, dim, offset);
}

void op::CropIE::validate_and_infer_types() {
    auto input_shape = get_input_partial_shape(0).to_shape();
    NODE_VALIDATION_CHECK(this, axes.size() == dim.size(), "axes and dim needs to have same number of values");

    NODE_VALIDATION_CHECK(this, axes.size() == offset.size(), "axes and offset needs to have same number of values");

    ngraph::Shape output_shape(input_shape);
    for (int i = 0; i < axes.size(); ++i) {
        NODE_VALIDATION_CHECK(this, axes[i] >= 0 && axes[i] < static_cast<int64_t>(output_shape.size()),
                              "axes should be positive and less than number of input dims");
        output_shape[axes[i]] = dim[i];
    }

    set_output_type(0, get_input_element_type(0), PartialShape(output_shape));
}
