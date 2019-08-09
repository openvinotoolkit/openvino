//
// Created by gkazanta on 6/11/19.
//

//*****************************************************************************
// Copyright 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <memory>
#include <vector>
#include <algorithm>

#include "crop_ie.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

op::CropIE::CropIE(const std::shared_ptr<ngraph::Node> &data, std::vector<int64_t> axes, std::vector<int64_t> dim, std::vector<int64_t> offset)
        : Op("CropIE", check_single_output_args({data})), axes(axes), dim(dim), offset(offset) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::CropIE::copy_with_new_args(const NodeVector& new_args) const {
    if (new_args.size() != 1) {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<CropIE>(new_args.at(0), axes, dim, offset);
}

void op::CropIE::validate_and_infer_types() {
    auto input_shape = get_input_partial_shape(0).to_shape();
    NODE_VALIDATION_CHECK(this,
                          axes.size() == dim.size(),
                          "axes and dim needs to have same number of values");

    NODE_VALIDATION_CHECK(this,
                          axes.size() == offset.size(),
                          "axes and offset needs to have same number of values");

    ngraph::Shape output_shape(input_shape);
    for (int i = 0; i < axes.size(); ++i) {
        NODE_VALIDATION_CHECK(this, axes[i] >= 0 && axes[i] < output_shape.size(),
                              "axes should be positive and less than number of input dims");
        output_shape[axes[i]] = dim[i];
    }

    set_output_type(0, get_input_element_type(0), PartialShape(output_shape));
}
