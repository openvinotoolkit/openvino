/* Copyright (C) 2018-2021 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 *
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * We modified "values_from_const_node" function from
 * tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc file
 * to integrate it with our infrastructure. The purpose and basic
 * functionality remains the same.
==============================================================================*/

#pragma once

#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph_conversions.hpp"
#include "openvino/opsets/opset8.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

void set_out_name(const std::string& out_name, const Output<Node>& output);

void set_node_name(const std::string& node_name, const std::shared_ptr<Node>& node);

static bool vec_str_cmp(const std::vector<std::string>& a, const std::vector<std::string>& b) {
    return a == b;
}

template <typename T>
void make_padding(const std::string& tf_padding_type,
                  const ov::Shape& ng_image_shape,
                  const ov::Shape& ng_kernel_shape,
                  const ov::Strides& ng_strides,
                  const ov::Shape& ng_dilations,
                  T& ng_padding_below,
                  T& ng_padding_above) {
    if (tf_padding_type == "SAME") {
        ov::Shape img_shape = {0, 0};
        img_shape.insert(img_shape.end(), ng_image_shape.begin(), ng_image_shape.end());
        ov::infer_auto_padding(img_shape,
                               ng_kernel_shape,
                               ng_strides,
                               ng_dilations,
                               ov::op::PadType::SAME_UPPER,
                               ng_padding_above,
                               ng_padding_below);
    } else if (tf_padding_type == "VALID") {
        ng_padding_below.assign(ng_image_shape.size(), 0);
        ng_padding_above.assign(ng_image_shape.size(), 0);
    }
}
}  // namespace tf
}  // namespace frontend
}  // namespace ov
