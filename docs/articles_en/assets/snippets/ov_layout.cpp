// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/core/layout.hpp"
#include "openvino/core/model.hpp"

int main() {
    ov::Layout layout;
//! [ov:layout:simple]
layout = ov::Layout("NHWC");
//! [ov:layout:simple]

//! [ov:layout:complex]
// Each dimension has name separated by comma, layout is wrapped with square brackets
layout = ov::Layout("[time,temperature,humidity]");
//! [ov:layout:complex]

//! [ov:layout:partially_defined]
// First dimension is batch, 4th is 'channels'. Others are not important for us
layout = ov::Layout("N??C");
// Or the same using advanced syntax
layout = ov::Layout("[n,?,?,c]");
//! [ov:layout:partially_defined]

//! [ov:layout:dynamic]
// First dimension is 'batch' others are whatever
layout = ov::Layout("N...");

// Second dimension is 'channels' others are whatever
layout = ov::Layout("?C...");

// Last dimension is 'channels' others are whatever
layout = ov::Layout("...C");
//! [ov:layout:dynamic]

//! [ov:layout:predefined]
// returns 0 for batch
ov::layout::batch_idx("NCDHW");

// returns 1 for channels
ov::layout::channels_idx("NCDHW");

// returns 2 for depth
ov::layout::depth_idx("NCDHW");

// returns -2 for height
ov::layout::height_idx("...HW");

// returns -1 for width
ov::layout::width_idx("...HW");
//! [ov:layout:predefined]

//! [ov:layout:dump]
layout = ov::Layout("NCHW");
std::cout << layout.to_string(); // prints [N,C,H,W]
//! [ov:layout:dump]

std::shared_ptr<ov::Model> model = std::make_shared<ov::Model>(ov::OutputVector{}, ov::ParameterVector{});
//! [ov:layout:get_from_model]
// Get layout for model input
layout = ov::layout::get_layout(model->input("input_tensor_name"));
// Get layout for model with single output
layout = ov::layout::get_layout(model->output());
//! [ov:layout:get_from_model]

    return 0;
}
