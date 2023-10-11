// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/type_prop.hpp"

#include "openvino/core/dimension.hpp"
#include "sequnce_generator.hpp"

ov::TensorLabel get_shape_labels(const ov::PartialShape& p_shape) {
    ov::TensorLabel labels;
    transform(p_shape.cbegin(), p_shape.cend(), back_inserter(labels), [](const ov::Dimension& dim) {
        return ov::DimensionTracker::get_label(dim);
    });
    return labels;
}

void set_shape_labels(ov::PartialShape& p_shape, const ov::label_t first_label) {
    ov::TensorLabel labels;
    std::generate_n(std::back_inserter(labels), p_shape.size(), ov::SeqGen<ov::label_t>(first_label));
    set_shape_labels(p_shape, labels);
}

void set_shape_labels(ov::PartialShape& p_shape, const ov::TensorLabel& labels) {
    ASSERT_EQ(labels.size(), p_shape.size());
    auto label_it = labels.begin();

    std::for_each(p_shape.begin(), p_shape.end(), [&label_it](ov::Dimension& dim) {
        if (*label_it > 0) {
            ov::DimensionTracker::set_label(dim, *label_it);
        }
        ++label_it;
    });
}
