// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "type_prop.hpp"

#include <functional>
#include <vector>

#include "dimension_tracker.hpp"
#include "openvino/core/dimension.hpp"

std::vector<size_t> get_shape_labels(const ov::PartialShape& p_shape) {
    std::vector<size_t> labels;
    transform(p_shape.cbegin(), p_shape.cend(), back_inserter(labels), [](const ov::Dimension& dim) {
        return ov::DimensionTracker::get_label(dim);
    });
    return labels;
}

void set_shape_labels(ov::PartialShape& p_shape, const std::vector<size_t>& labels) {
    ASSERT_EQ(labels.size(), p_shape.size());
    auto label_it = labels.begin();

    std::for_each(p_shape.begin(), p_shape.end(), [&label_it](ov::Dimension& dim) {
        if (*label_it > 0) {
            ov::DimensionTracker::set_label(dim, *label_it);
        }
        ++label_it;
    });
}
