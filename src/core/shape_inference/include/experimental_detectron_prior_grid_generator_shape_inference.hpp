// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/op/experimental_detectron_prior_grid_generator.hpp>

namespace ov {
namespace op {
namespace v6 {

template <class T>
void shape_infer(const ExperimentalDetectronPriorGridGenerator* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 3 && output_shapes.size() == 1);
    const auto& priors_shape = input_shapes[0];
    const auto& featmap_shape = input_shapes[1];
    const auto& im_data_shape = input_shapes[2];

    auto& output_shape = output_shapes[0];
    size_t output_size = op->m_attrs.flatten ? 2 : 4;

    output_shape.resize(output_size);
    output_shape[output_size - 1] = 4;

    bool prior_rank_static = priors_shape.rank().is_static();
    bool featmap_rank_static = featmap_shape.rank().is_static();
    bool im_data_rank_static = im_data_shape.rank().is_static();

    if (prior_rank_static) {
        NODE_VALIDATION_CHECK(op, priors_shape.size() == 2, "Priors rank must be equal to 2.");
        NODE_VALIDATION_CHECK(op,
                              priors_shape[1].compatible(4),
                              "The last dimension of the 'priors' input must be equal to 4. Got: ",
                              priors_shape[1]);
    }

    if (featmap_rank_static) {
        NODE_VALIDATION_CHECK(op, featmap_shape.size() == 4, "Feature_map rank must be equal to 4.");
    }

    if (im_data_rank_static) {
        NODE_VALIDATION_CHECK(op, im_data_shape.size() == 4, "Im_data rank must be equal to 4.");
    }

    if (featmap_rank_static && im_data_rank_static) {
        const auto& num_batches_featmap = featmap_shape[0];
        const auto& num_batches_im_data = im_data_shape[0];

        NODE_VALIDATION_CHECK(op,
                              num_batches_featmap.compatible(num_batches_im_data),
                              "The first dimension of both 'feature_map' and 'im_data' must match. "
                              "Feature_map: ",
                              num_batches_featmap,
                              "; Im_data: ",
                              num_batches_im_data);
    }

    if (op->m_attrs.flatten) {
        if (prior_rank_static && featmap_rank_static) {
            output_shape[0] = featmap_shape[2] * featmap_shape[3] * priors_shape[0];
        }
    } else {
        if (featmap_rank_static) {
            output_shape[0] = featmap_shape[2];
            output_shape[1] = featmap_shape[3];
        }
        if (prior_rank_static) {
            output_shape[2] = priors_shape[0];
        }
    }
}

}  // namespace v6
}  // namespace op
}  // namespace ov
