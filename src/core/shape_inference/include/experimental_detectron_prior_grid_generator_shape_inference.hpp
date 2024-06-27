// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/experimental_detectron_prior_grid_generator.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v6 {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const ExperimentalDetectronPriorGridGenerator* op,
                                 const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 3);
    const auto& priors_shape = input_shapes[0];
    const auto& featmap_shape = input_shapes[1];
    const auto& im_data_shape = input_shapes[2];

    const auto is_flatten = op->get_attrs().flatten;
    const size_t output_size = is_flatten ? 2 : 4;

    auto output_shapes = std::vector<TRShape>(1);
    auto& output_shape = output_shapes[0];
    output_shape.resize(output_size);
    output_shape[output_size - 1] = 4;

    const auto prior_rank_static = priors_shape.rank().is_static();
    const auto featmap_rank_static = featmap_shape.rank().is_static();
    const auto im_data_rank_static = im_data_shape.rank().is_static();

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

        if (featmap_rank_static) {
            NODE_VALIDATION_CHECK(op,
                                  featmap_shape[0].compatible(im_data_shape[0]),
                                  "The first dimension of both 'feature_map' and 'im_data' must match. "
                                  "Feature_map: ",
                                  featmap_shape[0],
                                  "; Im_data: ",
                                  im_data_shape[0]);
        }
    }

    if (is_flatten) {
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

    return output_shapes;
}
}  // namespace v6
}  // namespace op
}  // namespace ov
