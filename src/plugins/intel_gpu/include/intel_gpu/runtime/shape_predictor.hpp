// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "layout.hpp"

#include <deque>

namespace cldnn {

class engine;

struct ShapePredictor {
public:
    struct Settings {
        // Iterations mode preallocation
        size_t next_iters_preallocation_count = 10;
        size_t max_per_iter_size = 16 * 1024;
        size_t max_per_dim_diff = 2;

        // Percentage mode preallocation
        float buffers_preallocation_ratio = 1.1f;
    };

    using Ptr = std::shared_ptr<ShapePredictor>;
    ShapePredictor(const engine* engine, const Settings& settings)
        : _engine(engine)
        , _settings(settings) {
        static_assert(_max_deque_size >= 2, "[GPU] Deque is supposed to contain at least 2 elements for prediction");
    }

    /// \brief Predicts the next possible shapes sizes based on history collected by previous
    ///        predict_preallocation_shape() calls.
    ///        This function works in two modes: by default it tries to predict shape for the next
    ///        `_next_iters_preallocation_count` iterations, in case if per-iteration buffer size is less than
    ///        `_max_per_iter_size` and difference between shapes is less than `_max_per_dim_diff`; the second
    ///        operation mode is percentage preallocation - this mode can be configured with
    ///        ov::intel_gpu::buffers_preallocation_ratio property, it increases buffer size by
    ///        `_buffers_preallocation_ratio` value unconditionally.
    /// \param id Primitive id.
    /// \param layout Primitive's layout on current iteration.
    /// \param can_reuse_buffer Specifies if current memory buffer is enough to store data.
    /// \param out_idx output index of multiple outputs
    /// \param custom_next_iters_prealloc_couunt If it is specified, enforce prealloc size as the specified value
    /// \param custom_prealloc_dim If both custom_next_iters_prealloc_count and custom_prealloc_dim are specified,
    ///         increase custom_prealloc_dim with custom_next_iters_prealloc_count without checking shape history (e.g.,
    ///         used for first inference of kv cache)
    /// \return The result of shape size prediction as std::pair<bool, ov::Shape>, where
    ///         the first element says if shape is successfully predicted and can be preallocated, and the second
    ///         element is ov::Shape itself.
    std::pair<bool, ov::Shape> predict_preallocation_shape(const std::string& id,
                                                           const cldnn::layout& layout,
                                                           bool can_reuse_buffer,
                                                           const size_t out_idx = 0,
                                                           int32_t custom_next_iters_prealloc_count = -1,
                                                           int32_t custom_prealloc_dim = -1);

    bool can_preallocate(size_t desired_buffer_size);

    void reset() {
        _shapes_info.clear();
    }

private:
    void add_shape(const std::string& id, const ov::Shape& shape);

    static constexpr size_t _max_deque_size = 3;
    std::map<std::string, std::deque<ov::Shape>> _shapes_info;
    const engine* _engine;

    const Settings _settings;
};

}  // namespace cldnn

namespace ov::intel_gpu {
using ShapePredictor = cldnn::ShapePredictor;
}  // namespace ov::intel_gpu
