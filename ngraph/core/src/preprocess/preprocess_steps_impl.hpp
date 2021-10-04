// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>

#include "openvino/core/layout.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/preprocess/color_format.hpp"
#include "openvino/core/preprocess/preprocess_steps.hpp"

namespace ov {
namespace preprocess {

inline size_t get_and_check_width_idx(const Layout& layout, const PartialShape& shape) {
    OPENVINO_ASSERT(ov::layout::has_width(layout), "Layout ", layout.to_string(), " doesn't have `width` dimension");
    OPENVINO_ASSERT(shape.rank().is_static(), "Can't get shape width index for shape with dynamic rank");
    auto idx = ov::layout::width(layout);
    if (idx < 0) {
        idx = shape.rank().get_length() + idx;
    }
    OPENVINO_ASSERT(idx >= 0 && shape.rank().get_length() > idx,
                    "Width dimension is out of bounds ",
                    std::to_string(idx));
    return idx;
}

inline size_t get_and_check_height_idx(const Layout& layout, const PartialShape& shape) {
    OPENVINO_ASSERT(ov::layout::has_height(layout), "Layout ", layout.to_string(), " doesn't have `height` dimension");
    OPENVINO_ASSERT(shape.rank().is_static(), "Can't get shape height index for shape with dynamic rank");
    auto idx = ov::layout::height(layout);
    if (idx < 0) {
        idx = shape.rank().get_length() + idx;
    }
    OPENVINO_ASSERT(idx >= 0 && shape.rank().get_length() > idx,
                    "Height dimension is out of bounds ",
                    std::to_string(idx));
    return idx;
}

inline size_t get_and_check_channels_idx(const Layout& layout, const PartialShape& shape) {
    OPENVINO_ASSERT(ov::layout::has_channels(layout),
                    "Layout ",
                    layout.to_string(),
                    " doesn't have `channels` dimension");
    OPENVINO_ASSERT(shape.rank().is_static(), "Can't get shape channels index for shape with dynamic rank");
    auto idx = ov::layout::channels(layout);
    if (idx < 0) {
        idx = shape.rank().get_length() + idx;
    }
    OPENVINO_ASSERT(idx >= 0 && shape.rank().get_length() > idx,
                    "Channels dimension is out of bounds ",
                    std::to_string(idx));
    return idx;
}

/// \brief Preprocessing context passed to each preprocessing operation.
/// This is internal structure which is not shared to custom operations yet.
class PreprocessingContext {
public:
    explicit PreprocessingContext(Layout layout) : m_layout(std::move(layout)) {}

    const Layout& layout() const {
        return m_layout;
    }

    Layout& layout() {
        return m_layout;
    }

    const PartialShape& network_shape() const {
        return m_network_shape;
    }

    PartialShape& network_shape() {
        return m_network_shape;
    }

    const Layout& network_layout() const {
        return m_network_layout;
    }

    Layout& network_layout() {
        return m_network_layout;
    }

    size_t get_network_height_for_resize() const {
        auto network_height_idx = get_and_check_height_idx(network_layout(), network_shape());
        OPENVINO_ASSERT(network_shape()[network_height_idx].is_static(),
                        "Dynamic resize: Network height dimension shall be static");
        return network_shape()[network_height_idx].get_length();
    }

    size_t get_network_width_for_resize() const {
        auto network_width_idx = get_and_check_width_idx(network_layout(), network_shape());
        OPENVINO_ASSERT(network_shape()[network_width_idx].is_static(),
                        "Dynamic resize: Network width dimension shall be static");
        return network_shape()[network_width_idx].get_length();
    }

    const ColorFormat& color_format() const {
        return m_color_format;
    }

    ColorFormat& color_format() {
        return m_color_format;
    }

private:
    Layout m_layout;
    PartialShape m_network_shape;
    Layout m_network_layout;
    ColorFormat m_color_format = ColorFormat::UNDEFINED;
};

using InternalPreprocessOp =
    std::function<std::shared_ptr<ov::Node>(const std::vector<std::shared_ptr<ov::Node>>& nodes,
                                            PreprocessingContext& context)>;

/// \brief PreProcessStepsImpl - internal data structure
class PreProcessSteps::PreProcessStepsImpl {
public:
    void add_scale_impl(const std::vector<float>& values);
    void add_mean_impl(const std::vector<float>& values);
    void add_convert_impl(const element::Type& type);
    void add_resize_impl(ResizeAlgorithm alg, int dst_height, int dst_width);
    void add_convert_layout_impl(const Layout& layout);
    void add_convert_color_impl(const ColorFormat& dst_format);

    const std::list<std::tuple<InternalPreprocessOp, bool>>& actions() const {
        return m_actions;
    }
    std::list<std::tuple<InternalPreprocessOp, bool>>& actions() {
        return m_actions;
    }

private:
    std::list<std::tuple<InternalPreprocessOp, bool>> m_actions;
};

}  // namespace preprocess
}  // namespace ov
