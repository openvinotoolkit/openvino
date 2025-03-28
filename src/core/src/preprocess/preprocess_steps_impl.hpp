// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>

#include "openvino/core/layout.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/preprocess/color_format.hpp"
#include "openvino/core/preprocess/postprocess_steps.hpp"
#include "openvino/core/preprocess/preprocess_steps.hpp"
#include "tensor_name_util.hpp"

namespace ov {
namespace preprocess {

inline size_t get_and_check_width_idx(const Layout& layout, const PartialShape& shape) {
    OPENVINO_ASSERT(ov::layout::has_width(layout), "Layout ", layout.to_string(), " doesn't have `width` dimension");
    OPENVINO_ASSERT(shape.rank().is_static(), "Can't get shape width index for shape with dynamic rank");
    auto idx = ov::layout::width_idx(layout);
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
    auto idx = ov::layout::height_idx(layout);
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
    auto idx = ov::layout::channels_idx(layout);
    if (idx < 0) {
        idx = shape.rank().get_length() + idx;
    }
    OPENVINO_ASSERT(idx >= 0 && shape.rank().get_length() > idx,
                    "Channels dimension is out of bounds ",
                    std::to_string(idx));
    return idx;
}

/// \brief Context passed to each pre/post-processing operation.
/// This is internal structure which is not shared to custom operations yet.
class PrePostProcessingContextBase {
public:
    explicit PrePostProcessingContextBase(Layout layout) : m_layout(std::move(layout)) {}

    const Layout& layout() const {
        return m_layout;
    }

    Layout& layout() {
        return m_layout;
    }

    // Final layout. Needed if user specified convert_layout without arguments
    // For preprocessing it is parameter's model layout
    // For post-processing it is result's tensor layout
    const Layout& target_layout() const {
        return m_target_layout;
    }

    Layout& target_layout() {
        return m_target_layout;
    }

    element::Type target_element_type() const {
        return m_target_element_type;
    }

    element::Type& target_element_type() {
        return m_target_element_type;
    }

    const ColorFormat& color_format() const {
        return m_color_format;
    }

    ColorFormat& color_format() {
        return m_color_format;
    }

protected:
    Layout m_layout;
    Layout m_target_layout;
    element::Type m_target_element_type;
    ColorFormat m_color_format = ColorFormat::UNDEFINED;
};

/// \brief Preprocessing context passed to each preprocessing operation.
/// This is internal structure which is not shared to custom operations yet.
class PreprocessingContext : public PrePostProcessingContextBase {
public:
    explicit PreprocessingContext(const Layout& layout) : PrePostProcessingContextBase(layout) {}

    const PartialShape& model_shape() const {
        return m_model_shape;
    }

    PartialShape& model_shape() {
        return m_model_shape;
    }

    size_t get_model_height_for_resize() const {
        auto model_height_idx = get_and_check_height_idx(target_layout(), model_shape());
        OPENVINO_ASSERT(model_shape()[model_height_idx].is_static(),
                        "Dynamic resize: Model height dimension shall be static");
        return model_shape()[model_height_idx].get_length();
    }

    size_t get_model_width_for_resize() const {
        auto model_width_idx = get_and_check_width_idx(target_layout(), model_shape());
        OPENVINO_ASSERT(model_shape()[model_width_idx].is_static(),
                        "Dynamic resize: Model width dimension shall be static");
        return model_shape()[model_width_idx].get_length();
    }

private:
    PartialShape m_model_shape;
    Layout m_model_layout;
};

using InternalPreprocessOp =
    std::function<std::tuple<std::vector<Output<Node>>, bool>(const std::vector<Output<Node>>& nodes,
                                                              const std::shared_ptr<Model>& function,
                                                              PreprocessingContext& context)>;

struct InternalPreprocessAction {
    InternalPreprocessAction(InternalPreprocessOp op, std::string name)
        : m_op(std::move(op)),
          m_name(std::move(name)) {}
    InternalPreprocessOp m_op;
    std::string m_name;
};

/// \brief PreProcessStepsImpl - internal data structure
class PreStepsList {
public:
    void add_scale_impl(const std::vector<float>& values);
    void add_clamp(double min_value, double max_value);
    void add_mean_impl(const std::vector<float>& values);
    void add_pad_impl(const std::vector<int>& pads_begin,
                      const std::vector<int>& pads_end,
                      const std::vector<float>& values,
                      PaddingMode mode);
    void add_convert_impl(const element::Type& type);
    void add_crop_impl(const std::vector<int>& begin, const std::vector<int>& end);
    void add_resize_impl(ResizeAlgorithm alg, int dst_height, int dst_width);
    void add_convert_layout_impl(const Layout& layout);
    void add_convert_layout_impl(const std::vector<uint64_t>& dims);
    void add_convert_color_impl(const ColorFormat& dst_format);
    void add_reverse_channels();
    std::tuple<PartialShape, Layout> calculate_param_shape(const PartialShape& model_shape,
                                                           const Layout& model_layout) const;

    const std::list<InternalPreprocessAction>& actions() const {
        return m_actions;
    }
    std::list<InternalPreprocessAction>& actions() {
        return m_actions;
    }

    Layout propagate_layout(const Layout& tensor_layout) const;

private:
    static std::tuple<std::vector<Output<Node>>, bool> reverse_channels(const std::vector<Output<Node>>& nodes,
                                                                        const std::shared_ptr<Model>& function,
                                                                        PreprocessingContext& context);

    static std::tuple<std::vector<Output<Node>>, bool> cut_last_channel(const std::vector<Output<Node>>& nodes,
                                                                        const std::shared_ptr<Model>& function,
                                                                        PreprocessingContext& context);

private:
    std::list<InternalPreprocessAction> m_actions;
    std::list<std::vector<uint64_t>> m_layout_converts;
    std::list<std::vector<uint64_t>> m_forward_layout_converts;
    Layout m_last_explicit_layout;
    bool m_last_explicit_layout_set = false;
};

class PreProcessSteps::PreProcessStepsImpl : public PreStepsList {};

//------ Post process -----
class PostprocessingContext : public PrePostProcessingContextBase {
public:
    explicit PostprocessingContext(const Layout& layout) : PrePostProcessingContextBase(layout) {}
};

using InternalPostprocessOp = std::function<std::tuple<ov::Output<ov::Node>, bool>(const ov::Output<ov::Node>& node,
                                                                                   PostprocessingContext& context)>;

struct InternalPostprocessAction {
    InternalPostprocessAction(InternalPostprocessOp op, std::string name)
        : m_op(std::move(op)),
          m_name(std::move(name)) {}
    InternalPostprocessOp m_op;
    std::string m_name;
};

/// \brief PostProcessStepsImpl - internal data structure
class PostStepsList {
public:
    void add_clamp(double min_value, double max_value);
    void add_convert_impl(const element::Type& type);
    void add_convert_layout_impl(const Layout& layout);
    void add_convert_layout_impl(const std::vector<uint64_t>& dims);
    void add_convert_color_impl(const ColorFormat& dst_format);

    const std::list<InternalPostprocessAction>& actions() const {
        return m_actions;
    }
    std::list<InternalPostprocessAction>& actions() {
        return m_actions;
    }

private:
    static std::tuple<Output<Node>, bool> reverse_channels(const Output<Node>& nodes, PostprocessingContext& context);

private:
    std::list<InternalPostprocessAction> m_actions;
};

class PostProcessSteps::PostProcessStepsImpl : public PostStepsList {};

}  // namespace preprocess
}  // namespace ov
