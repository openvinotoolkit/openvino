// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/preprocess/input_info.hpp>
#include <openvino/core/preprocess/output_info.hpp>
#include <openvino/opsets/opset8.hpp>

#include "color_utils.hpp"
#include "preprocess_steps_impl.hpp"

namespace ov {
namespace preprocess {

/// \brief ModelInfoImpl - internal data structure
class ModelInfoImpl {
public:
    ModelInfoImpl() = default;

    void set_layout(const Layout& layout) {
        m_layout = layout;
        m_layout_set = true;
    }
    bool is_layout_set() const {
        return m_layout_set;
    }
    const Layout& get_layout() const {
        return m_layout;
    }

private:
    Layout m_layout = Layout();
    bool m_layout_set = false;
};

class InputModelInfo::InputModelInfoImpl : public ModelInfoImpl {};

class OutputModelInfo::OutputModelInfoImpl : public ModelInfoImpl {
public:
    void set_color_format(const ColorFormat& color_format, const std::vector<std::string>& sub_names = {}) {
        m_color_format_set = (color_format == ColorFormat::RGB) || (color_format == ColorFormat::BGR);
        OPENVINO_ASSERT(m_color_format_set);
        m_color_format = color_format;
        m_planes_sub_names = sub_names;
    }
    bool is_color_format_set() const {
        return m_color_format_set;
    }
    const ColorFormat& get_color_format() const {
        return m_color_format;
    }

private:
    ColorFormat m_color_format = ColorFormat::UNDEFINED;
    std::vector<std::string> m_planes_sub_names{};
    bool m_color_format_set = false;
};

/// \brief OutputInfoImpl - internal data structure
struct OutputInfo::OutputInfoImpl {
    OutputInfoImpl() = default;

    std::unique_ptr<OutputTensorInfo::OutputTensorInfoImpl>& get_tensor_data() {
        return m_tensor_info.m_impl;
    }

    const std::unique_ptr<OutputTensorInfo::OutputTensorInfoImpl>& get_tensor_data() const {
        return m_tensor_info.m_impl;
    }

    std::unique_ptr<PostProcessSteps::PostProcessStepsImpl>& get_postprocess() {
        return m_postprocess.m_impl;
    }

    const std::unique_ptr<PostProcessSteps::PostProcessStepsImpl>& get_postprocess() const {
        return m_postprocess.m_impl;
    }

    std::unique_ptr<OutputModelInfo::OutputModelInfoImpl>& get_model_data() {
        return m_model_info.m_impl;
    }

    const std::unique_ptr<OutputModelInfo::OutputModelInfoImpl>& get_model_data() const {
        return m_model_info.m_impl;
    }

    void build(ov::ResultVector& results);

    void dump(std::ostream& str) const;

    OutputTensorInfo m_tensor_info;
    PostProcessSteps m_postprocess;
    OutputModelInfo m_model_info;
    ov::Output<ov::Node> m_output_node;
};

class TensorInfoImplBase {
public:
    TensorInfoImplBase() = default;

    void set_element_type(const element::Type& type) {
        m_type = type;
        m_type_set = true;
    }
    bool is_element_type_set() const {
        return m_type_set;
    }
    const element::Type& get_element_type() const {
        return m_type;
    }

    void set_layout(const Layout& layout) {
        m_layout = layout;
        m_layout_set = true;
    }
    bool is_layout_set() const {
        return m_layout_set;
    }
    const Layout& get_layout() const {
        return m_layout;
    }

    void set_names_compatibility_mode(const bool compatiblity_mode) {
        m_names_compatiblity_mode = compatiblity_mode;
    }

    const bool get_names_compatibility_mode() const {
        return m_names_compatiblity_mode;
    }

protected:
    element::Type m_type = element::dynamic;
    bool m_type_set = false;

    Layout m_layout = Layout();
    bool m_layout_set = false;
    bool m_names_compatiblity_mode = false;
};

class OutputTensorInfo::OutputTensorInfoImpl : public TensorInfoImplBase {};

/// \brief InputTensorInfoImpl - internal data structure
class InputTensorInfo::InputTensorInfoImpl : public TensorInfoImplBase {
public:
    InputTensorInfoImpl() = default;

    bool is_spatial_shape_set() const {
        return m_spatial_shape_set;
    }

    int get_spatial_width() const {
        return m_spatial_width;
    }

    int get_spatial_height() const {
        return m_spatial_height;
    }

    bool is_spatial_shape_dynamic() const {
        return m_spatial_shape_set && m_spatial_width == -1 && m_spatial_height == -1;
    }

    void set_spatial_dynamic_shape() {
        OPENVINO_ASSERT(!m_shape_set,
                        "'set_spatial_dynamic_shape' and 'set_shape/set_from' shall not be used together");
        m_spatial_shape_set = true;
        m_spatial_width = -1;
        m_spatial_height = -1;
    }

    void set_spatial_static_shape(size_t height, size_t width) & {
        OPENVINO_ASSERT(!m_shape_set, "'set_spatial_static_shape' and 'set_shape/set_from' shall not be used together");
        m_spatial_shape_set = true;
        m_spatial_height = static_cast<int>(height);
        m_spatial_width = static_cast<int>(width);
    }

    const ColorFormat& get_color_format() const {
        return m_color_format;
    }

    void set_color_format(ColorFormat format, const std::vector<std::string>& sub_names) {
        auto info = ColorFormatInfo::get(format);
        if (info->planes_count() == 1) {
            OPENVINO_ASSERT(sub_names.empty(),
                            "Plane names are not allowed for single plane color format '",
                            color_format_name(format),
                            "'");
        } else if (!sub_names.empty()) {
            OPENVINO_ASSERT(sub_names.size() == info->planes_count(),
                            "Number of sub-names (",
                            sub_names.size(),
                            ") shall match with number of planes for '",
                            color_format_name(format),
                            "' color format (",
                            info->planes_count(),
                            ")");
        }
        m_planes_sub_names = sub_names;
        m_color_format = format;
    }

    const std::vector<std::string>& planes_sub_names() const {
        return m_planes_sub_names;
    }

    void set_memory_type(const std::string& mem_type) {
        m_memory_type_set = true;
        m_memory_type = mem_type;
    }

    const std::string& get_memory_type() const {
        return m_memory_type;
    }

    bool is_memory_type_set() const {
        return m_memory_type_set;
    }

    void set_shape(const PartialShape& shape) {
        OPENVINO_ASSERT(
            !m_spatial_shape_set,
            "'set_spatial_static_shape', 'set_spatial_dynamic_shape', 'set_shape/set_from' shall not be used together");
        m_shape = shape;
        m_shape_set = true;
    }

    void set_from(const ov::Tensor& runtime_tensor) {
        set_shape(runtime_tensor.get_shape());
        set_element_type(runtime_tensor.get_element_type());
    }

    bool is_shape_set() const {
        return m_shape_set;
    }

    const PartialShape& get_shape() const {
        return m_shape;
    }

private:
    ColorFormat m_color_format = ColorFormat::UNDEFINED;
    std::vector<std::string> m_planes_sub_names;

    element::Type m_type = element::dynamic;
    Layout m_layout = Layout();

    int m_spatial_width = -1;
    int m_spatial_height = -1;
    bool m_spatial_shape_set = false;

    std::string m_memory_type = {};
    bool m_memory_type_set = false;

    PartialShape m_shape = {};
    bool m_shape_set = false;
};

/// \brief InputInfoImpl - internal data structure
struct InputInfo::InputInfoImpl {
    struct InputInfoData {
        std::vector<std::shared_ptr<opset8::Parameter>> m_new_params;
        std::shared_ptr<opset8::Parameter> m_param;
        Layout m_model_layout;
        Layout m_tensor_layout;
        std::vector<Output<Node>> as_nodes() const {
            std::vector<Output<Node>> res;
            std::transform(m_new_params.begin(),
                           m_new_params.end(),
                           std::back_inserter(res),
                           [](const std::shared_ptr<opset8::Parameter>& param) {
                               return param;
                           });
            return res;
        }
    };
    InputInfoImpl() = default;

    std::unique_ptr<InputTensorInfo::InputTensorInfoImpl>& get_tensor_data() {
        return m_tensor_info.m_impl;
    }

    const std::unique_ptr<InputTensorInfo::InputTensorInfoImpl>& get_tensor_data() const {
        return m_tensor_info.m_impl;
    }

    std::unique_ptr<PreProcessSteps::PreProcessStepsImpl>& get_preprocess() {
        return m_preprocess.m_impl;
    }

    const std::unique_ptr<PreProcessSteps::PreProcessStepsImpl>& get_preprocess() const {
        return m_preprocess.m_impl;
    }

    const std::unique_ptr<InputModelInfo::InputModelInfoImpl>& get_model() const {
        return m_model_data.m_impl;
    }

    InputInfoData create_new_params(std::tuple<std::unordered_set<std::string>, bool>& existing_names,
                                    const std::shared_ptr<Model>& model) const;

    static PreStepsList create_implicit_steps(const PreprocessingContext& context, element::Type type);

    bool build(const std::shared_ptr<Model>& model,
               std::tuple<std::unordered_set<std::string>, bool>& existing_names,
               std::list<std::shared_ptr<opset8::Parameter>>& parameters_list);

    void dump(std::ostream& str,
              const std::shared_ptr<Model>& model,
              std::tuple<std::unordered_set<std::string>, bool>& existing_names) const;

    InputTensorInfo m_tensor_info;
    PreProcessSteps m_preprocess;
    InputModelInfo m_model_data;
    std::shared_ptr<op::v0::Parameter> m_resolved_param;
};

}  // namespace preprocess
}  // namespace ov
