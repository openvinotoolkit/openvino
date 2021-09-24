// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/preprocess/pre_post_process.hpp"

#include "ngraph/opsets/opset1.hpp"
#include "openvino/core/function.hpp"
#include "preprocess_steps_impl.hpp"

namespace ov {
namespace preprocess {

/// \brief InputTensorInfoImpl - internal data structure
class InputTensorInfo::InputTensorInfoImpl {
public:
    InputTensorInfoImpl() = default;

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

    bool is_spacial_shape_set() const {
        return m_spacial_shape_set;
    }

    int get_spacial_width() const {
        return m_spacial_width;
    }

    int get_spacial_height() const {
        return m_spacial_height;
    }

    bool is_spacial_shape_dynamic() const {
        return m_spacial_shape_set && m_spacial_width == -1 && m_spacial_height == -1;
    }

    void set_spacial_dynamic_shape() {
        m_spacial_shape_set = true;
        m_spacial_width = -1;
        m_spacial_height = -1;
    }

    void set_spacial_static_shape(size_t height, size_t width) & {
        m_spacial_shape_set = true;
        m_spacial_height = static_cast<int>(height);
        m_spacial_width = static_cast<int>(width);
    }

private:
    element::Type m_type = element::dynamic;
    bool m_type_set = false;

    Layout m_layout = Layout();
    bool m_layout_set = false;

    int m_spacial_width = -1;
    int m_spacial_height = -1;
    bool m_spacial_shape_set = false;
};

/// \brief InputNetworkInfoImpl - internal data structure
class InputNetworkInfo::InputNetworkInfoImpl {
public:
    InputNetworkInfoImpl() = default;

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

/// \brief InputInfoImpl - internal data structure
struct InputInfo::InputInfoImpl {
    InputInfoImpl() = default;
    explicit InputInfoImpl(size_t idx) : m_has_index(true), m_index(idx) {}

    bool has_index() const {
        return m_has_index;
    }

    void create_tensor_data(const element::Type& type, const Layout& layout) {
        auto data = std::unique_ptr<InputTensorInfo::InputTensorInfoImpl>(new InputTensorInfo::InputTensorInfoImpl());
        data->set_layout(layout);
        data->set_element_type(type);
        m_tensor_data = std::move(data);
    }

    bool m_has_index = false;
    size_t m_index = 0;
    std::unique_ptr<InputTensorInfo::InputTensorInfoImpl> m_tensor_data;
    std::unique_ptr<PreProcessSteps::PreProcessStepsImpl> m_preprocess;
    std::unique_ptr<InputNetworkInfo::InputNetworkInfoImpl> m_network_data;
};

//-------------- InputInfo ------------------
InputInfo::InputInfo() : m_impl(std::unique_ptr<InputInfoImpl>(new InputInfoImpl)) {}
InputInfo::InputInfo(size_t input_index) : m_impl(std::unique_ptr<InputInfoImpl>(new InputInfoImpl(input_index))) {}
InputInfo::InputInfo(InputInfo&&) noexcept = default;
InputInfo& InputInfo::operator=(InputInfo&&) noexcept = default;
InputInfo::~InputInfo() = default;

InputInfo& InputInfo::tensor(InputTensorInfo&& builder) & {
    m_impl->m_tensor_data = std::move(builder.m_impl);
    return *this;
}

InputInfo&& InputInfo::tensor(InputTensorInfo&& builder) && {
    m_impl->m_tensor_data = std::move(builder.m_impl);
    return std::move(*this);
}

InputInfo&& InputInfo::preprocess(PreProcessSteps&& builder) && {
    m_impl->m_preprocess = std::move(builder.m_impl);
    return std::move(*this);
}

InputInfo& InputInfo::preprocess(PreProcessSteps&& builder) & {
    m_impl->m_preprocess = std::move(builder.m_impl);
    return *this;
}

InputInfo& InputInfo::network(InputNetworkInfo&& builder) & {
    m_impl->m_network_data = std::move(builder.m_impl);
    return *this;
}

InputInfo&& InputInfo::network(InputNetworkInfo&& builder) && {
    m_impl->m_network_data = std::move(builder.m_impl);
    return std::move(*this);
}

// ------------------------ PrePostProcessor --------------------
struct PrePostProcessor::PrePostProcessorImpl {
public:
    std::list<std::unique_ptr<InputInfo::InputInfoImpl>> in_contexts;
};

PrePostProcessor::PrePostProcessor() : m_impl(std::unique_ptr<PrePostProcessorImpl>(new PrePostProcessorImpl())) {}
PrePostProcessor::PrePostProcessor(PrePostProcessor&&) noexcept = default;
PrePostProcessor& PrePostProcessor::operator=(PrePostProcessor&&) noexcept = default;
PrePostProcessor::~PrePostProcessor() = default;

PrePostProcessor& PrePostProcessor::input(InputInfo&& builder) & {
    m_impl->in_contexts.push_back(std::move(builder.m_impl));
    return *this;
}

PrePostProcessor&& PrePostProcessor::input(InputInfo&& builder) && {
    m_impl->in_contexts.push_back(std::move(builder.m_impl));
    return std::move(*this);
}

std::shared_ptr<Function> PrePostProcessor::build(const std::shared_ptr<Function>& function) {
    bool tensor_data_updated = false;
    for (const auto& input : m_impl->in_contexts) {
        std::shared_ptr<op::v0::Parameter> param;
        OPENVINO_ASSERT(input, "Internal error: Invalid preprocessing input, please report a problem");
        if (input->has_index()) {
            param = function->get_parameters().at(input->m_index);
        } else {
            // Default case
            OPENVINO_ASSERT(function->get_parameters().size() == 1,
                            std::string("Preprocessing info expects having 1 input, however function has ") +
                                std::to_string(function->get_parameters().size()) +
                                " inputs. Please use ov::preprocess::InputInfo constructor specifying "
                                "particular input instead of default one");
            param = function->get_parameters().front();
        }
        // Set parameter layout from 'network' information
        if (input->m_network_data && input->m_network_data->is_layout_set() && param->get_layout() == Layout()) {
            param->set_layout(input->m_network_data->get_layout());
        }
        auto consumers = param->output(0).get_target_inputs();
        if (!input->m_tensor_data) {
            input->create_tensor_data(param->get_element_type(), param->get_layout());
        }
        if (!input->m_tensor_data->is_layout_set() && param->get_layout() != Layout()) {
            input->m_tensor_data->set_layout(param->get_layout());
        }
        if (!input->m_tensor_data->is_element_type_set()) {
            input->m_tensor_data->set_element_type(param->get_element_type());
        }
        auto new_param_shape = param->get_partial_shape();
        if (input->m_tensor_data->is_spacial_shape_dynamic()) {
            // Use dynamic spacial dimensions
            OPENVINO_ASSERT(input->m_tensor_data->is_layout_set(),
                            "Can't set spacial dynamic dimensions when tensor or network layout are not specified");
            auto height_idx = get_and_check_height_idx(input->m_tensor_data->get_layout(), new_param_shape);
            auto width_idx = get_and_check_width_idx(input->m_tensor_data->get_layout(), new_param_shape);
            new_param_shape[height_idx] = Dimension::dynamic();
            new_param_shape[width_idx] = Dimension::dynamic();
        }
        auto new_param = std::make_shared<op::v0::Parameter>(input->m_tensor_data->get_element_type(), new_param_shape);
        if (input->m_tensor_data->is_layout_set()) {
            new_param->set_layout(input->m_tensor_data->get_layout());
        }
        // Old param will be removed, so friendly name can be reused
        new_param->set_friendly_name(param->get_friendly_name());

        // Also reuse names of original tensor
        new_param->get_output_tensor(0).set_names(param->get_output_tensor(0).get_names());

        std::shared_ptr<Node> node = new_param;
        PreprocessingContext context(new_param->get_layout());
        context.network_layout() = param->get_layout();
        context.network_shape() = param->get_partial_shape();
        // 2. Apply preprocessing
        for (const auto& action : input->m_preprocess->actions()) {
            node = std::get<0>(action)({node}, context);
            tensor_data_updated |= std::get<1>(action);
        }

        // Check final type
        OPENVINO_ASSERT(node->get_element_type() == param->get_element_type(),
                        std::string("Element type after preprocessing {") + node->get_element_type().c_type_string() +
                            std::string("} doesn't match with network element type {") +
                            param->get_element_type().c_type_string() +
                            "}. Please add 'convert_element_type' explicitly");

        // Replace parameter
        for (auto consumer : consumers) {
            consumer.replace_source_output(node);
        }
        function->add_parameters({new_param});
        // remove old parameter
        function->remove_parameter(param);
    }
    if (tensor_data_updated) {
        function->validate_nodes_and_infer_types();
    }
    return function;
}

// --------------------- InputTensorInfo ------------------
InputTensorInfo::InputTensorInfo() : m_impl(std::unique_ptr<InputTensorInfoImpl>(new InputTensorInfoImpl())) {}
InputTensorInfo::InputTensorInfo(InputTensorInfo&&) noexcept = default;
InputTensorInfo& InputTensorInfo::operator=(InputTensorInfo&&) noexcept = default;
InputTensorInfo::~InputTensorInfo() = default;

InputTensorInfo& InputTensorInfo::set_element_type(const element::Type& type) & {
    m_impl->set_element_type(type);
    return *this;
}

InputTensorInfo&& InputTensorInfo::set_element_type(const element::Type& type) && {
    m_impl->set_element_type(type);
    return std::move(*this);
}

InputTensorInfo& InputTensorInfo::set_layout(const Layout& layout) & {
    m_impl->set_layout(layout);
    return *this;
}

InputTensorInfo&& InputTensorInfo::set_layout(const Layout& layout) && {
    m_impl->set_layout(layout);
    return std::move(*this);
}

InputTensorInfo& InputTensorInfo::set_spacial_dynamic_shape() & {
    m_impl->set_spacial_dynamic_shape();
    return *this;
}

InputTensorInfo&& InputTensorInfo::set_spacial_dynamic_shape() && {
    m_impl->set_spacial_dynamic_shape();
    return std::move(*this);
}

InputTensorInfo& InputTensorInfo::set_spacial_static_shape(size_t height, size_t width) & {
    m_impl->set_spacial_static_shape(height, width);
    return *this;
}

InputTensorInfo&& InputTensorInfo::set_spacial_static_shape(size_t height, size_t width) && {
    m_impl->set_spacial_static_shape(height, width);
    return std::move(*this);
}

// --------------------- InputNetworkInfo ------------------
InputNetworkInfo::InputNetworkInfo() : m_impl(std::unique_ptr<InputNetworkInfoImpl>(new InputNetworkInfoImpl())) {}
InputNetworkInfo::InputNetworkInfo(InputNetworkInfo&&) noexcept = default;
InputNetworkInfo& InputNetworkInfo::operator=(InputNetworkInfo&&) noexcept = default;
InputNetworkInfo::~InputNetworkInfo() = default;

InputNetworkInfo& InputNetworkInfo::set_layout(const Layout& layout) & {
    m_impl->set_layout(layout);
    return *this;
}

InputNetworkInfo&& InputNetworkInfo::set_layout(const Layout& layout) && {
    m_impl->set_layout(layout);
    return std::move(*this);
}

// --------------------- PreProcessSteps ------------------

PreProcessSteps::PreProcessSteps() : m_impl(std::unique_ptr<PreProcessStepsImpl>(new PreProcessStepsImpl())) {}
PreProcessSteps::PreProcessSteps(PreProcessSteps&&) noexcept = default;
PreProcessSteps& PreProcessSteps::operator=(PreProcessSteps&&) noexcept = default;
PreProcessSteps::~PreProcessSteps() = default;

PreProcessSteps& PreProcessSteps::scale(float value) & {
    m_impl->add_scale_impl(std::vector<float>{value});
    return *this;
}

PreProcessSteps&& PreProcessSteps::scale(float value) && {
    m_impl->add_scale_impl(std::vector<float>{value});
    return std::move(*this);
}

PreProcessSteps& PreProcessSteps::scale(const std::vector<float>& values) & {
    m_impl->add_scale_impl(values);
    return *this;
}

PreProcessSteps&& PreProcessSteps::scale(const std::vector<float>& values) && {
    m_impl->add_scale_impl(values);
    return std::move(*this);
}

PreProcessSteps& PreProcessSteps::mean(float value) & {
    m_impl->add_mean_impl(std::vector<float>{value});
    return *this;
}

PreProcessSteps&& PreProcessSteps::mean(float value) && {
    m_impl->add_mean_impl(std::vector<float>{value});
    return std::move(*this);
}

PreProcessSteps& PreProcessSteps::mean(const std::vector<float>& values) & {
    m_impl->add_mean_impl(values);
    return *this;
}

PreProcessSteps&& PreProcessSteps::mean(const std::vector<float>& values) && {
    m_impl->add_mean_impl(values);
    return std::move(*this);
}

PreProcessSteps& PreProcessSteps::convert_element_type(const element::Type& type) & {
    m_impl->add_convert_impl(type);
    return *this;
}

PreProcessSteps&& PreProcessSteps::convert_element_type(const element::Type& type) && {
    m_impl->add_convert_impl(type);
    return std::move(*this);
}

PreProcessSteps& PreProcessSteps::resize(ResizeAlgorithm alg, size_t dst_height, size_t dst_width) & {
    m_impl->add_resize_impl(alg, static_cast<int>(dst_height), static_cast<int>(dst_width));
    return *this;
}

PreProcessSteps&& PreProcessSteps::resize(ResizeAlgorithm alg, size_t dst_height, size_t dst_width) && {
    m_impl->add_resize_impl(alg, static_cast<int>(dst_height), static_cast<int>(dst_width));
    return std::move(*this);
}

PreProcessSteps& PreProcessSteps::resize(ResizeAlgorithm alg) & {
    m_impl->add_resize_impl(alg, -1, -1);
    return *this;
}

PreProcessSteps&& PreProcessSteps::resize(ResizeAlgorithm alg) && {
    m_impl->add_resize_impl(alg, -1, -1);
    return std::move(*this);
}

PreProcessSteps& PreProcessSteps::custom(const CustomPreprocessOp& preprocess_cb) & {
    // 'true' indicates that custom preprocessing step will trigger validate_and_infer_types
    m_impl->actions().emplace_back(std::make_tuple(
        [preprocess_cb](const std::vector<std::shared_ptr<ov::Node>>& nodes, PreprocessingContext&) {
            OPENVINO_ASSERT(nodes.size() == 1,
                            "Can't apply custom preprocessing step for multi-plane input. Suggesting to convert "
                            "current image to RGB/BGR color format using 'convert_color'");
            return preprocess_cb(nodes[0]);
        },
        true));
    return *this;
}

PreProcessSteps&& PreProcessSteps::custom(const CustomPreprocessOp& preprocess_cb) && {
    // 'true' indicates that custom preprocessing step will trigger validate_and_infer_types
    m_impl->actions().emplace_back(std::make_tuple(
        [preprocess_cb](const std::vector<std::shared_ptr<ov::Node>>& nodes, PreprocessingContext&) {
            OPENVINO_ASSERT(nodes.size() == 1,
                            "Can't apply custom preprocessing step for multi-plane input. Suggesting to convert "
                            "current image to RGB/BGR color format using 'convert_color'");
            return preprocess_cb(nodes[0]);
        },
        true));
    return std::move(*this);
}

}  // namespace preprocess
}  // namespace ov
