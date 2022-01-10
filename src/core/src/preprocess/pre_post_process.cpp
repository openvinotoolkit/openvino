// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/preprocess/pre_post_process.hpp"

#include "color_utils.hpp"
#include "function_guard.hpp"
#include "layout_utils.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "openvino/core/model.hpp"
#include "preprocess_steps_impl.hpp"

namespace ov {
namespace preprocess {

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

protected:
    element::Type m_type = element::dynamic;
    bool m_type_set = false;

    Layout m_layout = Layout();
    bool m_layout_set = false;
};

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
        OPENVINO_ASSERT(!m_shape_set, "'set_spatial_dynamic_shape' and 'set_shape' shall not be used together");
        m_spatial_shape_set = true;
        m_spatial_width = -1;
        m_spatial_height = -1;
    }

    void set_spatial_static_shape(size_t height, size_t width) & {
        OPENVINO_ASSERT(!m_shape_set, "'set_spatial_static_shape' and 'set_shape' shall not be used together");
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
            "'set_spatial_static_shape', 'set_spatial_dynamic_shape', 'set_shape' shall not be used together");
        m_shape = shape;
        m_shape_set = true;
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
    bool m_type_set = false;

    Layout m_layout = Layout();
    bool m_layout_set = false;

    int m_spatial_width = -1;
    int m_spatial_height = -1;
    bool m_spatial_shape_set = false;

    std::string m_memory_type = {};
    bool m_memory_type_set = false;

    PartialShape m_shape = {};
    bool m_shape_set = false;
};

class OutputTensorInfo::OutputTensorInfoImpl : public TensorInfoImplBase {};

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

class OutputModelInfo::OutputModelInfoImpl : public ModelInfoImpl {};

/// \brief InputInfoImpl - internal data structure
struct InputInfo::InputInfoImpl {
    InputInfoImpl() = default;

    std::unique_ptr<InputTensorInfo::InputTensorInfoImpl>& get_tensor_data() {
        return m_tensor_info.m_impl;
    }

    std::unique_ptr<PreProcessSteps::PreProcessStepsImpl>& get_preprocess() {
        return m_preprocess.m_impl;
    }

    std::unique_ptr<InputModelInfo::InputModelInfoImpl>& get_model() {
        return m_model_data.m_impl;
    }

    InputTensorInfo m_tensor_info;
    PreProcessSteps m_preprocess;
    InputModelInfo m_model_data;
    std::shared_ptr<op::v0::Parameter> m_resolved_param;
};

/// \brief OutputInfoImpl - internal data structure
struct OutputInfo::OutputInfoImpl {
    OutputInfoImpl() = default;

    std::unique_ptr<OutputTensorInfo::OutputTensorInfoImpl>& get_tensor_data() {
        return m_tensor_info.m_impl;
    }

    std::unique_ptr<PostProcessSteps::PostProcessStepsImpl>& get_postprocess() {
        return m_postprocess.m_impl;
    }

    std::unique_ptr<OutputModelInfo::OutputModelInfoImpl>& get_model_data() {
        return m_model_info.m_impl;
    }

    OutputTensorInfo m_tensor_info;
    PostProcessSteps m_postprocess;
    OutputModelInfo m_model_info;
    ov::Output<ov::Node> m_output_node;
};

//-------------- InputInfo ------------------
InputInfo::InputInfo() : m_impl(std::unique_ptr<InputInfoImpl>(new InputInfoImpl)) {}
InputInfo::InputInfo(InputInfo&& other) noexcept = default;
InputInfo& InputInfo::operator=(InputInfo&& other) noexcept = default;

InputInfo::~InputInfo() = default;

InputTensorInfo& InputInfo::tensor() {
    return m_impl->m_tensor_info;
}

PreProcessSteps& InputInfo::preprocess() {
    return m_impl->m_preprocess;
}

InputModelInfo& InputInfo::model() {
    return m_impl->m_model_data;
}

//-------------- OutputInfo ------------------
OutputInfo::OutputInfo() : m_impl(std::unique_ptr<OutputInfoImpl>(new OutputInfoImpl)) {}
OutputInfo::OutputInfo(OutputInfo&& other) noexcept = default;
OutputInfo& OutputInfo::operator=(OutputInfo&& other) noexcept = default;
OutputInfo::~OutputInfo() = default;

OutputModelInfo& OutputInfo::model() {
    return m_impl->m_model_info;
}

PostProcessSteps& OutputInfo::postprocess() {
    return m_impl->m_postprocess;
}

OutputTensorInfo& OutputInfo::tensor() {
    return m_impl->m_tensor_info;
}

// ------------------------ PrePostProcessor --------------------
struct PrePostProcessor::PrePostProcessorImpl {
public:
    PrePostProcessorImpl() = default;
    explicit PrePostProcessorImpl(const std::shared_ptr<ov::Model>& f) : m_function(f) {
        OPENVINO_ASSERT(f, "Function can't be nullptr for PrePostProcessor");
        for (size_t i = 0; i < m_function->inputs().size(); ++i) {
            auto info = InputInfo();
            info.m_impl->m_resolved_param = m_function->get_parameters()[i];
            m_inputs.push_back(std::move(info));
        }
        for (size_t i = 0; i < m_function->outputs().size(); ++i) {
            auto info = OutputInfo();
            info.m_impl->m_output_node = m_function->output(i);
            m_outputs.push_back(std::move(info));
        }
    }

    InputInfo& find_input(const std::string& tensor_name) {
        size_t index;
        for (index = 0; index < m_function->inputs().size(); index++) {
            if (m_function->input(index).get_tensor().get_names().count(tensor_name)) {
                return m_inputs[index];
            }
        }
        OPENVINO_ASSERT(false, "Function doesn't have input with name ", tensor_name);
    }

    OutputInfo& find_output(const std::string& tensor_name) {
        size_t index;
        for (index = 0; index < m_function->outputs().size(); index++) {
            if (m_function->output(index).get_tensor().get_names().count(tensor_name)) {
                return m_outputs[index];
            }
        }
        OPENVINO_ASSERT(false, "Function doesn't have output with name ", tensor_name);
    }

    std::vector<InputInfo> m_inputs;
    std::vector<OutputInfo> m_outputs;
    std::shared_ptr<Model> m_function = nullptr;
};

PrePostProcessor::PrePostProcessor(const std::shared_ptr<Model>& function)
    : m_impl(std::unique_ptr<PrePostProcessorImpl>(new PrePostProcessorImpl(function))) {}
PrePostProcessor::PrePostProcessor(PrePostProcessor&&) noexcept = default;
PrePostProcessor& PrePostProcessor::operator=(PrePostProcessor&&) noexcept = default;
PrePostProcessor::~PrePostProcessor() = default;

InputInfo& PrePostProcessor::input() {
    OPENVINO_ASSERT(m_impl->m_inputs.size() == 1,
                    "PrePostProcessor::input() - function must have exactly one input, got ",
                    m_impl->m_inputs.size());
    return m_impl->m_inputs.front();
}

InputInfo& PrePostProcessor::input(size_t input_index) {
    OPENVINO_ASSERT(m_impl->m_inputs.size() > input_index,
                    "PrePostProcessor::input(size_t) - function doesn't have input with index ",
                    input_index,
                    ". Total number of inputs is ",
                    m_impl->m_inputs.size());
    return m_impl->m_inputs[input_index];
}

InputInfo& PrePostProcessor::input(const std::string& tensor_name) {
    return m_impl->find_input(tensor_name);
}

OutputInfo& PrePostProcessor::output() {
    OPENVINO_ASSERT(m_impl->m_outputs.size() == 1,
                    "PrePostProcessor::output() - function must have exactly one output, got ",
                    m_impl->m_outputs.size());
    return m_impl->m_outputs.front();
}

OutputInfo& PrePostProcessor::output(size_t output_index) {
    OPENVINO_ASSERT(m_impl->m_outputs.size() > output_index,
                    "PrePostProcessor::output(size_t) - function doesn't have input with index ",
                    output_index,
                    ". Total number of inputs is ",
                    m_impl->m_inputs.size());
    return m_impl->m_outputs[output_index];
}

OutputInfo& PrePostProcessor::output(const std::string& tensor_name) {
    return m_impl->find_output(tensor_name);
}

std::shared_ptr<Model> PrePostProcessor::build() {
    auto function = m_impl->m_function;
    FunctionGuard guard(function);
    std::tuple<std::unordered_set<std::string>, bool> existing_names{std::unordered_set<std::string>{}, false};
    bool tensor_data_updated = false;
    for (const auto& input_info : m_impl->m_inputs) {
        auto& input = input_info.m_impl;
        // Set parameter layout from 'model' information
        if (input->get_model()->is_layout_set()) {
            // Overwrite existing model's layout here (fix 74065)
            input->m_resolved_param->set_layout(input->get_model()->get_layout());
        }
    }
    auto results = function->get_results();
    auto parameters_list = std::list<std::shared_ptr<op::v0::Parameter>>(function->get_parameters().begin(),
                                                                         function->get_parameters().end());

    for (const auto& input_info : m_impl->m_inputs) {
        const auto& input = input_info.m_impl;
        auto param = input->m_resolved_param;
        auto consumers = param->output(0).get_target_inputs();
        if (!input->get_tensor_data()->is_element_type_set()) {
            input->get_tensor_data()->set_element_type(param->get_element_type());
        }
        auto color_info = ColorFormatInfo::get(input->get_tensor_data()->get_color_format());
        if (!input->get_tensor_data()->is_layout_set()) {
            if (!color_info->default_layout().empty()) {
                input->get_tensor_data()->set_layout(color_info->default_layout());
            }
        }

        auto net_shape = param->get_partial_shape();
        auto new_param_shape = net_shape;
        if (input->get_tensor_data()->is_shape_set()) {
            new_param_shape = input->get_tensor_data()->get_shape();
        }
        if (input->get_tensor_data()->is_layout_set() && !param->get_layout().empty() &&
            param->get_layout() != input->get_tensor_data()->get_layout()) {
            // Find transpose between model and tensor layouts and update tensor shape
            auto net_to_tensor =
                layout::utils::find_permutation(param->get_layout(), net_shape, input->get_tensor_data()->get_layout());
            if (!net_to_tensor.empty()) {
                std::vector<ov::Dimension> dims(new_param_shape.size());
                std::transform(net_to_tensor.begin(), net_to_tensor.end(), dims.begin(), [&](int64_t v) {
                    return new_param_shape[v];
                });
                new_param_shape = PartialShape(dims);
            }
        } else {
            Layout new_layout;
            std::tie(new_param_shape, new_layout) =
                input->get_preprocess()->calculate_param_shape(new_param_shape, param->get_layout());
            if (!input->get_tensor_data()->is_layout_set()) {
                // Reusing param's layout according to converted calculated layout
                input->get_tensor_data()->set_layout(new_layout);
            }
        }

        if (input->get_tensor_data()->is_spatial_shape_set()) {
            auto height_idx = get_and_check_height_idx(input->get_tensor_data()->get_layout(), new_param_shape);
            auto width_idx = get_and_check_width_idx(input->get_tensor_data()->get_layout(), new_param_shape);
            if (input->get_tensor_data()->is_spatial_shape_dynamic()) {
                // Use dynamic spatial dimensions
                new_param_shape[height_idx] = Dimension::dynamic();
                new_param_shape[width_idx] = Dimension::dynamic();
            } else {
                // Use static spatial dimensions
                new_param_shape[height_idx] = input->get_tensor_data()->get_spatial_height();
                new_param_shape[width_idx] = input->get_tensor_data()->get_spatial_width();
            }
        }

        std::vector<Output<Node>> nodes;
        std::vector<std::shared_ptr<op::v0::Parameter>> new_params;

        // Create separate parameter for each plane. Shape is based on color format
        for (size_t plane = 0; plane < color_info->planes_count(); plane++) {
            auto plane_shape = color_info->shape(plane, new_param_shape);
            auto plane_param =
                std::make_shared<op::v0::Parameter>(input->get_tensor_data()->get_element_type(), plane_shape);
            if (plane < input->get_tensor_data()->planes_sub_names().size()) {
                std::unordered_set<std::string> plane_tensor_names;
                std::string sub_name;
                sub_name = std::string("/") + input->get_tensor_data()->planes_sub_names()[plane];
                if (!std::get<1>(existing_names)) {
                    existing_names = std::make_tuple(get_function_tensor_names(function), true);
                }
                for (const auto& tensor_name : param->get_default_output().get_tensor().get_names()) {
                    auto new_name = tensor_name + sub_name;
                    OPENVINO_ASSERT(
                        std::get<0>(existing_names).count(new_name) == 0,
                        "Error while trying to create plane input with name '",
                        new_name,
                        "' - name already exists in model. Please specify another sub-name for set_color_format");
                    plane_tensor_names.insert(new_name);
                }
                plane_param->get_default_output().get_tensor().set_names(plane_tensor_names);
                plane_param->set_friendly_name(param->get_friendly_name() + sub_name);
            } else if (color_info->planes_count() == 1) {
                plane_param->get_default_output().get_tensor().set_names(
                    param->get_default_output().get_tensor().get_names());
                plane_param->set_friendly_name(param->get_friendly_name());
            }
            // Fill runtime info
            plane_param->get_rt_info() = param->get_rt_info();
            plane_param->output(0).get_rt_info() = param->output(0).get_rt_info();
            if (!input->get_tensor_data()->get_layout().empty()) {
                plane_param->set_layout(input->get_tensor_data()->get_layout());
            }
            if (input->get_tensor_data()->is_memory_type_set()) {
                if (input->get_tensor_data()->get_memory_type().empty()) {
                    plane_param->output(0).get_rt_info().erase(TensorInfoMemoryType::get_type_info_static());
                } else {
                    plane_param->output(0).get_rt_info()[TensorInfoMemoryType::get_type_info_static()] =
                        TensorInfoMemoryType(input->get_tensor_data()->get_memory_type());
                }
            }
            new_params.push_back(plane_param);
            nodes.emplace_back(plane_param);
        }

        PreprocessingContext context(input->get_tensor_data()->get_layout());
        context.color_format() = input->get_tensor_data()->get_color_format();
        context.target_layout() = param->get_layout();
        context.model_shape() = param->get_partial_shape();
        context.target_element_type() = param->get_element_type();

        // 2. Apply preprocessing
        for (const auto& action : input->get_preprocess()->actions()) {
            auto action_result = action(nodes, function, context);
            nodes = std::get<0>(action_result);
            tensor_data_updated |= std::get<1>(action_result);
        }

        OPENVINO_ASSERT(nodes.size() == 1,
                        "Multiple plane input is not allowed as model input. Consider using of convert_color "
                        "preprocessing operation. Current format is '",
                        color_format_name(context.color_format()),
                        "'");
        OPENVINO_ASSERT(is_rgb_family(context.color_format()) || context.color_format() == ColorFormat::UNDEFINED,
                        "model shall have RGB/BGR color format. Consider add 'convert_color' preprocessing operation "
                        "to convert current color format '",
                        color_format_name(context.color_format()),
                        "'to RGB/BGR");

        // Implicit: Convert element type + layout to user's tensor implicitly
        PreStepsList implicit_steps;
        implicit_steps.add_convert_impl(param->get_element_type());
        if (!context.target_layout().empty()) {
            implicit_steps.add_convert_layout_impl(context.target_layout());
        }

        for (const auto& action : implicit_steps.actions()) {
            auto action_result = action(nodes, function, context);
            nodes = std::get<0>(action_result);
        }

        auto node = nodes[0];
        if (node.get_partial_shape() != param->get_partial_shape()) {
            tensor_data_updated = true;  // Trigger revalidation if input parameter shape is changed
        }
        // Check final shape
        OPENVINO_ASSERT(node.get_partial_shape().compatible(param->get_partial_shape()),
                        "Resulting shape '",
                        node.get_partial_shape(),
                        "' after preprocessing is not aligned with original parameter's shape: ",
                        param->get_partial_shape());

        // Replace parameter
        for (auto consumer : consumers) {
            consumer.replace_source_output(node);
        }
        {
            auto param_it = std::find(parameters_list.begin(), parameters_list.end(), param);
            OPENVINO_ASSERT(param_it != parameters_list.end(),
                            "Parameter to replace has been replaced by previous steps of preprocessing. Use only one "
                            "InputInfo for one input parameter");
            // Insert list of new parameters to the place of original parameter
            param_it = parameters_list.erase(param_it);
            parameters_list.insert(param_it, new_params.begin(), new_params.end());
        }
    }

    // Add parameters with right order
    {
        while (!function->get_parameters().empty()) {
            function->remove_parameter(*function->get_parameters().begin());
        }
        auto parameters_vec = ParameterVector(parameters_list.begin(), parameters_list.end());
        function->add_parameters(parameters_vec);
    }
    // Validate nodes after preprocessing if needed (no need to repeat it after post-processing)
    if (tensor_data_updated) {
        function->validate_nodes_and_infer_types();
    }

    // Post processing
    for (const auto& output_info : m_impl->m_outputs) {
        const auto& output = output_info.m_impl;
        std::shared_ptr<op::v0::Result> result;
        Output<Node> node = output->m_output_node;
        auto start_out_node_names = node.get_tensor().get_names();
        node.get_tensor().set_names({});
        result = std::dynamic_pointer_cast<op::v0::Result>(node.get_node_shared_ptr());
        // Set result layout from 'model' information
        if (output->get_model_data()->is_layout_set()) {
            // Overwrite existing model's layout here (fix 74065)
            result->set_layout(output->get_model_data()->get_layout());
        }
        auto parent = result->get_input_source_output(0);
        PostprocessingContext context(result->get_layout());
        if (output->get_tensor_data()->is_layout_set()) {
            context.target_layout() = output->get_tensor_data()->get_layout();
        }
        if (output->get_tensor_data()->is_element_type_set()) {
            context.target_element_type() = output->get_tensor_data()->get_element_type();
        }
        // Apply post-processing
        node = result->get_input_source_output(0);
        bool post_processing_applied = false;
        for (const auto& action : output->get_postprocess()->actions()) {
            auto action_result = action({node}, context);
            node = std::get<0>(action_result);
            post_processing_applied = true;
        }
        // Implicit: Convert element type + layout to user's tensor implicitly
        PostStepsList implicit_steps;
        if (node.get_element_type() != output->get_tensor_data()->get_element_type() &&
            output->get_tensor_data()->is_element_type_set() && node.get_element_type() != element::dynamic) {
            implicit_steps.add_convert_impl(output->get_tensor_data()->get_element_type());
        }

        if (!context.target_layout().empty() && context.target_layout() != context.layout()) {
            implicit_steps.add_convert_layout_impl(context.target_layout());
        }
        for (const auto& action : implicit_steps.actions()) {
            auto action_result = action({node}, context);
            node = std::get<0>(action_result);
            post_processing_applied = true;
        }
        node.get_node_shared_ptr()->set_friendly_name(
            result->get_input_source_output(0).get_node_shared_ptr()->get_friendly_name());

        // Reset friendly name of input node to avoid names collision
        // when there is at a new node inserted by post-processing steps
        // If no new nodes are inserted by post-processing, then we need to preserve friendly name of input
        // as it's required for old API correct work
        if (post_processing_applied)
            result->get_input_source_output(0).get_node_shared_ptr()->set_friendly_name("");

        // Create result
        auto new_result = std::make_shared<ov::op::v0::Result>(node);
        new_result->set_friendly_name(result->get_friendly_name());
        node.get_tensor().set_names(start_out_node_names);

        // Preserve runtime info of original result
        new_result->get_rt_info() = result->get_rt_info();
        new_result->input(0).get_rt_info() = result->input(0).get_rt_info();
        new_result->output(0).get_rt_info() = result->output(0).get_rt_info();

        // Update layout
        if (!context.layout().empty()) {
            new_result->set_layout(context.layout());
        }

        for (auto& old_result : results) {
            if (result == old_result) {
                old_result = new_result;
                break;
            }
        }
    }
    // Add results with right order
    while (!function->get_results().empty())
        function->remove_result(*function->get_results().begin());
    function->add_results(results);

    guard.reset();
    return function;
}

// --------------------- InputTensorInfo ------------------
InputTensorInfo::InputTensorInfo() : m_impl(std::unique_ptr<InputTensorInfoImpl>(new InputTensorInfoImpl())) {}
InputTensorInfo::~InputTensorInfo() = default;

InputTensorInfo& InputTensorInfo::set_element_type(const element::Type& type) {
    m_impl->set_element_type(type);
    return *this;
}

InputTensorInfo& InputTensorInfo::set_layout(const Layout& layout) {
    m_impl->set_layout(layout);
    return *this;
}

InputTensorInfo& InputTensorInfo::set_spatial_dynamic_shape() {
    m_impl->set_spatial_dynamic_shape();
    return *this;
}

InputTensorInfo& InputTensorInfo::set_spatial_static_shape(size_t height, size_t width) {
    m_impl->set_spatial_static_shape(height, width);
    return *this;
}

// --------------------- InputModelInfo ------------------
InputModelInfo::InputModelInfo() : m_impl(std::unique_ptr<InputModelInfoImpl>(new InputModelInfoImpl())) {}
InputModelInfo::~InputModelInfo() = default;

InputModelInfo& InputModelInfo::set_layout(const Layout& layout) {
    m_impl->set_layout(layout);
    return *this;
}

InputTensorInfo& InputTensorInfo::set_color_format(const ov::preprocess::ColorFormat& format,
                                                   const std::vector<std::string>& sub_names) {
    m_impl->set_color_format(format, sub_names);
    return *this;
}

InputTensorInfo& InputTensorInfo::set_memory_type(const std::string& memory_type) {
    m_impl->set_memory_type(memory_type);
    return *this;
}

InputTensorInfo& InputTensorInfo::set_shape(const PartialShape& shape) {
    m_impl->set_shape(shape);
    return *this;
}

// --------------------- PreProcessSteps ------------------

PreProcessSteps::PreProcessSteps() : m_impl(std::unique_ptr<PreProcessStepsImpl>(new PreProcessStepsImpl())) {}
PreProcessSteps::~PreProcessSteps() = default;

PreProcessSteps& PreProcessSteps::scale(float value) {
    m_impl->add_scale_impl(std::vector<float>{value});
    return *this;
}

PreProcessSteps& PreProcessSteps::scale(const std::vector<float>& values) {
    m_impl->add_scale_impl(values);
    return *this;
}

PreProcessSteps& PreProcessSteps::mean(float value) {
    m_impl->add_mean_impl(std::vector<float>{value});
    return *this;
}

PreProcessSteps& PreProcessSteps::mean(const std::vector<float>& values) {
    m_impl->add_mean_impl(values);
    return *this;
}

PreProcessSteps& PreProcessSteps::convert_element_type(const element::Type& type) {
    m_impl->add_convert_impl(type);
    return *this;
}

PreProcessSteps& PreProcessSteps::resize(ResizeAlgorithm alg, size_t dst_height, size_t dst_width) {
    OPENVINO_ASSERT(dst_height <= std::numeric_limits<int>::max() && dst_width <= std::numeric_limits<int>::max(),
                    "Resize: Width/Height dimensions cannot be greater than ",
                    std::to_string(std::numeric_limits<int>::max()));
    m_impl->add_resize_impl(alg, static_cast<int>(dst_height), static_cast<int>(dst_width));
    return *this;
}

PreProcessSteps& PreProcessSteps::resize(ResizeAlgorithm alg) {
    m_impl->add_resize_impl(alg, -1, -1);
    return *this;
}

PreProcessSteps& PreProcessSteps::convert_layout(const Layout& dst_layout) {
    m_impl->add_convert_layout_impl(dst_layout);
    return *this;
}

PreProcessSteps& PreProcessSteps::convert_layout(const std::vector<uint64_t>& dims) {
    m_impl->add_convert_layout_impl(dims);
    return *this;
}

PreProcessSteps& PreProcessSteps::convert_color(const ov::preprocess::ColorFormat& dst_format) {
    m_impl->add_convert_color_impl(dst_format);
    return *this;
}

PreProcessSteps& PreProcessSteps::custom(const CustomPreprocessOp& preprocess_cb) {
    // 'true' indicates that custom preprocessing step will trigger validate_and_infer_types
    m_impl->actions().emplace_back([preprocess_cb](const std::vector<Output<Node>>& nodes,
                                                   const std::shared_ptr<ov::Model>&,
                                                   PreprocessingContext&) {
        OPENVINO_ASSERT(nodes.size() == 1,
                        "Can't apply custom preprocessing step for multi-plane input. Suggesting to convert "
                        "current image to RGB/BGR color format using 'convert_color'");
        return std::make_tuple(std::vector<Output<Node>>{preprocess_cb(nodes[0])}, true);
    });
    return *this;
}

PreProcessSteps& PreProcessSteps::reverse_channels() {
    m_impl->add_reverse_channels();
    return *this;
}

// --------------------- OutputTensorInfo ------------------
OutputTensorInfo::OutputTensorInfo() : m_impl(std::unique_ptr<OutputTensorInfoImpl>(new OutputTensorInfoImpl())) {}
OutputTensorInfo::~OutputTensorInfo() = default;

OutputTensorInfo& OutputTensorInfo::set_element_type(const element::Type& type) {
    m_impl->set_element_type(type);
    return *this;
}

OutputTensorInfo& OutputTensorInfo::set_layout(const Layout& layout) {
    m_impl->set_layout(layout);
    return *this;
}

// --------------------- OutputModelInfo ------------------
OutputModelInfo::OutputModelInfo() : m_impl(std::unique_ptr<OutputModelInfoImpl>(new OutputModelInfoImpl())) {}
OutputModelInfo::~OutputModelInfo() = default;

OutputModelInfo& OutputModelInfo::set_layout(const Layout& layout) {
    m_impl->set_layout(layout);
    return *this;
}

// --------------------- PostProcessSteps ------------------

PostProcessSteps::PostProcessSteps() : m_impl(std::unique_ptr<PostProcessStepsImpl>(new PostProcessStepsImpl())) {}
PostProcessSteps::~PostProcessSteps() = default;

PostProcessSteps& PostProcessSteps::convert_element_type(const element::Type& type) {
    m_impl->add_convert_impl(type);
    return *this;
}

PostProcessSteps& PostProcessSteps::convert_layout(const Layout& dst_layout) {
    m_impl->add_convert_layout_impl(dst_layout);
    return *this;
}

PostProcessSteps& PostProcessSteps::convert_layout(const std::vector<uint64_t>& dims) {
    m_impl->add_convert_layout_impl(dims);
    return *this;
}

PostProcessSteps& PostProcessSteps::custom(const CustomPostprocessOp& postprocess_cb) {
    // 'true' indicates that custom postprocessing step will trigger validate_and_infer_types
    m_impl->actions().emplace_back([postprocess_cb](const Output<ov::Node>& node, PostprocessingContext&) {
        return std::make_tuple(postprocess_cb(node), true);
    });
    return *this;
}

}  // namespace preprocess
}  // namespace ov
