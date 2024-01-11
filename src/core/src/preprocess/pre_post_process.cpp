// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/preprocess/pre_post_process.hpp"

#include "color_utils.hpp"
#include "function_guard.hpp"
#include "layout_utils.hpp"
#include "openvino/core/model.hpp"
#include "preprocess_impls.hpp"

namespace ov {
namespace preprocess {

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
        OPENVINO_ASSERT(f, "Model can't be nullptr for PrePostProcessor");
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
        OPENVINO_ASSERT(false, "Model doesn't have input with name ", tensor_name);
    }

    OutputInfo& find_output(const std::string& tensor_name) {
        size_t index;
        for (index = 0; index < m_function->outputs().size(); index++) {
            if (m_function->output(index).get_tensor().get_names().count(tensor_name)) {
                return m_outputs[index];
            }
        }
        OPENVINO_ASSERT(false, "Model doesn't have output with name ", tensor_name);
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
                    "PrePostProcessor::input() - Model must have exactly one input, got ",
                    m_impl->m_inputs.size());
    return m_impl->m_inputs.front();
}

InputInfo& PrePostProcessor::input(size_t input_index) {
    OPENVINO_ASSERT(m_impl->m_inputs.size() > input_index,
                    "PrePostProcessor::input(size_t) - Model doesn't have input with index ",
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
                    "PrePostProcessor::output() - Model must have exactly one output, got ",
                    m_impl->m_outputs.size());
    return m_impl->m_outputs.front();
}

OutputInfo& PrePostProcessor::output(size_t output_index) {
    OPENVINO_ASSERT(m_impl->m_outputs.size() > output_index,
                    "PrePostProcessor::output(size_t) - Model doesn't have input with index ",
                    output_index,
                    ". Total number of inputs is ",
                    m_impl->m_inputs.size());
    return m_impl->m_outputs[output_index];
}

OutputInfo& PrePostProcessor::output(const std::string& tensor_name) {
    return m_impl->find_output(tensor_name);
}

std::ostream& operator<<(std::ostream& str, const PrePostProcessor& prePostProcessor) {
    try {
        prePostProcessor.dump(str);
    } catch (ov::AssertFailure& ex) {
        str << std::endl << "Error occurred: " << ex.what();
    }
    return str;
}

void PrePostProcessor::dump(std::ostream& str) const {
    auto model = m_impl->m_function;
    std::tuple<std::unordered_set<std::string>, bool> existing_names{std::unordered_set<std::string>{}, false};
    for (const auto& input_info : m_impl->m_inputs) {
        input_info.m_impl->dump(str, model, existing_names);
    }
    for (const auto& output_info : m_impl->m_outputs) {
        output_info.m_impl->dump(str);
    }
}

std::shared_ptr<Model> PrePostProcessor::build() {
    auto function = m_impl->m_function;
    std::tuple<std::unordered_set<std::string>, bool> existing_names{std::unordered_set<std::string>{}, false};
    FunctionGuard guard(function);
    bool need_validate = false;
    auto results = function->get_results();
    auto parameters_list = std::list<std::shared_ptr<opset8::Parameter>>(function->get_parameters().begin(),
                                                                         function->get_parameters().end());

    for (const auto& input_info : m_impl->m_inputs) {
        if (input_info.m_impl->build(function, existing_names, parameters_list)) {
            need_validate = true;
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
    if (need_validate) {
        function->validate_nodes_and_infer_types();
    }

    // Post processing
    for (const auto& output_info : m_impl->m_outputs) {
        output_info.m_impl->build(results);
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

InputTensorInfo& InputTensorInfo::set_from(const ov::Tensor& runtime_tensor) {
    m_impl->set_from(runtime_tensor);
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
    OPENVINO_ASSERT(dst_height <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
                        dst_width <= static_cast<size_t>(std::numeric_limits<int>::max()),
                    "Resize: Width/Height dimensions cannot be greater than ",
                    std::to_string(std::numeric_limits<int>::max()));
    m_impl->add_resize_impl(alg, static_cast<int>(dst_height), static_cast<int>(dst_width));
    return *this;
}

PreProcessSteps& PreProcessSteps::resize(ResizeAlgorithm alg) {
    m_impl->add_resize_impl(alg, -1, -1);
    return *this;
}

PreProcessSteps& PreProcessSteps::crop(const std::vector<int>& begin, const std::vector<int>& end) {
    m_impl->add_crop_impl(begin, end);
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
    m_impl->actions().emplace_back(
        [preprocess_cb](const std::vector<Output<Node>>& nodes,
                        const std::shared_ptr<ov::Model>&,
                        PreprocessingContext&) {
            OPENVINO_ASSERT(nodes.size() == 1,
                            "Can't apply custom preprocessing step for multi-plane input. Suggesting to convert "
                            "current image to RGB/BGR color format using 'convert_color'");
            return std::make_tuple(std::vector<Output<Node>>{preprocess_cb(nodes[0])}, true);
        },
        "custom");
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

OutputModelInfo& OutputModelInfo::set_color_format(const ov::preprocess::ColorFormat& format,
                                                   const std::vector<std::string>& sub_names) {
    m_impl->set_color_format(format);
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

PostProcessSteps& PostProcessSteps::convert_color(const ov::preprocess::ColorFormat& dst_format) {
    m_impl->add_convert_color_impl(dst_format);
    return *this;
}

PostProcessSteps& PostProcessSteps::custom(const CustomPostprocessOp& postprocess_cb) {
    // 'true' indicates that custom postprocessing step will trigger validate_and_infer_types
    m_impl->actions().emplace_back(
        [postprocess_cb](const Output<ov::Node>& node, PostprocessingContext&) {
            return std::make_tuple(postprocess_cb(node), true);
        },
        "custom");
    return *this;
}

}  // namespace preprocess
}  // namespace ov
