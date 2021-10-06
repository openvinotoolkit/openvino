// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/preprocess/pre_post_process.hpp"

#include "ngraph/opsets/opset1.hpp"
#include "openvino/core/function.hpp"
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
        m_spatial_shape_set = true;
        m_spatial_width = -1;
        m_spatial_height = -1;
    }

    void set_spatial_static_shape(size_t height, size_t width) & {
        m_spatial_shape_set = true;
        m_spatial_height = static_cast<int>(height);
        m_spatial_width = static_cast<int>(width);
    }

private:
    int m_spatial_width = -1;
    int m_spatial_height = -1;
    bool m_spatial_shape_set = false;
};

class OutputTensorInfo::OutputTensorInfoImpl : public TensorInfoImplBase {};

/// \brief InputNetworkInfoImpl - internal data structure
class NetworkInfoImpl {
public:
    NetworkInfoImpl() = default;

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

class InputNetworkInfo::InputNetworkInfoImpl : public NetworkInfoImpl {};

class OutputNetworkInfo::OutputNetworkInfoImpl : public NetworkInfoImpl {};

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

/// \brief OutputInfoImpl - internal data structure
struct OutputInfo::OutputInfoImpl {
    OutputInfoImpl() = default;
    explicit OutputInfoImpl(size_t idx) : m_has_index(true), m_index(idx) {}
    explicit OutputInfoImpl(const std::string& name) : m_has_name(true), m_name(name) {}

    bool has_index() const {
        return m_has_index;
    }

    bool has_name() const {
        return m_has_name;
    }

    void create_tensor_data() {
        m_tensor_data =
            std::unique_ptr<OutputTensorInfo::OutputTensorInfoImpl>(new OutputTensorInfo::OutputTensorInfoImpl());
    }

    bool m_has_index = false;
    size_t m_index = 0;
    bool m_has_name = false;
    std::string m_name;
    std::unique_ptr<OutputTensorInfo::OutputTensorInfoImpl> m_tensor_data;
    std::unique_ptr<PostProcessSteps::PostProcessStepsImpl> m_postprocess;
    std::unique_ptr<OutputNetworkInfo::OutputNetworkInfoImpl> m_network_data;
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

//-------------- OutputInfo ------------------
OutputInfo::OutputInfo() : m_impl(std::unique_ptr<OutputInfoImpl>(new OutputInfoImpl)) {}
OutputInfo::OutputInfo(size_t output_index)
    : m_impl(std::unique_ptr<OutputInfoImpl>(new OutputInfoImpl(output_index))) {}
OutputInfo::OutputInfo(const std::string& output_tensor_name)
    : m_impl(std::unique_ptr<OutputInfoImpl>(new OutputInfoImpl(output_tensor_name))) {}

OutputInfo::OutputInfo(OutputInfo&&) noexcept = default;
OutputInfo& OutputInfo::operator=(OutputInfo&&) noexcept = default;
OutputInfo::~OutputInfo() = default;

OutputInfo& OutputInfo::tensor(OutputTensorInfo&& builder) & {
    m_impl->m_tensor_data = std::move(builder.m_impl);
    return *this;
}

OutputInfo&& OutputInfo::tensor(OutputTensorInfo&& builder) && {
    m_impl->m_tensor_data = std::move(builder.m_impl);
    return std::move(*this);
}

OutputInfo&& OutputInfo::postprocess(PostProcessSteps&& builder) && {
    m_impl->m_postprocess = std::move(builder.m_impl);
    return std::move(*this);
}

OutputInfo& OutputInfo::postprocess(PostProcessSteps&& builder) & {
    m_impl->m_postprocess = std::move(builder.m_impl);
    return *this;
}

OutputInfo& OutputInfo::network(OutputNetworkInfo&& builder) & {
    m_impl->m_network_data = std::move(builder.m_impl);
    return *this;
}

OutputInfo&& OutputInfo::network(OutputNetworkInfo&& builder) && {
    m_impl->m_network_data = std::move(builder.m_impl);
    return std::move(*this);
}

// ------------------------ PrePostProcessor --------------------
struct PrePostProcessor::PrePostProcessorImpl {
public:
    std::list<std::unique_ptr<InputInfo::InputInfoImpl>> in_contexts;
    std::list<std::unique_ptr<OutputInfo::OutputInfoImpl>> out_contexts;
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

PrePostProcessor& PrePostProcessor::output(OutputInfo&& builder) & {
    m_impl->out_contexts.push_back(std::move(builder.m_impl));
    return *this;
}

PrePostProcessor&& PrePostProcessor::output(OutputInfo&& builder) && {
    m_impl->out_contexts.push_back(std::move(builder.m_impl));
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
        if (input->m_network_data && input->m_network_data->is_layout_set() && param->get_layout().empty()) {
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
        auto net_shape = param->get_partial_shape();
        auto new_param_shape = net_shape;
        if (input->m_tensor_data->is_layout_set() && param->get_layout() != Layout() &&
            param->get_layout() != input->m_tensor_data->get_layout()) {
            // Find transpose between network and tensor layouts and update tensor shape
            auto net_to_tensor =
                layout::find_permutation(param->get_layout(), net_shape.rank(), input->m_tensor_data->get_layout());
            std::vector<ov::Dimension> dims(new_param_shape.size());
            std::transform(net_to_tensor.begin(), net_to_tensor.end(), dims.begin(), [&](int64_t v) {
                return new_param_shape[v];
            });
            new_param_shape = PartialShape(dims);
        }
        if (input->m_tensor_data->is_spatial_shape_set()) {
            auto height_idx = get_and_check_height_idx(input->m_tensor_data->get_layout(), new_param_shape);
            auto width_idx = get_and_check_width_idx(input->m_tensor_data->get_layout(), new_param_shape);
            if (input->m_tensor_data->is_spatial_shape_dynamic()) {
                // Use dynamic spatial dimensions
                new_param_shape[height_idx] = Dimension::dynamic();
                new_param_shape[width_idx] = Dimension::dynamic();
            } else {
                // Use static spatial dimensions
                new_param_shape[height_idx] = input->m_tensor_data->get_spatial_height();
                new_param_shape[width_idx] = input->m_tensor_data->get_spatial_width();
            }
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
        context.target_layout() = param->get_layout();
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

    // Post processing
    for (const auto& output : m_impl->out_contexts) {
        std::shared_ptr<op::v0::Result> result;
        Output<Node> node;
        OPENVINO_ASSERT(output, "Internal error: Invalid postprocessing output, please report a problem");
        if (output->has_index()) {
            node = function->output(output->m_index);
        } else if (output->has_name()) {
            node = function->output(output->m_name);
        } else {
            node = function->output();
        }
        result = std::dynamic_pointer_cast<op::v0::Result>(node.get_node_shared_ptr());
        // Set result layout from 'network' information
        if (output->m_network_data && output->m_network_data->is_layout_set() && result->get_layout().empty()) {
            result->set_layout(output->m_network_data->get_layout());
        }
        auto parent = result->get_input_source_output(0);
        if (!output->m_tensor_data) {
            output->create_tensor_data();
        }
        PostprocessingContext context(result->get_layout());
        if (output->m_tensor_data->is_layout_set()) {
            context.target_layout() = output->m_tensor_data->get_layout();
        }
        if (output->m_tensor_data->is_element_type_set()) {
            context.target_element_type() = output->m_tensor_data->get_element_type();
        }
        // 2. Apply post-processing
        node = result->get_input_source_output(0);
        if (output->m_postprocess) {
            for (const auto& action : output->m_postprocess->actions()) {
                node = std::get<0>(action)({node}, context);
                tensor_data_updated |= std::get<1>(action);
            }
        }
        // Implicit: Convert element type + layout to user's tensor implicitly
        PostStepsList implicit_steps;
        if (node.get_element_type() != output->m_tensor_data->get_element_type() &&
            output->m_tensor_data->is_element_type_set() && node.get_element_type() != element::dynamic) {
            implicit_steps.add_convert_impl(output->m_tensor_data->get_element_type());
        }

        if (!context.target_layout().empty() && context.target_layout() != context.layout()) {
            implicit_steps.add_convert_layout_impl(context.target_layout());
        }
        for (const auto& action : implicit_steps.actions()) {
            node = std::get<0>(action)({node}, context);
            tensor_data_updated |= std::get<1>(action);
        }

        // Create result
        auto new_result = std::make_shared<ov::op::v0::Result>(node);
        if (!context.layout().empty()) {
            new_result->set_layout(context.layout());
        }
        new_result->get_input_tensor(0).set_names(result->get_input_tensor(0).get_names());
        function->add_results({new_result});
        function->remove_result(result);
    }

    // Validate nodes
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

InputTensorInfo& InputTensorInfo::set_spatial_dynamic_shape() & {
    m_impl->set_spatial_dynamic_shape();
    return *this;
}

InputTensorInfo&& InputTensorInfo::set_spatial_dynamic_shape() && {
    m_impl->set_spatial_dynamic_shape();
    return std::move(*this);
}

InputTensorInfo& InputTensorInfo::set_spatial_static_shape(size_t height, size_t width) & {
    m_impl->set_spatial_static_shape(height, width);
    return *this;
}

InputTensorInfo&& InputTensorInfo::set_spatial_static_shape(size_t height, size_t width) && {
    m_impl->set_spatial_static_shape(height, width);
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
    OPENVINO_ASSERT(dst_height <= std::numeric_limits<int>::max() && dst_width <= std::numeric_limits<int>::max(),
                    "Resize: Width/Height dimensions cannot be greater than ",
                    std::to_string(std::numeric_limits<int>::max()));
    m_impl->add_resize_impl(alg, static_cast<int>(dst_height), static_cast<int>(dst_width));
    return *this;
}

PreProcessSteps&& PreProcessSteps::resize(ResizeAlgorithm alg, size_t dst_height, size_t dst_width) && {
    OPENVINO_ASSERT(dst_height <= std::numeric_limits<int>::max() && dst_width <= std::numeric_limits<int>::max(),
                    "Resize: Width/Height dimensions cannot be greater than ",
                    std::to_string(std::numeric_limits<int>::max()));
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

PreProcessSteps& PreProcessSteps::convert_layout(const Layout& dst_layout) & {
    m_impl->add_convert_layout_impl(dst_layout);
    return *this;
}

PreProcessSteps&& PreProcessSteps::convert_layout(const Layout& dst_layout) && {
    m_impl->add_convert_layout_impl(dst_layout);
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

// --------------------- OutputTensorInfo ------------------
OutputTensorInfo::OutputTensorInfo() : m_impl(std::unique_ptr<OutputTensorInfoImpl>(new OutputTensorInfoImpl())) {}
OutputTensorInfo::OutputTensorInfo(OutputTensorInfo&&) noexcept = default;
OutputTensorInfo& OutputTensorInfo::operator=(OutputTensorInfo&&) noexcept = default;
OutputTensorInfo::~OutputTensorInfo() = default;

OutputTensorInfo& OutputTensorInfo::set_element_type(const element::Type& type) & {
    m_impl->set_element_type(type);
    return *this;
}

OutputTensorInfo&& OutputTensorInfo::set_element_type(const element::Type& type) && {
    m_impl->set_element_type(type);
    return std::move(*this);
}

OutputTensorInfo& OutputTensorInfo::set_layout(const Layout& layout) & {
    m_impl->set_layout(layout);
    return *this;
}

OutputTensorInfo&& OutputTensorInfo::set_layout(const Layout& layout) && {
    m_impl->set_layout(layout);
    return std::move(*this);
}

// --------------------- OutputNetworkInfo ------------------
OutputNetworkInfo::OutputNetworkInfo() : m_impl(std::unique_ptr<OutputNetworkInfoImpl>(new OutputNetworkInfoImpl())) {}
OutputNetworkInfo::OutputNetworkInfo(OutputNetworkInfo&&) noexcept = default;
OutputNetworkInfo& OutputNetworkInfo::operator=(OutputNetworkInfo&&) noexcept = default;
OutputNetworkInfo::~OutputNetworkInfo() = default;

OutputNetworkInfo& OutputNetworkInfo::set_layout(const Layout& layout) & {
    m_impl->set_layout(layout);
    return *this;
}

OutputNetworkInfo&& OutputNetworkInfo::set_layout(const Layout& layout) && {
    m_impl->set_layout(layout);
    return std::move(*this);
}

// --------------------- PostProcessSteps ------------------

PostProcessSteps::PostProcessSteps() : m_impl(std::unique_ptr<PostProcessStepsImpl>(new PostProcessStepsImpl())) {}
PostProcessSteps::PostProcessSteps(PostProcessSteps&&) noexcept = default;
PostProcessSteps& PostProcessSteps::operator=(PostProcessSteps&&) noexcept = default;
PostProcessSteps::~PostProcessSteps() = default;

PostProcessSteps& PostProcessSteps::convert_element_type(const element::Type& type) & {
    m_impl->add_convert_impl(type);
    return *this;
}

PostProcessSteps&& PostProcessSteps::convert_element_type(const element::Type& type) && {
    m_impl->add_convert_impl(type);
    return std::move(*this);
}

PostProcessSteps& PostProcessSteps::convert_layout(const Layout& dst_layout) & {
    m_impl->add_convert_layout_impl(dst_layout);
    return *this;
}

PostProcessSteps&& PostProcessSteps::convert_layout(const Layout& dst_layout) && {
    m_impl->add_convert_layout_impl(dst_layout);
    return std::move(*this);
}

PostProcessSteps& PostProcessSteps::custom(const CustomPostprocessOp& postprocess_cb) & {
    // 'true' indicates that custom postprocessing step will trigger validate_and_infer_types
    m_impl->actions().emplace_back(std::make_tuple(
        [postprocess_cb](const Output<ov::Node>& node, PostprocessingContext&) {
            return postprocess_cb(node);
        },
        true));
    return *this;
}

PostProcessSteps&& PostProcessSteps::custom(const CustomPostprocessOp& postprocess_cb) && {
    // 'true' indicates that custom postprocessing step will trigger validate_and_infer_types
    m_impl->actions().emplace_back(std::make_tuple(
        [postprocess_cb](const Output<ov::Node>& node, PostprocessingContext&) {
            return postprocess_cb(node);
        },
        true));
    return std::move(*this);
}

}  // namespace preprocess
}  // namespace ov
