// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/pre_post_process/pre_post_process.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset7.hpp"

namespace ov {
namespace preprocess {

class PreProcessActionBase;

struct InputTensorInfo::InputTensorImpl {
    ov::element::Type m_type = ov::element::dynamic;
    ov::PartialLayout m_layout = {};
    ov::PartialShape m_shape = {};
};

struct PreProcessSteps::PreProcessImpl {
    std::vector<std::shared_ptr<PreProcessActionBase>> m_actions;
};

struct InputNetworkInfo::InputNetworkInfoImpl {
    ov::PartialLayout m_layout = {};
};

struct InputInfo::InputInfoImpl {
    InputInfoImpl() = default;
    bool is_default() const {
        return m_port_index == -1 && m_tensor_name.empty();
    }
    int m_port_index = -1;
    std::string m_tensor_name;
    InputTensorInfo m_tensor_data;
    PreProcessSteps m_preprocess;
    InputNetworkInfo m_network_info;
};

InputInfo::InputInfo() : m_impl(std::unique_ptr<InputInfoImpl>(new InputInfoImpl)) {}
InputInfo::InputInfo(InputInfo&&) = default;
InputInfo::~InputInfo() = default;

InputInfo& InputInfo::tensor(InputTensorInfo&& builder) & {
    m_impl->m_tensor_data = std::move(builder);
    return *this;
}

InputInfo&& InputInfo::tensor(InputTensorInfo&& builder) && {
    m_impl->m_tensor_data = std::move(builder);
    return std::move(*this);
}

InputInfo&& InputInfo::preprocess(PreProcessSteps&& builder) && {
    m_impl->m_preprocess = std::move(builder);
    return std::move(*this);
}

InputInfo& InputInfo::preprocess(PreProcessSteps&& builder) & {
    m_impl->m_preprocess = std::move(builder);
    return *this;
}

InputInfo& InputInfo::network(InputNetworkInfo&& builder) & {
    m_impl->m_network_info = std::move(builder);
    return *this;
}

InputInfo&& InputInfo::network(InputNetworkInfo&& builder) && {
    m_impl->m_network_info = std::move(builder);
    return std::move(*this);
}

const InputTensorInfo& InputInfo::get_tensor() const {
    return m_impl->m_tensor_data;
}

const PreProcessSteps& InputInfo::get_preprocess() const {
    return m_impl->m_preprocess;
}

const InputNetworkInfo& InputInfo::get_network() const {
    return m_impl->m_network_info;
}

// InPreProcess actions hierarchy

struct PreProcessActionBase {
    virtual std::shared_ptr<ngraph::Node> apply(std::shared_ptr<ngraph::Node> node, PreProcessContext& state) = 0;
};

struct PreProcessConvertElementType : public PreProcessActionBase {
    PreProcessConvertElementType(ov::element::Type type) : m_type(type) {}
    ov::element::Type m_type;
    std::shared_ptr<ngraph::Node> apply(std::shared_ptr<ngraph::Node> node, PreProcessContext& state) override {
        if (node->get_element_type().is_dynamic()) {
            throw ngraph::ngraph_error("Can't insert 'convert_element_type' for dynamic source tensor type."
                                       " Please specify source type using tensor().set_element_type(...)");
        }
        auto convert = std::make_shared<ngraph::op::v0::Convert>(node, m_type);
        convert->set_friendly_name(node->get_friendly_name() + "/convert_element_type");
        state.set_element_type(m_type);
        return convert;
    }
};

struct PreProcessScale : public PreProcessActionBase {
    PreProcessScale(const std::vector<float>& values) : m_values(values) {}
    std::vector<float> m_values;
    std::shared_ptr<ngraph::Node> apply(std::shared_ptr<ngraph::Node> node, PreProcessContext& state) override {
        ngraph::Shape shape;
        if (m_values.size() == 1) {
            shape = ngraph::Shape{1};
        } else {
            NGRAPH_CHECK(layouts::has_channels(state.get_layout()), "Can't apply scale for unknown channels layout");
            auto channels_idx = layouts::channels(state.get_layout());
            std::vector<std::size_t> v(state.get_layout().size(), 1);
            v[channels_idx] = m_values.size();
            shape = ngraph::Shape(v);
        }
        auto constant = ngraph::op::v0::Constant::create(ngraph::element::f32, shape, m_values);
        constant->set_friendly_name(node->get_friendly_name() + "/scale/Divide_Factor");

        auto new_op = std::make_shared<ngraph::op::v1::Divide>(node, constant);
        new_op->set_friendly_name(node->get_friendly_name() + "/scale/Divide");
        return new_op;
    }
};

struct PreProcessMean : public PreProcessActionBase {
    PreProcessMean(const std::vector<float>& values) : m_values(values) {}
    std::vector<float> m_values;
    std::shared_ptr<ngraph::Node> apply(std::shared_ptr<ngraph::Node> node, PreProcessContext& state) override {
        ngraph::Shape shape;
        if (m_values.size() == 1) {
            shape = ngraph::Shape{1};
        } else {
            NGRAPH_CHECK(layouts::has_channels(state.get_layout()), "Can't apply mean for unknown channels layout");
            auto channels_idx = layouts::channels(state.get_layout());
            std::vector<std::size_t> v(state.get_layout().size(), 1);
            v[channels_idx] = m_values.size();
            shape = ngraph::Shape(v);
        }
        auto constant = ngraph::op::v0::Constant::create(ngraph::element::f32, shape, m_values);
        constant->set_friendly_name(node->get_friendly_name() + "/mean/Mean_Const");

        auto new_op = std::make_shared<ngraph::op::v1::Subtract>(node, constant);
        new_op->set_friendly_name(node->get_friendly_name() + "/mean/Subtract");
        return new_op;
    }
};

class PrePostProcessor::PrePostProcessorImpl {
public:
    PrePostProcessorImpl() = default;
    std::vector<InputInfo> in_contexts;
    InputInfo& last_in() {
        return in_contexts.back();
    }
};

PrePostProcessor::PrePostProcessor() : m_impl(std::unique_ptr<PrePostProcessorImpl>(new PrePostProcessorImpl())) {}
PrePostProcessor::~PrePostProcessor() = default;

PrePostProcessor& PrePostProcessor::in(InputInfo&& builder) & {
    m_impl->in_contexts.push_back(std::move(builder));
    return *this;
}

PrePostProcessor&& PrePostProcessor::in(InputInfo&& builder) && {
    m_impl->in_contexts.push_back(std::move(builder));
    return std::move(*this);
}

size_t PrePostProcessor::inputs_size() const {
    return m_impl->in_contexts.size();
}

const InputInfo& PrePostProcessor::get_inputs_data(size_t index) const {
    return m_impl->in_contexts.at(index);
}

std::shared_ptr<Function> PrePostProcessor::build(const std::shared_ptr<Function>& function) {
    // TODO: maybe wrap this logic with 'FunctionPass'???
    for (const auto& in_context : m_impl->in_contexts) {
        std::shared_ptr<ngraph::op::Parameter> param;
        auto& input = in_context.m_impl;
        if (input->is_default()) {
            param = function->get_parameters().front();
        } else if (input->m_port_index >= 0) {
            param = function->get_parameters().at(input->m_port_index);
        } else {
            // TODO: find parameter by input tensor name (new API)
        }
        auto consumers = param->output(0).get_target_inputs();
        // TODO: use back propagation to identify preprocessing input shape/type/layout
        if (input->m_tensor_data.get_element_type() == ov::element::dynamic) {
            input->m_tensor_data.set_element_type(param->get_element_type());
        }
        auto new_param_layout = PartialLayout();
        if (input->m_tensor_data.get_layout().is_empty()) {
            input->m_tensor_data.set_layout(input->m_network_info.get_layout());
        }
        auto new_param_shape = param->get_partial_shape();
        auto new_param =
            std::make_shared<ngraph::op::v0::Parameter>(input->m_tensor_data.get_element_type(), new_param_shape);
        //            new_param->get_output_tensor(0).set_names()
        // Old param will be removed, so friendly name can be reused
        new_param->set_friendly_name(param->get_friendly_name());
        std::shared_ptr<ngraph::Node> node = new_param;

        // 2. Apply preprocessing
        PreProcessContext state;
        state.set_element_type(input->m_tensor_data.get_element_type());
        state.set_layout(input->m_tensor_data.get_layout());

        for (auto action : input->m_preprocess.m_impl->m_actions) {
            node = action->apply(node, state);
        }

        // 3. Apply 'network()' data
        // Check final type
        if (state.get_element_type() != param->get_element_type()) {
            throw ngraph::ngraph_error(
                std::string("Element type after preprocessing {") + state.get_element_type().c_type_string() +
                std::string("} doesn't match with network element type {") + param->get_element_type().c_type_string() +
                "}. Please add 'convert_element_type' explicitly");
        }
        // TODO: add layout/shape conversion if needed

        // TODO: use passes to replace selected parameter
        // Replace parameter
        for (auto consumer : consumers) {
            consumer.replace_source_output(node);
        }
        function->replace_parameter(0, new_param);
    }
    function->validate_nodes_and_infer_types();
    return function;
}

InputTensorInfo::InputTensorInfo() : m_impl(std::unique_ptr<InputTensorImpl>(new InputTensorImpl())) {}
InputTensorInfo::InputTensorInfo(InputTensorInfo&&) = default;
InputTensorInfo& InputTensorInfo::operator=(InputTensorInfo&&) = default;
InputTensorInfo::~InputTensorInfo() = default;

InputTensorInfo& InputTensorInfo::set_element_type(ov::element::Type type) & {
    m_impl->m_type = type;
    return *this;
}

InputTensorInfo&& InputTensorInfo::set_element_type(ov::element::Type type) && {
    m_impl->m_type = type;
    return std::move(*this);
}

InputTensorInfo& InputTensorInfo::set_layout(const PartialLayout& layout) & {
    m_impl->m_layout = layout;
    return *this;
}

InputTensorInfo&& InputTensorInfo::set_layout(const PartialLayout& layout) && {
    m_impl->m_layout = layout;
    return std::move(*this);
}

ov::element::Type InputTensorInfo::get_element_type() const { return m_impl->m_type; }
const PartialLayout& InputTensorInfo::get_layout() const { return m_impl->m_layout; }

//------------------------------------

PreProcessSteps::PreProcessSteps() : m_impl(std::unique_ptr<PreProcessImpl>(new PreProcessImpl())) {}
PreProcessSteps::PreProcessSteps(PreProcessSteps&&) = default;
PreProcessSteps& PreProcessSteps::operator=(PreProcessSteps&&) = default;
PreProcessSteps::~PreProcessSteps() = default;

PreProcessSteps& PreProcessSteps::scale(float value) & {
    m_impl->m_actions.push_back(std::make_shared<PreProcessScale>(std::vector<float>{value}));
    return *this;
}

PreProcessSteps&& PreProcessSteps::scale(float value) && {
    m_impl->m_actions.push_back(std::make_shared<PreProcessScale>(std::vector<float>{value}));
    return std::move(*this);
}

PreProcessSteps& PreProcessSteps::scale(const std::vector<float>& values) & {
    m_impl->m_actions.push_back(std::make_shared<PreProcessScale>(values));
    return *this;
}

PreProcessSteps&& PreProcessSteps::scale(const std::vector<float>& values) && {
    m_impl->m_actions.push_back(std::make_shared<PreProcessScale>(values));
    return std::move(*this);
}

PreProcessSteps&& PreProcessSteps::mean(float value) && {
    m_impl->m_actions.push_back(std::make_shared<PreProcessMean>(std::vector<float>{value}));
    return std::move(*this);
}

PreProcessSteps& PreProcessSteps::mean(const std::vector<float>& values) & {
    m_impl->m_actions.push_back(std::make_shared<PreProcessMean>(values));
    return *this;
}

PreProcessSteps&& PreProcessSteps::mean(const std::vector<float>& values) && {
    m_impl->m_actions.push_back(std::make_shared<PreProcessMean>(values));
    return std::move(*this);
}

PreProcessSteps& PreProcessSteps::convert_element_type(ov::element::Type type) & {
    m_impl->m_actions.push_back(std::make_shared<PreProcessConvertElementType>(type));
    return *this;
}

PreProcessSteps&& PreProcessSteps::convert_element_type(ov::element::Type type) && {
    m_impl->m_actions.push_back(std::make_shared<PreProcessConvertElementType>(type));
    return std::move(*this);
}

InputNetworkInfo::InputNetworkInfo() : m_impl(std::unique_ptr<InputNetworkInfoImpl>(new InputNetworkInfoImpl())) {}
InputNetworkInfo::InputNetworkInfo(InputNetworkInfo&&) = default;
InputNetworkInfo& InputNetworkInfo::operator=(InputNetworkInfo&&) = default;
InputNetworkInfo::~InputNetworkInfo() = default;

InputNetworkInfo& InputNetworkInfo::set_layout(const PartialLayout& layout) & {
    m_impl->m_layout = layout;
    return *this;
}

InputNetworkInfo&& InputNetworkInfo::set_layout(const PartialLayout& layout) && {
    m_impl->m_layout = layout;
    return std::move(*this);
}

const PartialLayout& InputNetworkInfo::get_layout() const { return m_impl->m_layout; }

}  // namespace preprocess
}  // namespace ov
