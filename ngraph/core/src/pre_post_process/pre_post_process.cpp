// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/pre_post_process/pre_post_process.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset7.hpp"

namespace ov {
namespace preprocess {

class PreProcessActionBase;

struct InTensorData {
    ov::element::Type m_type = ov::element::dynamic;
    ov::PartialLayout m_layout = {};
    ov::PartialShape m_shape = {};
};

struct PreProcessData {
    std::vector<std::shared_ptr<PreProcessActionBase>> m_actions;
};

struct InNetworkData {
    ov::PartialLayout m_layout = {};
};

struct InContextImpl {
    InContextImpl() = default;
    InContextImpl(int port_index) : m_port_index(port_index) {}
    InContextImpl(std::string&& tensor_name) : m_tensor_name(std::move(tensor_name)) {}
    bool is_default() const {
        return m_port_index == -1 && m_tensor_name.empty();
    }
    int m_port_index = -1;
    std::string m_tensor_name;
    InTensorData m_tensor_data;
    PreProcessData m_preprocess_data;
    InNetworkData m_network_data;
};

// InPreProcess actions hierarchy

struct PreProcessActionBase {
    virtual std::shared_ptr<ngraph::Node> apply(std::shared_ptr<ngraph::Node> node, InTensorData& state) = 0;
};

struct PreProcessConvertElementType : public PreProcessActionBase {
    PreProcessConvertElementType(ov::element::Type type) : m_type(type) {}
    ov::element::Type m_type;
    std::shared_ptr<ngraph::Node> apply(std::shared_ptr<ngraph::Node> node, InTensorData& state) override {
        auto convert = std::make_shared<ngraph::op::v0::Convert>(node, m_type);
        convert->set_friendly_name(node->get_friendly_name() + "/convert_element_type");
        state.m_type = m_type;
        return convert;
    }
};

struct PreProcessScale : public PreProcessActionBase {
    PreProcessScale(const std::vector<float>& values) : m_values(values) {}
    std::vector<float> m_values;
    std::shared_ptr<ngraph::Node> apply(std::shared_ptr<ngraph::Node> node, InTensorData& state) override {
        ngraph::Shape shape;
        if (m_values.size() == 1) {
            shape = ngraph::Shape{1};
        } else {
            NGRAPH_CHECK(layouts::has_channels(state.m_layout), "Can't apply scale for unknown channels layout");
            auto channels_idx = layouts::channels(state.m_layout);
            std::vector<std::size_t> v(state.m_layout.size(), 1);
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

class PrePostProcessorInternalData {
public:
    PrePostProcessorInternalData(const std::shared_ptr<Function>& function) : m_function(function) {}
    std::shared_ptr<Function> m_function = nullptr;
    std::vector<InContextImpl> in_contexts;
    InContextImpl& last_in() {
        return in_contexts.back();
    }
};

PrePostProcessorBase::PrePostProcessorBase(std::unique_ptr<PrePostProcessorInternalData>&& impl)
    : m_impl(std::move(impl)) {}

PrePostProcessorBase::~PrePostProcessorBase() = default;
PrePostProcessorBase::PrePostProcessorBase(PrePostProcessorBase&&) = default;

InContext PrePostProcessorBase::in() {
    m_impl->in_contexts.push_back(InContextImpl());
    return InContext(std::move(m_impl));
}

InContext PrePostProcessorBase::in(int port_index) {
    m_impl->in_contexts.push_back(InContextImpl(port_index));
    return InContext(std::move(m_impl));
}

InContext PrePostProcessorBase::in(std::string tensor_name) {
    m_impl->in_contexts.push_back(InContextImpl(std::move(tensor_name)));
    return InContext(std::move(m_impl));
}

OutProcessor PrePostProcessorBase::out() {
    return OutProcessor(std::move(m_impl));
}

std::shared_ptr<Function> PrePostProcessorBase::build() {
    // TODO: maybe wrap this logic with 'FunctionPass'???
    for (auto in_context : m_impl->in_contexts) {
        if (in_context.is_default()) {
            auto param = m_impl->m_function->get_parameters().front();
            auto consumers = param->output(0).get_target_inputs();
            // TODO: use back propagation to identify preprocessing input shape/type/layout
            auto new_param_type = param->get_element_type();
            auto new_param_shape = param->get_partial_shape();
            auto new_param_layout = PartialLayout();
            if (in_context.m_tensor_data.m_type != ov::element::dynamic) {
                new_param_type = in_context.m_tensor_data.m_type;
            } else {
                // Look for pre_process data
                for (auto action : in_context.m_preprocess_data.m_actions) {
                    if (std::dynamic_pointer_cast<PreProcessConvertElementType>(action)) {
                        throw ngraph::ngraph_error("Can't convert element type from unknown type");
                    }
                }
                in_context.m_tensor_data.m_type = param->get_element_type();
            }
            if (in_context.m_tensor_data.m_layout.is_empty()) {
                for (auto action : in_context.m_preprocess_data.m_actions) {
                    // TODO: if it is convert_layout - throw exception
                }
                if (!in_context.m_network_data.m_layout.is_empty()) {
                    in_context.m_tensor_data.m_layout = in_context.m_network_data.m_layout;
                }
            }
            auto new_param = std::make_shared<ngraph::op::v0::Parameter>(new_param_type, new_param_shape);
            // Old param will be removed, so friendly name can be reused
            new_param->set_friendly_name(param->get_friendly_name());
            std::shared_ptr<ngraph::Node> node = new_param;

            // 2. Apply preprocessing
            InTensorData state = in_context.m_tensor_data;

            for (auto action : in_context.m_preprocess_data.m_actions) {
                node = action->apply(node, state);
            }

            // 3. Apply 'network()' data
            // Add type conversion if needed
            if (state.m_type != param->get_element_type()) {
                std::cout << "Need add convert element type " << state.m_type << " to " << param->get_element_type()
                          << "\n";
                node = std::make_shared<ngraph::op::v0::Convert>(node, param->get_element_type());
            }
            // TODO: add layout/shape conversion if needed

            // TODO: use passes to replace selected parameter
            // Replace parameter
            for (auto consumer : consumers) {
                consumer.replace_source_output(node);
            }
            m_impl->m_function->replace_parameter(0, new_param);
        }
    }
    // TODO: shall we call validate_nodes_and_infer_types?
    return m_impl->m_function;
}

PrePostProcessor::PrePostProcessor(const std::shared_ptr<Function>& function)
    : PrePostProcessorBase(std::unique_ptr<PrePostProcessorInternalData>(new PrePostProcessorInternalData(function))) {}

InContext::InContext(std::unique_ptr<PrePostProcessorInternalData>&& impl) : PrePostProcessorBase(std::move(impl)) {}
InContext::InContext(InContext&&) = default;

InTensorContext InContext::tensor() {
    return InTensorContext(std::move(m_impl));
}

PreProcessContext InContext::preprocess() {
    return PreProcessContext(std::move(m_impl));
}

InNetworkContext InContext::network() {
    return InNetworkContext(std::move(m_impl));
}

InTensorContext::InTensorContext(std::unique_ptr<PrePostProcessorInternalData>&& impl)
    : PrePostProcessorBase(std::move(impl)) {}
InTensorContext::InTensorContext(InTensorContext&&) = default;

InTensorContext& InTensorContext::set_element_type(ov::element::Type type) {
    std::cout << "\nSet Element type: " << type;
    m_impl->last_in().m_tensor_data.m_type = type;
    return *this;
}

InTensorContext& InTensorContext::set_layout(const PartialLayout& layout) {
    std::cout << "\nTensor: Set layout: " << layouts::has_channels(layout);
    m_impl->last_in().m_tensor_data.m_layout = layout;
    return *this;
}

PreProcessContext InTensorContext::preprocess() {
    return PreProcessContext(std::move(m_impl));
}

InNetworkContext InTensorContext::network() {
    return InNetworkContext(std::move(m_impl));
}

PreProcessContext::PreProcessContext(std::unique_ptr<PrePostProcessorInternalData>&& impl)
    : PrePostProcessorBase(std::move(impl)) {}
PreProcessContext::PreProcessContext(PreProcessContext&&) = default;

PreProcessContext& PreProcessContext::scale(float value) {
    m_impl->last_in().m_preprocess_data.m_actions.push_back(
        std::make_shared<PreProcessScale>(std::vector<float>{value}));
    std::cout << "\nAdd scale: " << value;
    return *this;
}

PreProcessContext& PreProcessContext::scale(const std::vector<float>& values) {
    m_impl->last_in().m_preprocess_data.m_actions.push_back(std::make_shared<PreProcessScale>(values));
    std::cout << "\nAdd scale vector: " << values.size();
    return *this;
}

PreProcessContext& PreProcessContext::convert_element_type(ov::element::Type type) {
    m_impl->last_in().m_preprocess_data.m_actions.push_back(std::make_shared<PreProcessConvertElementType>(type));
    std::cout << "\nconvert_element_type: " << type;
    return *this;
}

InNetworkContext PreProcessContext::network() {
    return InNetworkContext(std::move(m_impl));
}

InNetworkContext::InNetworkContext(std::unique_ptr<PrePostProcessorInternalData>&& impl)
    : PrePostProcessorBase(std::move(impl)) {}
InNetworkContext::InNetworkContext(InNetworkContext&&) = default;

InNetworkContext& InNetworkContext::set_layout(const PartialLayout& layout) {
    if (layouts::has_channels(layout)) {
        std::cout << "Network: set layout: " << layouts::channels(layout);
    } else {
        std::cout << "Network: set layout2: " << layouts::has_channels(layout);
    }
    m_impl->last_in().m_network_data.m_layout = layout;
    return *this;
}

OutProcessor::OutProcessor(std::unique_ptr<PrePostProcessorInternalData>&& impl)
    : PrePostProcessorBase(std::move(impl)) {}
OutProcessor::OutProcessor(OutProcessor&&) = default;

}  // namespace preprocess
}  // namespace ov
