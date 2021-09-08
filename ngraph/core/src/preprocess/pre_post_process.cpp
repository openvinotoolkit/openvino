// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/preprocess/pre_post_process.hpp"

#include "ngraph/opsets/opset1.hpp"
#include "openvino/core/function.hpp"

namespace ov {
namespace preprocess {

/// \brief InputTensorInfoImpl - internal data structure
struct InputTensorInfo::InputTensorInfoImpl {
    InputTensorInfoImpl() = default;
    explicit InputTensorInfoImpl(const element::Type& type) : m_type(type) {}

    element::Type m_type = element::dynamic;
};

/// \brief PreProcessStepsImpl - internal data structure
struct PreProcessSteps::PreProcessStepsImpl {
    void add_scale_impl(const std::vector<float>& values) {
        m_actions.emplace_back(std::make_tuple(
            [values](const std::shared_ptr<Node>& node) {
                ngraph::Shape shape;
                if (values.size() == 1) {
                    shape = ngraph::Shape{1};
                } else {
                    // TODO: implement when Layout API is available
                }
                auto constant = op::v0::Constant::create(element::f32, shape, values);
                constant->set_friendly_name(node->get_friendly_name() + "/scale/Divide_Factor");

                auto new_op = std::make_shared<op::v1::Divide>(node, constant);
                new_op->set_friendly_name(node->get_friendly_name() + "/scale/Divide");
                return new_op;
            },
            false));
    }

    void add_mean_impl(const std::vector<float>& values) {
        m_actions.emplace_back(std::make_tuple(
            [values](const std::shared_ptr<Node>& node) {
                ngraph::Shape shape;
                if (values.size() == 1) {
                    shape = ngraph::Shape{1};
                } else {
                    // TODO: implement when Layout API is available
                }
                auto constant = op::v0::Constant::create(element::f32, shape, values);
                constant->set_friendly_name(node->get_friendly_name() + "/mean/Mean_Const");

                auto new_op = std::make_shared<op::v1::Subtract>(node, constant);
                new_op->set_friendly_name(node->get_friendly_name() + "/mean/Subtract");
                return new_op;
            },
            false));
    }

    void add_convert_impl(const element::Type& type) {
        m_actions.emplace_back(std::make_tuple(
            [type](const std::shared_ptr<Node>& node) {
                if (node->get_element_type().is_dynamic()) {
                    throw ngraph::ngraph_error("Can't insert 'convert_element_type' for dynamic source tensor type.");
                }
                auto convert = std::make_shared<op::v0::Convert>(node, type);
                convert->set_friendly_name(node->get_friendly_name() + "/convert_element_type");
                return convert;
            },
            true));
    }
    std::list<std::tuple<PreProcessSteps::CustomPreprocessOp, bool>> m_actions;
};

/// \brief InputInfoImpl - internal data structure
struct InputInfo::InputInfoImpl {
    InputInfoImpl() = default;
    explicit InputInfoImpl(size_t idx) : m_has_index(true), m_index(idx) {}

    bool has_index() const {
        return m_has_index;
    }

    void create_tensor_data(const element::Type& type) {
        m_tensor_data =
            std::unique_ptr<InputTensorInfo::InputTensorInfoImpl>(new InputTensorInfo::InputTensorInfoImpl(type));
    }

    bool m_has_index = false;
    size_t m_index = 0;
    std::unique_ptr<InputTensorInfo::InputTensorInfoImpl> m_tensor_data;
    std::unique_ptr<PreProcessSteps::PreProcessStepsImpl> m_preprocess;
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
        OV_CHECK(input, "Internal error: Invalid preprocessing input, please report a problem");
        if (input->has_index()) {
            param = function->get_parameters().at(input->m_index);
        } else {
            // Default case
            OV_CHECK(function->get_parameters().size() == 1,
                     std::string("Preprocessing info expects having 1 input, however function has ") +
                         std::to_string(function->get_parameters().size()) +
                         " inputs. Please use ov::preprocess::InputInfo constructor specifying "
                         "particular input instead of default one");
            param = function->get_parameters().front();
        }
        auto consumers = param->output(0).get_target_inputs();
        if (!input->m_tensor_data) {
            input->create_tensor_data(param->get_element_type());
        }
        auto new_param_shape = param->get_partial_shape();
        auto new_param = std::make_shared<op::v0::Parameter>(input->m_tensor_data->m_type, new_param_shape);
        // Old param will be removed, so friendly name can be reused
        new_param->set_friendly_name(param->get_friendly_name());
        std::shared_ptr<Node> node = new_param;

        // 2. Apply preprocessing
        for (const auto& action : input->m_preprocess->m_actions) {
            node = std::get<0>(action)(node);
            tensor_data_updated |= std::get<1>(action);
        }

        // Check final type
        if (node->get_element_type() != param->get_element_type()) {
            throw ngraph::ngraph_error(
                std::string("Element type after preprocessing {") + node->get_element_type().c_type_string() +
                std::string("} doesn't match with network element type {") + param->get_element_type().c_type_string() +
                "}. Please add 'convert_element_type' explicitly");
        }

        // Replace parameter
        for (auto consumer : consumers) {
            consumer.replace_source_output(node);
        }
        if (input->has_index()) {
            function->replace_parameter(input->m_index, new_param);
        } else {
            function->replace_parameter(0, new_param);
        }
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
    m_impl->m_type = type;
    return *this;
}

InputTensorInfo&& InputTensorInfo::set_element_type(const element::Type& type) && {
    m_impl->m_type = type;
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

PreProcessSteps& PreProcessSteps::mean(float value) & {
    m_impl->add_mean_impl(std::vector<float>{value});
    return *this;
}

PreProcessSteps&& PreProcessSteps::mean(float value) && {
    m_impl->add_mean_impl(std::vector<float>{value});
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

PreProcessSteps& PreProcessSteps::custom(const CustomPreprocessOp& preprocess_cb) & {
    // 'true' indicates that custom preprocessing step will trigger validate_and_infer_types
    m_impl->m_actions.emplace_back(std::make_tuple(preprocess_cb, true));
    return *this;
}

PreProcessSteps&& PreProcessSteps::custom(const CustomPreprocessOp& preprocess_cb) && {
    // 'true' indicates that custom preprocessing step will trigger validate_and_infer_types
    m_impl->m_actions.emplace_back(std::make_tuple(preprocess_cb, true));
    return std::move(*this);
}

}  // namespace preprocess
}  // namespace ov
