// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/result.hpp"

#include <memory>
#include <typeindex>
#include <typeinfo>

#include "itt.hpp"
#include "openvino/core/descriptor_tensor.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace descriptor {

/**
 * @brief
 *
 */
class ResultOutputTensor final : public Tensor {
public:
    ResultOutputTensor(std::shared_ptr<Tensor> tensor) : Tensor{}, m_input_tensor{std::move(tensor)} {
        OPENVINO_ASSERT(m_input_tensor, "Cannot link to NULL tensor");
        m_element_type = m_input_tensor->get_element_type();
        m_partial_shape = m_input_tensor->get_partial_shape();
        update_legacy_name();
    }

    void set_names(const std::unordered_set<std::string>& names) override {
        m_names = names;
        m_input_tensor->add_names(m_names);
    }

    void add_names(const std::unordered_set<std::string>& names) override {
        m_names.insert(names.begin(), names.end());
        m_input_tensor->add_names(m_names);
    }

    const std::string& get_any_name() const override {
        return m_input_tensor->get_any_name();
    };

    const std::unordered_set<std::string>& get_names() const override {
        return m_input_tensor->get_names();
    };

    const Shape& get_shape() const override {
        return m_input_tensor->get_shape();
    };

    void set_lower_value(const ov::Tensor& value) override {
        m_input_tensor->set_lower_value(value);
    };

    void set_upper_value(const ov::Tensor& value) override {
        m_input_tensor->set_upper_value(value);
    };

    void set_value_symbol(const TensorSymbol& value_symbol) override {
        m_input_tensor->set_value_symbol(value_symbol);
    };

    const ov::Tensor& get_lower_value() const override {
        return m_input_tensor->get_lower_value();
    }

    const ov::Tensor& get_upper_value() const override {
        return m_input_tensor->get_upper_value();
    }

    TensorSymbol get_value_symbol() const override {
        return m_input_tensor->get_value_symbol();
    }

    bool has_and_set_bound() const override {
        return m_input_tensor->has_and_set_bound();
    }

    size_t size() const override {
        return m_input_tensor->size();
    }

    RTMap& get_rt_info() override {
        return m_input_tensor->get_rt_info();
    }

    const RTMap& get_rt_info() const override {
        return m_input_tensor->get_rt_info();
    }

    void clone_from(const ov::descriptor::Tensor& old) override {
        m_input_tensor->clone_from(old);
    }

    void invalidate_values() override {
        m_input_tensor->invalidate_values();
    }

    void set_input_tensor(std::shared_ptr<Tensor> tensor) {
        OPENVINO_ASSERT(tensor, "Cannot link to NULL tensor");
        // restore result input names (remove output specific names)
        const auto& names = get_names();
        std::unordered_set<std::string> restored_input_names;
        std::set_difference(names.begin(),
                            names.end(),
                            m_names.begin(),
                            m_names.end(),
                            std::inserter(restored_input_names, restored_input_names.end()));
        m_input_tensor->set_names(restored_input_names);

        // copy rt info to set on new linked input tensor
        auto rt_info = get_rt_info();

        m_input_tensor = std::move(tensor);
        m_element_type = m_input_tensor->get_element_type();
        m_partial_shape = m_input_tensor->get_partial_shape();

        auto& current_rt_info = m_input_tensor->get_rt_info();
        for (auto&& item : rt_info) {
            current_rt_info[item.first] = std::move(item.second);
        }

        m_input_tensor->add_names(m_names);
        update_legacy_name();
    }

private:
    void update_legacy_name() {
        OPENVINO_SUPPRESS_DEPRECATED_START
        m_legacy_name = ov::descriptor::get_ov_tensor_legacy_name(*m_input_tensor);
        OPENVINO_SUPPRESS_DEPRECATED_END
    }

    std::shared_ptr<Tensor> m_input_tensor;
};
}  // namespace descriptor
namespace op {
namespace v0 {

Result::Result(const Output<Node>& arg) : Op({arg}) {
    constructor_validate_and_infer_types();
}

void Result::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Result_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, get_input_size() == 1, "Argument has ", get_input_size(), " outputs (1 expected).");

    // Result doesn't change change in/out tensors
    // Make output description base on input and sync required fields
    // but have possibility to have specific tensor names for input/output
    const auto& input_tensor = get_input_descriptor(0).get_tensor_ptr();
    if (auto output_tensor =
            std::dynamic_pointer_cast<descriptor::ResultOutputTensor>(get_output_descriptor(0).get_tensor_ptr())) {
        output_tensor->set_input_tensor(input_tensor);
    } else {
        const auto result_tensor = std::make_shared<descriptor::ResultOutputTensor>(input_tensor);
        get_output_descriptor(0).set_tensor_ptr(result_tensor);
    }
}

std::shared_ptr<Node> Result::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Result_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    return std::make_shared<Result>(new_args.at(0));
}

bool Result::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_Result_evaluate);
    OPENVINO_ASSERT(inputs.size() == 1);

    if (outputs.empty()) {
        outputs.emplace_back(inputs[0].get_element_type(), inputs[0].get_shape());
    } else {
        OPENVINO_ASSERT(outputs.size() == 1);
        if (!outputs[0]) {
            outputs[0] = Tensor(inputs[0].get_element_type(), inputs[0].get_shape());
        }
    }

    outputs[0].set_shape(inputs[0].get_shape());
    if (inputs[0].get_element_type() == element::string) {
        // memcpy for element::string Tensor does not work because output elements
        // will refer to input string elements but they must be separate objects in memory
        inputs[0].copy_to(outputs[0]);
    } else {
        void* output = outputs[0].data();
        const void* input = inputs[0].data();
        memcpy(output, input, outputs[0].get_byte_size());
    }

    return true;
}

bool Result::has_evaluate() const {
    OV_OP_SCOPE(v0_Result_has_evaluate);
    return true;
}

bool Result::constant_fold(OutputVector& output_values, const OutputVector& inputs_values) {
    return false;
}

ov::Layout Result::get_layout() const {
    return ov::layout::get_layout(output(0));
}

void Result::set_layout(const ov::Layout& layout) {
    ov::layout::set_layout(output(0), layout);
}
}  // namespace v0
}  // namespace op

AttributeAdapter<ResultVector>::AttributeAdapter(ResultVector& ref) : m_ref(ref) {}

bool AttributeAdapter<ResultVector>::visit_attributes(AttributeVisitor& visitor) {
    size_t size = m_ref.size();
    visitor.on_attribute("size", size);
    if (size != m_ref.size()) {
        m_ref.resize(size);
    }
    std::ostringstream index;
    for (size_t i = 0; i < size; i++) {
        index.str("");
        index << i;
        std::string id;
        if (m_ref[i]) {
            id = visitor.get_registered_node_id(m_ref[i]);
        }
        visitor.on_attribute(index.str(), id);
        if (!m_ref[i]) {
            m_ref[i] = as_type_ptr<op::v0::Result>(visitor.get_registered_node(id));
        }
    }
    return true;
}
}  // namespace ov
