// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/result.hpp"

#include <memory>
#include <typeindex>
#include <typeinfo>

#include "itt.hpp"
#include "openvino/core/descriptor_tensor.hpp"

namespace ov {
namespace descriptor {

/**
 * @brief Dedicated tensor descriptor implementation to share input descriptor.
 */
class SharedTensor : public ITensorDescriptor {
public:
    SharedTensor(std::shared_ptr<ITensorDescriptor> tensor) : m_shared_tensor{std::move(tensor)}, m_output_names{} {
        OPENVINO_ASSERT(m_shared_tensor, "Cannot set NULL tensor descriptor");
    }

    /**
     * @brief Update output tensor descriptor with shared tensors from new input.
     *
     * @param output     Output descriptor to be updated with shared tensor.
     * @param new_input  New input with tensor description.
     */
    static void update(Output& output, const Input& new_input) {
        auto& output_descriptor = TensorExtension::get_descriptor_ptr(output.get_tensor());
        const auto& input_descriptor = TensorExtension::get_descriptor_ptr(new_input.get_output().get_tensor());
        if (auto* result_ptr = dynamic_cast<SharedTensor*>(output_descriptor.get())) {
            result_ptr->set_tensor(input_descriptor);
        } else {
            output_descriptor = std::make_shared<SharedTensor>(input_descriptor);
        }
    }

    // --- ITensorDescriptor API
    virtual const element::Type& get_element_type() const override {
        return m_shared_tensor->get_element_type();
    }

    virtual const PartialShape& get_partial_shape() const override {
        return m_shared_tensor->get_partial_shape();
    }

    virtual const Shape& get_shape() const override {
        return m_shared_tensor->get_shape();
    }

    virtual void set_type_shape(const element::Type& et, const PartialShape& shape) override {
        m_shared_tensor->set_type_shape(et, shape);
    }

    void set_names(const std::unordered_set<std::string>& names) override {
        rm_tensor_output_names();
        m_output_names = names;
        m_shared_tensor->add_names(m_output_names);
    }

    void add_names(const std::unordered_set<std::string>& names) override {
        m_output_names.insert(names.begin(), names.end());
        m_shared_tensor->add_names(names);
    }

    const std::unordered_set<std::string>& get_names() const override {
        return m_shared_tensor->get_names();
    }

    const std::string& get_any_name() const override {
        return m_shared_tensor->get_any_name();
    }

    /** @brief Gets runtime map from shared tensor. */
    const RTMap& rt_map() const override {
        return m_shared_tensor->rt_map();
    }

    RTMap& rt_map() override {
        return m_shared_tensor->rt_map();
    }

    size_t pointer_hash() const noexcept override {
        return m_shared_tensor->pointer_hash();
    }

    OPENVINO_SUPPRESS_DEPRECATED_START
    std::string& legacy_name() override {
        return m_shared_tensor->legacy_name();
    }

    const std::string& legacy_name() const override {
        return m_shared_tensor->legacy_name();
    }
    OPENVINO_SUPPRESS_DEPRECATED_END

private:
    void set_tensor(std::shared_ptr<ITensorDescriptor> tensor) {
        OPENVINO_ASSERT(tensor, "Cannot set NULL tensor descriptor");
        rm_tensor_output_names();
        auto prev_rt_map = rt_map();

        m_shared_tensor = std::move(tensor);
        m_shared_tensor->add_names(m_output_names);
        rt_map().insert(std::make_move_iterator(prev_rt_map.begin()), std::make_move_iterator(prev_rt_map.end()));
    }

    void rm_tensor_output_names() {
        auto names = m_shared_tensor->get_names();
        for (const auto& output_name : m_output_names) {
            names.erase(output_name);
        }

        m_shared_tensor->set_names(names);
    }

    std::shared_ptr<ITensorDescriptor> m_shared_tensor;
    std::unordered_set<std::string> m_output_names;
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

    // Result shares input tensor but can have specific properties which are added/removed to input.
    descriptor::SharedTensor::update(get_output_descriptor(0), get_input_descriptor(0));
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
            m_ref[i] = as_type_ptr<op::v0::Result>(visitor.get_registered_node(std::move(id)));
        }
    }
    return true;
}
}  // namespace ov
