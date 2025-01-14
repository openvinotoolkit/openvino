// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/descriptor/output.hpp"
#include "openvino/core/descriptor_tensor.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace descriptor {
/**
 * @brief Dedicated tensor descriptor implementation to share input descriptor.
 *
 * Shared tensor share input tensor but have specific properties:
 * - tensor names - if set these are used as descriptor names and appended to input tensor because is same tensor
 */
class SharedTensor : public ITensorDescriptor {
public:
    SharedTensor(std::shared_ptr<ITensorDescriptor> tensor)
        : m_shared_tensor{std::move(tensor)},
          m_output_names{},
          m_name_it{} {
        OPENVINO_ASSERT(m_shared_tensor, "Cannot set NULL tensor descriptor");
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
        m_name_it = std::min_element(m_output_names.begin(), m_output_names.end());
        m_shared_tensor->add_names(m_output_names);
    }

    void add_names(const std::unordered_set<std::string>& names) override {
        m_output_names.insert(names.begin(), names.end());
        m_name_it = std::min_element(m_output_names.begin(), m_output_names.end());
        m_shared_tensor->add_names(names);
    }

    const std::unordered_set<std::string>& get_names() const override {
        return m_output_names.empty() ? m_shared_tensor->get_names() : m_output_names;
    }

    const std::unordered_set<std::string>& get_all_names() const override {
        return m_shared_tensor->get_names();
    }

    const std::string& get_any_name() const override {
        return m_output_names.empty() ? m_shared_tensor->get_any_name() : *m_name_it;
    }

    const RTMap& rt_map() const override {
        return m_shared_tensor->rt_map();
    }

    RTMap& rt_map() override {
        return m_shared_tensor->rt_map();
    }

    size_t pointer_hash() const noexcept override {
        return m_shared_tensor->pointer_hash();
    }

    // --- SharedTensor specific interface
    void set_tensor(std::shared_ptr<ITensorDescriptor> tensor) {
        if (tensor != m_shared_tensor) {
            OPENVINO_ASSERT(tensor, "Cannot set NULL tensor descriptor");
            rm_tensor_output_names();
            auto prev_rt_map = rt_map();

            m_shared_tensor = std::move(tensor);
            m_shared_tensor->add_names(m_output_names);
            rt_map().insert(std::make_move_iterator(prev_rt_map.begin()), std::make_move_iterator(prev_rt_map.end()));
        }
    }

private:
    void rm_tensor_output_names() {
        auto names = m_shared_tensor->get_names();
        for (const auto& output_name : m_output_names) {
            names.erase(output_name);
        }

        m_shared_tensor->set_names(names);
    }

    std::shared_ptr<ITensorDescriptor> m_shared_tensor;
    std::unordered_set<std::string> m_output_names;
    std::unordered_set<std::string>::const_iterator m_name_it;
};

/**
 * @brief Set output tensor descriptor with shared tensor from new input.
 *
 * @param output  Output descriptor to be updated.
 * @param input   Input descriptor to set as shared tensor.
 */
void set_shared_tensor(Output& output, const Input& input) {
    auto& output_descriptor = TensorExtension::get_descriptor_ptr(output.get_tensor());
    const auto& input_descriptor = TensorExtension::get_descriptor_ptr(input.get_output().get_tensor());
    if (auto* result_ptr = dynamic_cast<SharedTensor*>(output_descriptor.get())) {
        result_ptr->set_tensor(input_descriptor);
    } else {
        output_descriptor = std::make_shared<SharedTensor>(input_descriptor);
    }
}

}  // namespace descriptor
}  // namespace ov
