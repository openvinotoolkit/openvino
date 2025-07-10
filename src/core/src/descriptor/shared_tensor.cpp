// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/descriptor/output.hpp"
#include "openvino/core/descriptor_tensor.hpp"

namespace ov::descriptor {

using ITensorDescriptorPtr = std::shared_ptr<ITensorDescriptor>;
using TensorNames = std::unordered_set<std::string>;

class SharedTensor;

/**
 * @brief Wraps existing tensor into ITensorDescriptor interface, to share tensor descriptor.
 */
class TensorWrapper : public ITensorDescriptor {
public:
    TensorWrapper(ITensorDescriptorPtr tensor) : m_tensor{std::move(tensor)} {
        OPENVINO_ASSERT(m_tensor, "Cannot set NULL tensor descriptor");
    }
    // --- ITensorDescriptor API
    const element::Type& get_element_type() const override {
        return m_tensor->get_element_type();
    }

    const PartialShape& get_partial_shape() const override {
        return m_tensor->get_partial_shape();
    }

    const Shape& get_shape() const override {
        return m_tensor->get_shape();
    }

    void set_type_shape(const element::Type& et, const PartialShape& shape) override {
        m_tensor->set_type_shape(et, shape);
    }

    void set_names(const TensorNames& names) override {
        m_tensor->set_names(names);
    }

    void add_names(const TensorNames& names) override {
        m_tensor->add_names(names);
    }

    const TensorNames& get_names() const override {
        return m_tensor->get_names();
    }

    const TensorNames& get_all_names() const override {
        return m_tensor->get_all_names();
    }

    const std::string& get_any_name() const override {
        return m_tensor->get_any_name();
    }

    const RTMap& rt_map() const override {
        return m_tensor->rt_map();
    }

    RTMap& rt_map() override {
        return m_tensor->rt_map();
    }

    size_t pointer_hash() const noexcept override {
        return m_tensor->pointer_hash();
    }

    const auto& base() const {
        return m_tensor;
    }

protected:
    void rm_names(const TensorNames& names) {
        auto tensor_names = m_tensor->get_names();
        for (const auto& name : names) {
            tensor_names.erase(name);
        }
        m_tensor->set_names(tensor_names);
    }

    ITensorDescriptorPtr m_tensor;
};

/**
 * @brief Wraps input tensor descriptor to share tensor descriptor.
 *
 * This wrapper is used for as shared tensor of Result input tensor to keep up-to-date tensor names when changed.
 * The Result's tensor keep ownerships to this tensor descriptor. When all usages of this tensor will be removed
 * then original tensor descriptor will be restored.
 */
class InputTensorWrapper : public TensorWrapper {
public:
    explicit InputTensorWrapper(ITensorDescriptorPtr descriptor, std::weak_ptr<Tensor> descriptor_owner)
        : TensorWrapper{std::move(descriptor)},
          m_descriptor_owner{std::move(descriptor_owner)},
          m_assigned_names{m_tensor->get_names()},
          m_self_observer{} {
        set_this_as_descriptor();
    }

    ~InputTensorWrapper() override {
        restore_descriptor();
    }

    // --- ITensorDescriptor API
    void set_names(const TensorNames& names) override {
        rm_names(m_assigned_names);
        m_assigned_names = names;
        TensorWrapper::add_names(names);
    }

    void add_names(const TensorNames& names) override {
        m_assigned_names.insert(names.begin(), names.end());
        TensorWrapper::add_names(names);
    }

    // --- InputTensorWrapper specific interface
    static auto make(ITensorDescriptorPtr&& descriptor, std::weak_ptr<Tensor>&& descriptor_owner) {
        auto input_tensor = std::make_shared<InputTensorWrapper>(std::move(descriptor), std::move(descriptor_owner));
        input_tensor->set_self_observer(input_tensor);
        return input_tensor;
    }

    static auto make(const InputTensorWrapper* tensor) {
        // increase shared pointer ownership counters
        return tensor->m_self_observer.lock();
    }

protected:
    void set_self_observer(std::weak_ptr<InputTensorWrapper> self_observer) {
        m_self_observer = std::move(self_observer);
    }

    const TensorNames& get_assigned_names() const {
        return m_assigned_names;
    }

private:
    void restore_descriptor() const {
        if (auto owner = m_descriptor_owner.lock()) {
            m_tensor->set_names(m_assigned_names);
            TensorExtension::get_descriptor_ptr(*owner) = m_tensor;
        }
    }

    void set_this_as_descriptor() {
        if (auto tensor = m_descriptor_owner.lock()) {
            // set this implementation as tensor descriptor
            TensorExtension::get_descriptor_ptr(*tensor).reset(this, [](auto&&) {
                // empty deleter because this object will be destroyed when all usages of this tensor will be removed
            });
        }
    }

    const std::weak_ptr<Tensor> m_descriptor_owner;  // tensor which holds this descriptor implementation
    TensorNames m_assigned_names;
    std::weak_ptr<InputTensorWrapper> m_self_observer;  // observe this class shared pointer instance
};

class ParameterTensorWrapper : public InputTensorWrapper {
public:
    explicit ParameterTensorWrapper(ITensorDescriptorPtr descriptor, std::weak_ptr<Tensor> descriptor_owner)
        : InputTensorWrapper{std::move(descriptor), std::move(descriptor_owner)} {}

    // --- ParameterTensorWrapper specific interface
    static auto make(ITensorDescriptorPtr&& descriptor, std::weak_ptr<Tensor>&& descriptor_owner) {
        auto input_tensor =
            std::make_shared<ParameterTensorWrapper>(std::move(descriptor), std::move(descriptor_owner));
        input_tensor->set_self_observer(input_tensor);
        return input_tensor;
    }
};

/**
 * @brief Dedicated tensor descriptor implementation to share input descriptor.
 *
 * Shared tensor share input tensor but have specific properties:
 * - tensor names - if set these are used as descriptor names and appended to input tensor because is same tensor
 */
class SharedTensor : public TensorWrapper {
public:
    SharedTensor(std::shared_ptr<InputTensorWrapper> tensor)
        : TensorWrapper{tensor->base()},
          m_input_descriptor{std::move(tensor)},
          m_output_names{},
          m_name_it{} {}

    // --- ITensorDescriptor API
    void set_names(const TensorNames& names) override {
        rm_names(m_output_names);
        m_output_names = names;
        m_name_it = std::min_element(m_output_names.begin(), m_output_names.end());
        TensorWrapper::add_names(m_output_names);
    }

    void add_names(const TensorNames& names) override {
        m_output_names.insert(names.begin(), names.end());
        m_name_it = std::min_element(m_output_names.begin(), m_output_names.end());
        TensorWrapper::add_names(names);
    }

    const TensorNames& get_names() const override {
        return m_output_names.empty() ? TensorWrapper::get_names() : m_output_names;
    }

    const std::string& get_any_name() const override {
        return m_output_names.empty() ? TensorWrapper::get_any_name() : *m_name_it;
    }

    // --- SharedTensor specific interface
    void set_tensor(std::shared_ptr<InputTensorWrapper> tensor) {
        m_input_descriptor = std::move(tensor);
        set_tensor(m_input_descriptor->base());
    }

    std::shared_ptr<InputTensorWrapper> get_input_descriptor() const {
        return m_input_descriptor;
    }

    const TensorNames& get_assigned_names() const {
        return m_output_names;
    }

private:
    void set_tensor(ITensorDescriptorPtr tensor) {
        if (tensor != m_tensor) {
            OPENVINO_ASSERT(tensor, "Cannot set NULL tensor descriptor");
            rm_names(m_output_names);
            auto prev_rt_map = rt_map();

            m_tensor = std::move(tensor);
            TensorWrapper::add_names(m_output_names);
            rt_map().insert(std::make_move_iterator(prev_rt_map.begin()), std::make_move_iterator(prev_rt_map.end()));
        }
    }

    std::shared_ptr<InputTensorWrapper> m_input_descriptor;
    TensorNames m_output_names;
    TensorNames::const_iterator m_name_it;
};

namespace {
/**
 * @brief Creates or returns InputTensorWrapper from Tensor's descriptor implementation.
 *
 * If given tensor descriptor is already InputTensorWrapper then return it, otherwise create new InputTensorWrapper.
 *
 * @param tensor A tensor to wrap its implementation.
 * @return A pointer to InputTensorWrapper.
 */
template <class TWrapper>
std::shared_ptr<InputTensorWrapper> make_wrapper_tensor(const std::shared_ptr<Tensor>& tensor) {
    auto org = TensorExtension::get_descriptor_ptr(*tensor);
    if (const auto input_wrapper = dynamic_cast<InputTensorWrapper*>(org.get())) {
        return InputTensorWrapper::make(input_wrapper);
    } else if (const auto shared = dynamic_cast<SharedTensor*>(org.get())) {
        return shared->get_input_descriptor();
    } else {
        return TWrapper::make(std::move(org), tensor);
    }
}

constexpr auto make_input_wrapper = make_wrapper_tensor<InputTensorWrapper>;
constexpr auto make_parameter_wrapper = make_wrapper_tensor<ParameterTensorWrapper>;

}  // namespace

/**
 * @brief Set output tensor descriptor with shared tensor from input.
 *
 * @param output  Output descriptor to be updated.
 * @param input   Input descriptor to set as shared tensor.
 */
void set_shared_tensor(Output& output, const Input& input, bool is_parameter) {
    auto& output_descriptor = TensorExtension::get_descriptor_ptr(output.get_tensor());

    if (auto* shared = dynamic_cast<SharedTensor*>(output_descriptor.get())) {
        shared->set_tensor(make_input_wrapper(input.get_output().get_tensor_ptr()));
    } else if (is_parameter) {
        output_descriptor = std::make_shared<SharedTensor>(make_parameter_wrapper(input.get_output().get_tensor_ptr()));
    } else {
        output_descriptor = std::make_shared<SharedTensor>(make_input_wrapper(input.get_output().get_tensor_ptr()));
    }
}

const std::unordered_set<std::string>& get_assigned_names(const Tensor& tensor) {
    if (auto&& descriptor = TensorExtension::get_descriptor(tensor);
        auto&& shared_tensor = dynamic_cast<const SharedTensor*>(&descriptor)) {
        return shared_tensor->get_assigned_names();
    } else {
        return descriptor.get_names();
    }
}

void add_not_parameter_names(Tensor& dst, const Tensor& src) {
    const auto& src_descriptor = TensorExtension::get_descriptor(src);
    if (dynamic_cast<const ParameterTensorWrapper*>(&src_descriptor) == nullptr) {
        dst.add_names(src_descriptor.get_names());
    }
}
}  // namespace ov::descriptor
