// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <unordered_map>

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/descriptor/tensor.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/aligned_buffer.hpp"

namespace ov {

/** @brief Generic atg suggests automated functionality.
 *
 * Can be use for template specialization or function overloading
 */
struct AutoTag {};
inline constexpr AutoTag AUTO{};

/** @brief Alias to map of port number and tensor names */
using TensorNamesMap = std::unordered_map<size_t, TensorNames>;
}  // namespace ov

namespace ov::util {

class BufferDescriptor final {
public:
    BufferDescriptor(const std::shared_ptr<ov::AlignedBuffer>& buffer, bool mmaped = false, size_t parent_id = 0)
        : m_buffer(buffer), m_id(generate_id()), m_parent_id(parent_id), m_mmaped(mmaped) {
            buffer->m_buffer_id = m_id;
        }

    std::shared_ptr<ov::AlignedBuffer> get_buffer() const {
        return m_buffer.lock();
    }

    size_t get_id() const {
        return m_id;
    }

    size_t get_parent_id() const {
        return m_parent_id;
    }

    bool is_mmaped() const {
        return m_mmaped;
    }
private:
    std::weak_ptr<ov::AlignedBuffer> m_buffer;
    size_t m_id;
    size_t m_parent_id;
    bool m_mmaped;
    
    static size_t generate_id() {
        static std::atomic_size_t id_counter{0};
        return ++id_counter;
    }
};

class OPENVINO_API BufferRegistry final {
public:
    static BufferRegistry& get() {
        static BufferRegistry registry;
        return registry;
    }
    size_t register_buffer(const std::shared_ptr<ov::AlignedBuffer>& buffer, bool mmaped = false) {
        auto desc = BufferDescriptor(buffer, mmaped);
        auto id = desc.get_id();
        m_registry.emplace(id, desc);
        return id;
    }

    size_t register_subbuffer(const std::shared_ptr<ov::AlignedBuffer>& buffer, size_t parent_id) {
        auto parent_desc = get_desc(parent_id);
        auto desc = BufferDescriptor(buffer, parent_desc.is_mmaped(), parent_id);
        auto id = desc.get_id();
        m_registry.emplace(id, desc);
        return id;
    }

    size_t register_subbuffer(const std::shared_ptr<ov::AlignedBuffer>& buffer, const std::shared_ptr<ov::AlignedBuffer>& parent_buffer) {
        auto parent_desc = get_desc(parent_buffer);
        return register_subbuffer(buffer, parent_desc.get_id());
    }

    BufferDescriptor get_desc(size_t id) {
        auto it = m_registry.find(id);
        if (it != m_registry.end()) {
            return it->second;
        }
        OPENVINO_THROW("Buffer with id ", id, " is not registered");
    }

    BufferDescriptor get_desc(const std::shared_ptr<ov::AlignedBuffer>& buffer) {
        if (!buffer) {
            OPENVINO_THROW("Cannot get buffer descriptor for nullptr buffer");
        }
        return get_desc(buffer->m_buffer_id);
    }

    void unregister_buffer(size_t id) {
        m_registry.erase(id);
    }

    void unregister_buffer(const std::shared_ptr<ov::AlignedBuffer>& buffer) {
        if (!buffer) {
            OPENVINO_THROW("Cannot unregister nullptr buffer");
        }
        unregister_buffer(buffer->m_buffer_id);
    }

private:
    std::unordered_map<size_t, BufferDescriptor> m_registry;
};

/** @brief Set input tensors names for the model
 *
 * Sets only tensors defined in the input map.
 * The tensors defined in map but not existing in the model will be ignored.
 *
 * @param model Model to set its input tensors names.
 * @param inputs_names Map of input tensor names.
 */
OPENVINO_API void set_input_tensors_names(Model& model, const TensorNamesMap& inputs_names);

/** @brief Set input tensors names for the model
 *
 * Sets tensors defined in the input map.
 * Tensors not defined in the map are set to default names if the tensor hasn't got names.
 * The tensors defined in the map but not existing in the model will be ignored.
 *
 * @param model Model to set its input tensors names.
 * @param inputs_names Map of input tensor names. Default empty.
 */
OPENVINO_API void set_input_tensors_names(const AutoTag&, Model& model, const TensorNamesMap& inputs_names = {});

/** @brief Set output tensors names for the model
 *
 * Sets only tensors defined in the output map.
 * The tensors defined in map but not existing in the model will be ignored.
 *
 * @param model Model to set its output tensors names.
 * @param outputs_names Map of output tensor names.AnyMap
 */
OPENVINO_API void set_output_tensor_names(Model& model, const TensorNamesMap& outputs_names);

/** @brief Set output tensors names for the model.
 *
 * Sets tensors defined in the output map.
 * Tensors not defined in the map are set to default names if the tensor hasn't got names.
 * The tensors defined in the map but not existing in the model will be ignored.
 *
 * @param model Model to set its output tensors names.
 * @param outputs_names Map of output tensor names. Default empty.
 */
OPENVINO_API void set_output_tensor_names(const AutoTag&, Model& model, const TensorNamesMap& outputs_names = {});

/** @brief Set input and output tensors names for the model
 *
 * Sets only tensors defined in the input and output maps.
 * The tensors defined in maps but not existing in the model will be ignored.
 *
 * @param model Model to set its input and output tensors names.
 * @param inputs_names Map of input tensor names.
 * @param outputs_names Map of output tensor names.
 */
OPENVINO_API void set_tensors_names(Model& model,
                                    const TensorNamesMap& inputs_names,
                                    const TensorNamesMap& outputs_names);

/** @brief Set input and output tensors names for the model.
 *
 * Sets tensors defined in the input and output maps.
 * Tensors not defined in the maps are set to default names if the tensor hasn't got names.
 * The tensors defined in the maps but not existing in the model will be ignored.
 *
 * @param model Model to set its input and output tensors names.
 * @param inputs_names Map of input tensor names. Default empty.
 * @param outputs_names Map of output tensor names. Default empty.
 */
OPENVINO_API void set_tensors_names(const AutoTag&,
                                    Model& model,
                                    const TensorNamesMap& inputs_names = {},
                                    const TensorNamesMap& outputs_names = {});

}  // namespace ov::util
