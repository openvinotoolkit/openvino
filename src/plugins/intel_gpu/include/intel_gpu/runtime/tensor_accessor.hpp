// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/tensor.hpp"
#include "tensor_data_accessor.hpp"

#include "memory.hpp"
#include "layout.hpp"

namespace cldnn {

inline ov::Tensor make_tensor(const layout& l, void* memory_pointer) {
    ov::element::Type et = data_type_to_element_type(l.data_type);

    return ov::Tensor(et, l.get_shape(), memory_pointer);
}

struct TensorsContainer final {
    using MemoryMap = std::unordered_map<size_t, cldnn::memory::ptr>;
    using TensorsMap = std::unordered_map<size_t, ov::Tensor>;

    explicit TensorsContainer(const cldnn::stream* stream) : m_stream(stream) { }

    ~TensorsContainer() {
        for (auto& mem : m_memories) {
            mem.second->unlock(*m_stream);
        }
    }

    void emplace(size_t port, cldnn::memory::ptr mem) {
        m_memories.emplace(port, mem);
        auto ptr = mem->lock(*m_stream, cldnn::mem_lock_type::read);
        m_tensors.emplace(port, make_tensor(mem->get_layout(), ptr));
    }

    void emplace(size_t port, const ov::Tensor& tensor) {
        m_tensors.emplace(port, tensor);
    }

    template<typename ElementType>
    void emplace(size_t port, std::vector<ElementType>& vector, data_types dt = data_types::i64) {
        ov::Shape shape{vector.size()};
        auto tensor = make_tensor({shape, data_types::i64, format::bfyx}, static_cast<void*>(vector.data()));
        m_tensors.emplace(port, tensor);
    }

    size_t size() const { return m_tensors.size(); }
    ov::Tensor operator[](std::size_t port) const {
        if (!m_tensors.count(port)) {
            return ov::Tensor();
        }
        return m_tensors.at(port);
    }

private:
    const cldnn::stream* m_stream;
    MemoryMap m_memories;
    TensorsMap m_tensors;
};

class TensorAccessor final : public ov::ITensorAccessor {
public:
    using MemoryMap = std::unordered_map<size_t, cldnn::memory::ptr>;
    using TensorsMap = std::unordered_map<size_t, ov::Tensor>;
    explicit TensorAccessor(const TensorsContainer& container) : m_container(container) { }

    ov::Tensor operator()(size_t port) const override {
        return m_container[port];
    }

private:
    const TensorsContainer& m_container;
};

inline cldnn::TensorAccessor make_tensor_accessor(const TensorsContainer& c) {
    return cldnn::TensorAccessor(c);
}

}  // namespace cldnn
