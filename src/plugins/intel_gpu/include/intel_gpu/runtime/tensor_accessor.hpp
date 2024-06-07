// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/tensor.hpp"
#include "tensor_data_accessor.hpp"

#include "memory.hpp"
#include "layout.hpp"

namespace cldnn {

inline ov::Tensor make_tensor(const layout& l, void* memory_pointer) {
    return ov::Tensor(l.data_type, l.get_shape(), memory_pointer);
}

struct TensorsContainer final {
    using MemoryMap = std::unordered_map<size_t, cldnn::memory::ptr>;
    using TensorsMap = std::unordered_map<size_t, ov::Tensor>;

    TensorsContainer(const cldnn::stream* stream, const std::map<size_t, cldnn::memory::ptr>& deps_map = {})
        : m_stream(stream)
        , m_memories(deps_map.begin(), deps_map.end()) { }

    ~TensorsContainer() {
        for (auto& port : m_locked_memories) {
            m_memories.at(port)->unlock(*m_stream);
        }
    }

    void emplace(size_t port, cldnn::memory::ptr mem) {
        m_memories.emplace(port, mem);
    }

    void emplace(size_t port, const ov::Tensor& tensor) {
        auto res = m_tensors.emplace(port, tensor);
        OPENVINO_ASSERT(res.first != m_tensors.end());
    }

    template<typename ElementType>
    void emplace(size_t port, std::vector<ElementType>& vector, data_types dt = data_types::i64) {
        ov::Shape shape{vector.size()};
        auto tensor = make_tensor({shape, dt, format::bfyx}, static_cast<void*>(vector.data()));
        m_tensors.emplace(port, tensor);
    }

    size_t size() const { return m_tensors.size(); }
    ov::Tensor operator[](std::size_t port) const {
        if (m_memories.count(port) > 0) {
            m_locked_memories.insert(port);
            auto mem = m_memories.at(port);
            auto ptr = mem->lock(*m_stream, cldnn::mem_lock_type::read);
            return make_tensor(mem->get_layout(), ptr);
        } else if (m_tensors.count(port) > 0) {
            return m_tensors.at(port);
        } else {
            return ov::Tensor{};
        }
    }

private:
    const cldnn::stream* m_stream;
    MemoryMap m_memories;
    TensorsMap m_tensors;

    mutable std::set<size_t> m_locked_memories = {};
};

class TensorAccessor final : public ov::ITensorAccessor {
public:
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
