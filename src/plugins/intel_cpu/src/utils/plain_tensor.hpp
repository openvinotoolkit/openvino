// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#    include <cstdlib>
#endif

namespace ov::intel_cpu {

template <typename T>
inline void assert_dt(ov::element::Type dt) {
    OPENVINO_ASSERT(false);
}

template <>
inline void assert_dt<float>(ov::element::Type dt) {
    OPENVINO_ASSERT(dt == ov::element::f32);
}

template <>
inline void assert_dt<ov::bfloat16>(ov::element::Type dt) {
    OPENVINO_ASSERT(dt == ov::element::bf16);
}

template <>
inline void assert_dt<uint8_t>(ov::element::Type dt) {
    OPENVINO_ASSERT(dt == ov::element::u8);
}

template <>
inline void assert_dt<int8_t>(ov::element::Type dt) {
    OPENVINO_ASSERT(dt == ov::element::i8);
}

template <>
inline void assert_dt<int32_t>(ov::element::Type dt) {
    OPENVINO_ASSERT(dt == ov::element::i32);
}

template <>
inline void assert_dt<float16>(ov::element::Type dt) {
    OPENVINO_ASSERT(dt == ov::element::f16);
}

template <typename T>
struct precision_of {
    static constexpr ov::element::Type_t value = ov::element::Type_t::dynamic;
};

template <>
struct precision_of<float> {
    static constexpr ov::element::Type_t value = ov::element::Type_t::f32;
};

template <>
struct precision_of<int32_t> {
    static constexpr ov::element::Type_t value = ov::element::Type_t::i32;
};

template <>
struct precision_of<bfloat16> {
    static constexpr ov::element::Type_t value = ov::element::Type_t::bf16;
};

template <>
struct precision_of<uint8_t> {
    static constexpr ov::element::Type_t value = ov::element::Type_t::u8;
};

template <>
struct precision_of<float16> {
    static constexpr ov::element::Type_t value = ov::element::Type_t::f16;
};

#define PLAINTENSOR_RANK_MAX 8

struct PlainTensor {
    size_t m_strides[PLAINTENSOR_RANK_MAX] = {};
    size_t m_dims[PLAINTENSOR_RANK_MAX] = {};
    size_t m_rank = 0;
    std::shared_ptr<uint8_t> m_ptr;
    size_t m_capacity = 0;
    size_t m_element_size = 0;
    size_t m_offset = 0;
    ov::element::Type_t m_dt = ov::element::Type_t::dynamic;
    MemoryPtr m_mem;  // hold memory ptr reference

    operator bool() const {
        return m_ptr != nullptr;
    }

    [[nodiscard]] VectorDims shape() const {
        return VectorDims(m_dims, m_dims + m_rank);
    }

    [[nodiscard]] size_t size(int i) const {
        if (i < 0) {
            i += m_rank;
        }
        assert(static_cast<typename std::make_unsigned<decltype(i)>::type>(i) < m_rank);
        return m_dims[i];
    }
    [[nodiscard]] size_t stride(int i) const {
        assert(i >= 0 && static_cast<typename std::make_unsigned<decltype(i)>::type>(i) < m_rank);
        return m_strides[i];
    }

    [[nodiscard]] size_t stride_bytes(int i) const {
        return stride(i) * m_element_size;
    }

    template <typename T>
    [[nodiscard]] std::vector<T> get_strides() const {
        std::vector<T> strides(m_rank);
        for (size_t i = 0; i < m_rank; i++) {
            strides[i] = static_cast<T>(m_strides[i]);
        }
        return strides;
    }

    PlainTensor(const MemoryPtr& mem) {
        reset(mem);
    }

    PlainTensor() = default;

    PlainTensor operator=(const PlainTensor& other) {
        memcpy(&m_strides, &other.m_strides, sizeof(m_strides));
        memcpy(&m_dims, &other.m_dims, sizeof(m_dims));
        m_rank = other.m_rank;
        m_ptr = other.m_ptr;
        m_dt = other.m_dt;
        m_element_size = other.m_element_size;
        m_capacity = other.m_capacity;
        m_offset = other.m_offset;
        return *this;
    }

    void reset(const MemoryPtr& mem) {
        auto mem_desc = mem->getDescWithType<BlockedMemoryDesc>();
        // not support block layout
        OPENVINO_ASSERT(mem_desc && mem_desc->getOrder().size() == mem->getStaticDims().size());
        m_mem = mem;
        VectorDims strides(mem_desc->getStrides().size());
        const auto& orders = mem_desc->getOrder();
        for (size_t i = 0; i < orders.size(); i++) {
            strides[orders[i]] = mem_desc->getStrides()[i];
        }
        // this reshape_to() can do reshape w/o additional cost
        resize(mem->getStaticDims(),
               mem_desc->getPrecision().size(),
               mem_desc->getPrecision(),
               mem->getData(),
               strides.data());
    }

    [[nodiscard]] ov::element::Type get_precision() const {
        return m_dt;
    }

    struct tensor_index {
        int start;
        int end;
        int step;
        int count;
        // select all
        tensor_index() : start(0), end(INT_MAX), step(1) {}
        bool slice_with_squeeze() {
            return end == INT_MIN;
        }
        // tensor_index(start)            : select 1 element (with squeeze)
        // tensor_index(start, end, step) : select a range w/o squeeze
        tensor_index(int start, int end = INT_MIN, int step = 1) : start(start), end(end), step(step) {}

        void regularize(int size) {
            if (start < 0) {
                start += size;
            }
            assert(start >= 0 && start < size);
            if (end != INT_MIN) {
                if (end < 0) {
                    end += size;
                }
                if (end > size) {
                    end = size;
                }
                assert(end >= 0 && end <= size);
                count = (end - start + step - 1) / step;
            } else {
                count = 1;
            }
        }
    };

    PlainTensor index(const std::initializer_list<tensor_index>& indices) {
        PlainTensor sub_tensor;
        assert(indices.size() <= m_rank);
        int i_src = 0;
        int i_dst = 0;
        sub_tensor.m_capacity = 0;
        size_t off = 0;
        for (auto idx : indices) {
            auto src_dim = m_dims[i_src];
            auto src_stride = m_strides[i_src];
            idx.regularize(src_dim);
            off += idx.start * src_stride;
            if (idx.slice_with_squeeze()) {
                // no output dimension
                i_src++;
                continue;
            }
            sub_tensor.m_dims[i_dst] = idx.count;
            sub_tensor.m_strides[i_dst] = src_stride;
            i_dst++;
            i_src++;
        }
        sub_tensor.m_rank = i_dst;  // index may imply squeeze
        sub_tensor.m_ptr = m_ptr;
        sub_tensor.m_offset = m_offset + off;
        sub_tensor.m_dt = m_dt;
        sub_tensor.m_element_size = m_element_size;
        return sub_tensor;
    }

    // slice: return a sub-view (w/o ownership/refcount to original data)
    [[nodiscard]] PlainTensor slice(int axis, int start, int end, int step = 1) const {
        PlainTensor sub_tensor;
        assert(axis >= 0 && static_cast<typename std::make_unsigned<decltype(axis)>::type>(axis) < m_rank);

        sub_tensor.m_capacity = 0;
        if (end > start) {
            sub_tensor.m_rank = m_rank;
            for (size_t i = 0; i < m_rank; i++) {
                sub_tensor.m_strides[i] = m_strides[i];
                sub_tensor.m_dims[i] = m_dims[i];
            }
            sub_tensor.m_dims[axis] = (end - start - 1) / step + 1;
        } else {
            // squeeze if end == start
            sub_tensor.m_rank = m_rank - 1;
            size_t k = 0;
            for (size_t i = 0; i < m_rank; i++) {
                if (i != static_cast<size_t>(axis)) {
                    sub_tensor.m_strides[k] = m_strides[i];
                    sub_tensor.m_dims[k] = m_dims[i];
                    k++;
                }
            }
        }

        auto off = start * m_strides[axis];
        sub_tensor.m_ptr = m_ptr;
        sub_tensor.m_offset = m_offset + off;
        sub_tensor.m_dt = m_dt;
        sub_tensor.m_element_size = m_element_size;

        return sub_tensor;
    }

    [[nodiscard]] bool is_dense() const {
        // check if it's dense tensor
        size_t stride = 1;
        for (int i = m_rank - 1; i >= 0; i--) {
            if (m_strides[i] != stride) {
                return false;
            }
            stride *= m_dims[i];
        }
        return true;
    }

    /*
       suppose current shape is [a0,a1,...,am]
       and target shape is [b0,b1,...,bn]
       reshape is only valid when (a0*a1*...*am) == (b0*b1*...*bn) <======= (A)

       uniform a tensor's shape into groups from last to first, the dimension is merged
       into current group if the subtensor in the group is still dense after merge.
       otherwise a new group is formed.

       then reshape is performed on group basis, the check (A) is performed on group bases.
       which means any reshape inside the group is OK, but not across the group boundary.

       this can be done in one-loop, while group is forming, and checks are performed.

       simplified form is when whole tensor is dense
    */
    [[nodiscard]] PlainTensor reshape(const std::vector<size_t>& target_shape) const {
        // only valid for dense memory
        PlainTensor new_tensor_view;
        assert(is_dense());
        new_tensor_view.resize(target_shape,
                               m_element_size,
                               m_dt,
                               static_cast<void*>(m_ptr.get() + m_element_size * m_offset));
        return new_tensor_view;
    }

    [[nodiscard]] PlainTensor permute(const std::vector<size_t>& order) const {
        PlainTensor new_tensor_view;
        assert(order.size() == m_rank);
        new_tensor_view.m_capacity = 0;
        // not hold memory reference
        new_tensor_view.m_ptr = m_ptr;
        new_tensor_view.m_rank = m_rank;
        new_tensor_view.m_dt = m_dt;
        new_tensor_view.m_element_size = m_element_size;
        new_tensor_view.m_offset = m_offset;
        auto it_order = order.begin();
        // also should check order has no repeat element
        for (size_t i = 0; i < m_rank; i++) {
            auto j = *it_order++;
            assert(j >= 0 && j < m_rank);
            new_tensor_view.m_dims[i] = m_dims[j];
            new_tensor_view.m_strides[i] = m_strides[j];
        }
        return new_tensor_view;
    }

    void resize(const VectorDims& new_dims,
                size_t element_size,
                ov::element::Type_t dt,
                void* data = nullptr,
                const size_t* strides = nullptr) {
        m_element_size = element_size;
        m_dt = dt;
        // initialize strides for compact/dense tensor
        m_rank = new_dims.size();
        assert(m_rank <= PLAINTENSOR_RANK_MAX);
        size_t stride = 1;
        for (int i = m_rank - 1; i >= 0; i--) {
            m_dims[i] = new_dims[i];
            m_strides[i] = strides ? strides[i] : stride;
            stride *= new_dims[i];
        }

        if (!data) {
            auto capacity_new = m_strides[0] * m_dims[0] * m_element_size;
            if (capacity_new > m_capacity) {
                void* ptr;
#ifdef _WIN32
                ptr = _aligned_malloc(capacity_new, 64);
#else
                int rc = ::posix_memalign(&ptr, 64, capacity_new);
                if (rc) {
                    OPENVINO_ASSERT(false, "PlainTensor call posix_memalign failed: ", rc);
                }
#endif
                m_ptr = std::shared_ptr<uint8_t>(static_cast<uint8_t*>(ptr), [](uint8_t* ptr) {
#ifdef _WIN32
                    _aligned_free(ptr);
#else
                        ::free(ptr);
#endif
                });
                m_capacity = capacity_new;
                m_offset = 0;
            }
        } else {
            // m_capacity is zero to indicate that we don't own the memory
            m_capacity = 0;
            m_ptr = std::shared_ptr<uint8_t>(static_cast<uint8_t*>(data), [](uint8_t*) {});
        }
    }

    template <typename DT>
    void resize(const VectorDims& new_dims, DT* data = nullptr, const size_t* strides = nullptr) {
        resize(new_dims, sizeof(DT), precision_of<DT>::value, data, strides);
    }

    template <int dim>
    [[nodiscard]] int64_t offset() const {
        return m_offset;
    }
    template <int dim, typename I>
    int64_t offset(I i) const {
        return m_offset + i * m_strides[dim];
    }
    template <int dim, typename I, typename... Is>
    int64_t offset(I i, Is... indices) const {
        return i * m_strides[dim] + offset<dim + 1>(indices...);
    }
    template <typename DT, typename... Is>
    DT* ptr(Is... indices) const {
        return reinterpret_cast<DT*>(m_ptr.get()) + offset<0>(indices...);
    }

    template <typename... Is>
    void* ptr_v(Is... indices) const {
        return reinterpret_cast<void*>(m_ptr.get() + offset<0>(indices...) * m_element_size);
    }

    // when allow_broadcast is true, index to size-1 dim will always access 0.
    template <typename DT>
    DT& at(const std::initializer_list<size_t>& index, bool allow_broadcast = false) const {
        size_t off = 0;
        auto it = index.begin();
        for (size_t i = 0; i < m_rank; i++) {
            size_t coordinate = (it != index.end()) ? (*it++) : 0;
            if (allow_broadcast && m_dims[i] == 1) {
                // allow_broadcast only works when the dimension is really 1
                coordinate = 0;
            } else {
                assert(coordinate < m_dims[i]);
            }
            off += m_strides[i] * coordinate;
        }
        return (reinterpret_cast<DT*>(m_ptr.get() + (off + m_offset) * m_element_size))[0];
    }

    template <typename DT>
    PlainTensor& operator=(const DT& value) {
        // assign every element to value
        std::vector<size_t> index(m_rank, 0);
        auto* dst = reinterpret_cast<DT*>(m_ptr.get() + m_offset * m_element_size);
        while (true) {
            size_t off = 0;
            for (int i = m_rank - 1; i >= 0; i--) {
                if (index[i] >= m_dims[i]) {
                    // carry on
                    if (i == 0) {
                        return *this;
                    }
                    index[i] = 0;
                    index[i - 1]++;
                }
                off += m_strides[i] * index[i];
            }
            dst[off] = value;
            // move to next coordinate
            index[m_rank - 1]++;
        }
        return *this;
    }

    template <typename DT>
    DT& operator()(const std::initializer_list<size_t>& index, bool allow_broadcast = false) const {
        return at<DT>(index, allow_broadcast);
    }

    void assert_dims(const std::initializer_list<size_t>& expect_dims, bool special_zero = false) const {
        bool match = false;
        if (m_rank == expect_dims.size()) {
            match = true;
            auto it = expect_dims.begin();
            for (size_t i = 0; i < m_rank; ++i, ++it) {
                if (*it == 0 && special_zero) {
                    continue;
                }
                if (*it != m_dims[i]) {
                    match = false;
                    break;
                }
            }
        }

        if (!match) {
            std::stringstream ss;
            ss << " m_dims=[";
            for (size_t i = 0; i < m_rank; i++) {
                ss << m_dims[i] << ",";
            }
            ss << "] expect_dims=[";
            for (auto& i : expect_dims)
                ss << i << ",";
            ss << "]";
            // asm("int3");
            OPENVINO_THROW(ss.str());
        }
    }

    int max_repr_len = 256;

    [[nodiscard]] std::string repr(int max_total_lines = 16, int lines_per_row = 1) const {
        if (!m_ptr) {
            return "{empty}";
        }
        std::stringstream ss;
        ss << m_dt << " shape=[";
        const char* sep = "";
        size_t sz = 1;
        for (size_t i = 0; i < m_rank; i++) {
            ss << sep << m_dims[i];
            sz *= m_dims[i];
            sep = ",";
        }
        ss << "] strides=[";
        sep = "";
        for (size_t i = 0; i < m_rank; i++) {
            ss << sep << m_strides[i];
            sep = ",";
        }
        ss << "] {";
        if (m_rank > 1) {
            ss << "\n";
        }
        auto last_dim_size = m_dims[m_rank - 1];
        int row_id = 0;
        int cur_row_lines_left = lines_per_row;
        size_t cur_line_elecnt = 0;
        size_t cur_row_elecnt = 0;
        size_t i;
        for (i = 0; i < sz && max_total_lines > 0; i++) {
            if ((i % last_dim_size) == 0 && m_rank > 1) {
                ss << row_id << ":\t\t";
                row_id++;
                cur_row_lines_left = lines_per_row;
            }

            // display current element if we still have buget
            if (cur_row_lines_left > 0) {
                if (m_dt == ov::element::Type_t::f32) {
                    ss << (ptr<float>())[i] << ",";
                } else if (m_dt == ov::element::Type_t::bf16) {
                    ss << (ptr<bfloat16>())[i] << ",";
                } else if (m_dt == ov::element::Type_t::f16) {
                    ss << (ptr<float16>())[i] << ",";
                } else if (m_dt == ov::element::Type_t::i32) {
                    ss << (ptr<int32_t>())[i] << ",";
                } else if (m_dt == ov::element::Type_t::i8) {
                    ss << (ptr<int8_t>())[i] << ",";
                } else if (m_dt == ov::element::Type_t::u8) {
                    ss << (ptr<uint8_t>())[i] << ",";
                } else if (m_dt == ov::element::Type_t::boolean) {
                    ss << static_cast<bool>((ptr<uint8_t>())[i]) << ",";
                } else {
                    ss << "?,";
                }
                cur_line_elecnt++;
                cur_row_elecnt++;
                if (((cur_line_elecnt % 16) == 15 || (cur_row_elecnt == last_dim_size)) && (m_rank > 1)) {
                    max_total_lines--;
                    cur_row_lines_left--;
                    if (cur_row_lines_left == 0) {
                        if (cur_row_elecnt == last_dim_size) {
                            ss << ",\n";
                        } else {
                            ss << "...\n";
                        }
                        cur_row_elecnt = 0;
                    } else {
                        ss << "\n\t\t";
                    }
                    cur_line_elecnt = 0;
                }
            }
        }
        if (i < sz) {
            ss << "... ... ... ... \n";
        }
        ss << "}";
        return ss.str();
    }

    friend std::ostream& operator<<(std::ostream& os, const PlainTensor& dt);
};

inline std::ostream& operator<<(std::ostream& os, const PlainTensor& dt) {
    os << dt.repr();
    return os;
}

}  // namespace ov::intel_cpu
