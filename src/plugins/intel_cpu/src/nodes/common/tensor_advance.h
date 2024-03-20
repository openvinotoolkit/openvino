// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstring>
#include "openvino/core/parallel.hpp"
#include <common/utils.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include "cpu_memory.h"
#include "cpu_shape.h"
#include "cpu_types.h"
#include "utils/plain_tensor.hpp"

namespace ov {
namespace intel_cpu {

struct DimCounter {
    explicit DimCounter(const VectorDims& shape, const size_t start, const size_t end) : 
            m_shape(shape), m_start(start), m_end(end), m_offset(start), m_rank(shape.size()) {
        values.resize(m_rank);
        std::fill(values.begin(), values.end(), 0);
        if (m_start == 0) {
            return;
        }

        int64_t linear_offset = m_start;
        int64_t dim = static_cast<int64_t>(m_rank - 1);
        for (; dim >= 0; dim--) {
            int64_t size = m_shape[dim];
            if (size > 0) {
                values[dim] = linear_offset % size;
                linear_offset /= size;
            }
        }
        OPENVINO_ASSERT(linear_offset == 0);
    }

    // increase step along the last two dimemsions
    void increment(const std::array<size_t, 2>& step) {
        m_offset += step[0] * step[1];
        size_t overflow = step[0];
        int i = m_rank - 1;
        if (step[1] != 1) {
            OPENVINO_ASSERT(step[0] == m_shape[m_rank - 1] && values[m_rank - 1] == 0);
            i = m_rank - 2;
            overflow = step[1];
        }
        for (; i >= 0 && overflow > 0; i--) {
            auto size = m_shape[i];
            auto prev = values[i];
            auto value = prev + overflow;
            if (value >= size) {
                overflow = 1;
                value -= size;
                OPENVINO_ASSERT(value < size);
            } else {
                overflow = 0;
            }
            values[i] = value;
        }
        OPENVINO_ASSERT(overflow == 0 || overflow == 1);
    }

    std::array<size_t, 2> max_2d_step() const {
        auto last_dim = m_rank - 1;
        size_t step0 = std::min(static_cast<size_t>(m_shape[last_dim] - values[last_dim]), m_end - m_offset);
        size_t step1 = 1;
        if (step0 == m_shape[last_dim] && !m_shape.empty()) {
            step1 = std::min(m_shape[last_dim-1] - values[last_dim-1], (m_end - m_offset) / m_shape[last_dim]);
        }
        return {step0, step1};
    }

    bool is_done() const {
        return m_offset >= m_end;
    }

    VectorDims values;
    const VectorDims m_shape;
    const size_t m_start, m_end;  // half-closed interval
    size_t m_offset = 0;
    size_t m_rank;
};

template<std::size_t N>
struct TensorAdvance {
    // shape may be squashed.
    explicit TensorAdvance(const VectorDims& shape, const std::array<PlainTensor, N>& operands) : m_operands(operands), m_shape(shape) {
        get_base_ptrs(m_base_ptrs);
        get_strides(m_strides, m_shape.size());
    };

    using loop1d_t = std::function<void(char** data, const size_t* strides, const size_t n)>;

    void run(loop1d_t loop, const size_t start, const size_t end) const {
        size_t rank(m_shape.size());
        if (rank <= 1) {
            if (start == 0) {
                loop(const_cast<char**>(m_base_ptrs.data()), m_strides.data(), end - start);
            } else {
                std::array<char*, N> ptrs;;
                std::array<size_t, 1> coord = {start};
                get_data_ptrs(ptrs.data(), m_base_ptrs, m_strides.data(), coord.data(), 1);
                loop(ptrs.data(), m_strides.data(), end - start);
            }
        } else {
            std::array<char*, N> ptrs;;
            auto counter = DimCounter(m_shape, start, end);
            while(!counter.is_done()) {
                get_data_ptrs(
                    ptrs.data(), m_base_ptrs, m_strides.data(), counter.values.data(), rank);
                auto step = counter.max_2d_step();

                // loop_2d along the last two dimensions
                std::array<char*, N> data;
                std::copy(ptrs.begin(), ptrs.end(), data.begin());
                const size_t* outer_strides = &m_strides[N*(rank-2)];
                for (size_t i = 0; i < step[1]; i++) { // outer loop
                    if (i > 0) {
                        for (size_t arg = 0; arg < N; arg++) {
                            data[arg] += outer_strides[arg];
                        }
                    }
                    // inner loop along the last dim
                    loop(data.data(), &m_strides[N*(rank-1)], step[0]);
                }

                // increment
                counter.increment(step);
            }
        }
    }

private:
    void get_base_ptrs(std::array<char*, N>& base_ptrs) {
        std::transform(m_operands.begin(), m_operands.end(), base_ptrs.begin(), [](const PlainTensor& op) {
            return op.data<char>();
        });
    }

    void get_strides(VectorDims& strides, size_t rank) {
        strides.resize(N * std::max(rank, (size_t)2));
        auto iter = strides.begin();
        for (size_t dim = 0; dim < rank; dim++) {
            std::transform(m_operands.begin(), m_operands.end(), iter, [&](const PlainTensor& op) {
                return op.stride_bytes(dim);
            });
            iter += N;
        }
        // Always at least 2d strides to support max_2d_step
        if (rank < 2) {
            std::fill_n(iter, (2 - rank) * N, 0);
        }
    }

    void get_data_ptrs(
        char** ptrs,
        const std::array<char*, N>& base,
        const size_t* strides,
        const size_t* counter, size_t rank) const {
        std::copy(base.begin(), base.end(), ptrs);
        for (size_t dim = 0; dim < rank; dim++) {
            size_t value = counter[dim];
            for (size_t arg = 0; arg < N; arg++) {
            ptrs[arg] += value * strides[dim * N + arg];
            }
        }
    }

private:
    const std::array<PlainTensor, N>& m_operands;
    const VectorDims m_shape;
    std::array<char*, N> m_base_ptrs;  // base address of each operand
    VectorDims m_strides;     // strides in bytes
};

}   // namespace intel_cpu
}   // namespace ov