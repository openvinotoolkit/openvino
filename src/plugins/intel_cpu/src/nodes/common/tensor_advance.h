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

template<std::size_t N>
inline void get_base_ptrs(char** ptrs, const std::array<PlainTensor, N>& operands) {
  std::transform(operands.begin(), operands.end(), ptrs, [](const PlainTensor& op) {
    return op.data<char>();
  });
}

template<std::size_t N>
inline void get_strides(size_t* strides, const std::array<PlainTensor, N>& operands, size_t ndim) {
  for (size_t dim = 0; dim < ndim; dim++) {
    for (const PlainTensor& op : operands) {
        *strides++ = op.get_byte_strides<size_t>()[dim];
    }
  }
  // Always at least 2d strides to support 2d for_each loops
  if (ndim < 2) {
    std::fill_n(strides, (2 - ndim) * N, 0);
  }
}

template<std::size_t N>
inline void get_data_ptrs(
    char** ptrs,
    const std::array<char*, N>& base,
    const size_t* strides,
    const size_t* counter, size_t ndim) {
  const size_t ntensors = base.size();
  std::copy(base.begin(), base.end(), ptrs);
  for (size_t dim = 0; dim < ndim; dim++) {
    size_t value = counter[dim];
    for (size_t arg = 0; arg < ntensors; arg++) {
      ptrs[arg] += value * strides[dim * ntensors + arg];
    }
  }
}

class TensorAdvance {
public:
    TensorAdvance(const VectorDims& squashed_shape, const size_t squashed_axis, const size_t start, const size_t end) : 
            m_squashed_shape(squashed_shape), m_squashed_axis(squashed_axis), m_start(start), m_end(end), m_offset(start),
            m_rank(squashed_shape.size()) {
        OPENVINO_ASSERT(m_squashed_shape[m_squashed_axis] == 1);
        values.resize(m_rank);
        std::fill(values.begin(), values.end(), 0);
        if (m_start == 0) {
            return;
        }

        int64_t linear_offset = m_start;
        int64_t dim = static_cast<int64_t>(m_rank - 1);
        for (; dim >= 0; dim--) {
            int64_t size = m_squashed_shape[dim];
            if (size > 0) {
                values[dim] = linear_offset % size;
                linear_offset /= size;
            }
        }
        OPENVINO_ASSERT(linear_offset == 0);
    }

    using loop1d_t = std::function<void(char** data, const size_t* strides, size_t n)>;

    template<std::size_t N>
    void run(loop1d_t loop, const std::array<PlainTensor, N>& operands) {
        std::array<char*, N> base_ptrs;
        std::vector<size_t> strides(N * std::max(m_rank, (size_t)2));

        get_base_ptrs(base_ptrs.data(), operands);
        get_strides(strides.data(), operands, m_rank);

        if (m_rank <= 1) {
            if (m_start == 0) {
                loop(base_ptrs.data(), strides.data(), m_end-m_start);
            } else {
                std::array<char*, N> ptrs;;
                std::array<size_t, 1> coord = {m_start};
                get_data_ptrs(ptrs.data(), base_ptrs, strides.data(), coord.data(), 1);
                loop(ptrs.data(), strides.data(), m_end-m_start);
            }
        } else {
            std::array<char*, N> ptrs;;           
            while(!is_done()) {
                get_data_ptrs(
                    ptrs.data(), base_ptrs, strides.data(), values.data(), m_rank);
                auto step = max_2d_step();

                // loop_2d along the last two dimensions
                std::array<char*, N> data;
                std::copy(ptrs.begin(), ptrs.end(), data.begin());
                const size_t* outer_strides = &strides[N*(m_rank-2)];
                for (size_t i = 0; i < step[1]; i++) { // outer loop
                    if (i > 0) {
                        for (size_t arg = 0; arg < N; arg++) {
                            data[arg] += outer_strides[arg];
                        }
                    }
                    // inner loop along the last dim
                    loop(data.data(), &strides[N*(m_rank-1)], step[0]);
                }

                // increment
                increment(step);
            }
        }
    }

private:
    // increase m_step at the last dimemsion
    void increment(const std::array<size_t, 2>& step) {
        m_offset += step[0] * step[1];
        size_t overflow = step[0];
        int i = m_rank - 1;
        if (step[1] != 1) {
            OPENVINO_ASSERT(step[0] == m_squashed_shape[m_rank - 1] && values[m_rank - 1] == 0);
            i = m_rank - 2;
            overflow = step[1];
        }
        for (; i >= 0 && overflow > 0; i--) {
            auto size = m_squashed_shape[i];
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

    std::array<size_t, 2> max_2d_step() {
        auto last_dim = m_rank - 1;
        size_t step0 = std::min(static_cast<size_t>(m_squashed_shape[last_dim] - values[last_dim]), m_end - m_offset);
        size_t step1 = 1;
        if (step0 == m_squashed_shape[last_dim] && !m_squashed_shape.empty()) {
            step1 = std::min(m_squashed_shape[last_dim-1] - values[last_dim-1], (m_end - m_offset) / m_squashed_shape[last_dim]);
        }
        return {step0, step1};
    }

    bool is_done() const {
        return m_offset >= m_end;
    }

    VectorDims values;
    const VectorDims m_squashed_shape;
    const size_t m_squashed_axis;
    const size_t m_start, m_end;  // half-closed interval
    size_t m_offset = 0;
    size_t m_rank;
};

}   // namespace intel_cpu
}   // namespace ov