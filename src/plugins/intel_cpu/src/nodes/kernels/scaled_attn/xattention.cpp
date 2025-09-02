#include "xattention.hpp"

#include <algorithm>
#include <chrono>
#include <common/z_magic.hpp>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <ios>
#include <limits>
#include <numeric>
#include <vector>

#include "nodes/kernels/x64/brgemm_kernel.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"

using namespace ov::intel_cpu;
namespace ov::Extensions::Cpu::XARCH {

namespace ref {
void softmax(float* a, int len) {
    float max = *std::max_element(a, a + len);
    float sum = 0.0F;
    for (int i = 0; i < len; i++) {
        a[i] = ::exp(a[i] - max);
        sum += a[i];
    }
    float scale = 1.0F / sum;
    for (int i = 0; i < len; i++) {
        a[i] *= scale;
    }
}

template <typename D>
float dot_product(const D* a, const D* b, int len, int stride_b = 1) {
    float result = 0;
    if (stride_b == 1) {
        for (int i = 0; i < len; i++) {
            result += static_cast<float>(a[i]) * static_cast<float>(b[i]);
        }
    } else {
        for (int i = 0; i < len; i++) {
            result += static_cast<float>(a[i]) * static_cast<float>(b[i * stride_b]);
        }
    }
    return result;
}

}  // namespace ref

PlainTensor xattn_estimate(PlainTensor& query,
                           PlainTensor& key,
                           size_t block_size,
                           size_t stride,
                           int norm = 1,
                           float threshold = 0.9f,
                           bool causal = true) {
    auto B = query.size(0);
    auto H = query.size(1);
    auto L = query.size(2);
    auto S = query.size(3);

    auto K_H = key.size(1);
    auto groups = H / K_H;

    auto q_num_strided = B / stride;
    auto k_num_strided = B / stride;
    auto q_num_blocks = B / block_size;
    auto k_num_blocks = B / block_size;
    auto num_per_block = block_size / stride;
    if (q_num_blocks == 0 || k_num_blocks == 0 || num_per_block == 0) {
        return {};
    }

    PlainTensor attn_sum_temp;
    attn_sum_temp.resize({q_num_strided, H, L, k_num_strided}, sizeof(float), ov::element::Type_t::f32);

    int n_block_size = 32;
    auto in_type = query.m_dt;
    auto kernel = std::make_shared<BrgemmKernel>(q_num_strided,      // M
                                                 n_block_size,       // N
                                                 S,                  // K
                                                 S * stride * H,     // lda
                                                 S * stride * K_H,   // ldb
                                                 q_num_strided * H,  // ldc
                                                 true,
                                                 in_type,
                                                 true);

    auto max_threads_num = parallel_get_max_threads();
    auto scratch_a_size = kernel->get_scratch_a_size();
    auto scratch_b_size = kernel->get_scratch_b_size();
    auto wsp_size = kernel->get_wsp_size();

    std::vector<uint8_t> scratch_a(scratch_a_size * max_threads_num, 0.0f);
    std::vector<uint8_t> scratch_b(scratch_b_size * max_threads_num, 0.0f);
    std::vector<uint8_t> wsp(wsp_size * max_threads_num, 0.0f);

    for (size_t i = 0; i < stride; i++) {
        auto m_block_size = kernel->get_mblk_size();
        auto m_blocks = (q_num_strided + kernel->get_mblk_size() - 1) / m_block_size;

        ov::parallel_for2d(H, m_blocks, [&](size_t h, size_t m_blk) {
            auto ithr = parallel_get_thread_num();
            auto m_start = m_blk * m_block_size;
            auto m_end = std::min(m_start + m_block_size, q_num_strided);
            auto m_cnt = m_end - m_start;
            auto* q_ptr = &query.at<float>({stride - 1 - i + m_start * stride, h, 0, 0});

            auto cur_k_block_num = m_blk + 1;
            for (size_t n_blk = 0; n_blk < cur_k_block_num; n_blk++) {
                auto n_start = n_blk * n_block_size;
                auto* k_ptr = &key.at<float>({i + n_start * stride, h / groups, 0, 0});
                kernel->copy_buffer_b(k_ptr, scratch_b.data() + scratch_b_size * ithr);

                auto* c_ptr = &attn_sum_temp.at<float>({m_start, h, 0, n_start});
                kernel->executeGemm(m_cnt < m_block_size,
                                    q_ptr,
                                    scratch_b.data() + scratch_b_size * ithr,
                                    c_ptr,
                                    nullptr,
                                    nullptr,
                                    wsp.data(),
                                    scratch_a.data() + scratch_a_size * ithr);
            }
        });
    }

    parallel_for3d(q_num_strided, H, L, [&](size_t b, size_t h, size_t l) {
        auto* data = attn_sum_temp.ptr<float>(b, h, l, 0);

        for (size_t s = 0; s < k_num_strided; s++) {
            if (causal && b < s) {
                data[s] = -std::numeric_limits<float>::infinity();
            } else {
                data[s] = data[s] / sqrt(S) / stride / norm;
            }
        }

        ref::softmax(data, k_num_strided);
    });

    PlainTensor attn_sum;
    attn_sum.resize({q_num_blocks, H, L, k_num_blocks}, attn_sum_temp.m_element_size, attn_sum_temp.m_dt);
    parallel_for2d(H, L, [&](size_t h, size_t l) {
        for (size_t row = 0; row < q_num_blocks; row++) {
            for (size_t col = 0; col < k_num_blocks; col++) {
                auto* out = attn_sum.ptr<float>(row, h, l, col);

                float value = 0.0f;
                for (size_t i = 0; i < num_per_block; i++) {
                    for (size_t j = 0; j < num_per_block; j++) {
                        auto* in = attn_sum_temp.ptr<float>(row * num_per_block + i, h, l, col * num_per_block + j);
                        value += *in;
                    }
                }

                *out = value;
            }
        }
    });

    // Find blocks
    PlainTensor mask;
    mask.resize({H, L, q_num_blocks, k_num_blocks}, sizeof(bool), ov::element::Type_t::boolean);
    parallel_for3d(q_num_blocks, H, L, [&](size_t b, size_t h, size_t l) {
        auto* row = attn_sum.ptr<float>(b, h, l, 0);
        float required_sum = std::accumulate(row, row + k_num_blocks, 0.0f, std::plus<>()) * threshold;

        std::vector<std::pair<float, int>> values_with_index(q_num_blocks);
        for (size_t k = 0; k < k_num_blocks; k++) {
            values_with_index[k] = std::make_pair(row[k], k);
        }

        if (causal) {
            values_with_index[b].first += 100000.0f;
        }

        std::sort(values_with_index.begin(),
                  values_with_index.end(),
                  [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                      return a.first > b.first;
                  });

        if (causal) {
            values_with_index[0].first = 0.0f;
        }

        std::vector<float> cumsum_without_self(k_num_blocks, 0.0f);
        for (size_t i = 1; i < k_num_blocks; i++) {
            cumsum_without_self[i] = values_with_index[i - 1].first + cumsum_without_self[i - 1];
        }

        for (size_t i = 0; i < k_num_blocks; i++) {
            bool value = (cumsum_without_self[i] < required_sum);

            if (causal && i > b) {
                value = false;
            }

            if (values_with_index[i].second == 0) {
                value = true;
            }

            *(mask.ptr<bool>(h, l, b, values_with_index[i].second)) = value;
        }
    });

    return mask;
}
}  // namespace ov::Extensions::Cpu::XARCH
