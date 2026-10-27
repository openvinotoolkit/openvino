// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>

#include "openvino/core/type/element_type.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

// target device, data precision (query/key/value/caches), num_selection_heads,
// num_sequences (each sequence gets its own past_len and its own run of physical cache blocks,
// see classes/sparse_pa.cpp), k_head_size, key_block_size, page_size (KEY_CACHE/VALUE_CACHE dim 2;
// must be a multiple of key_block_size), cache precision (u8/u4: quantized KV caches; anything else: plain caches at
// the data precision), key_bychannel (quantized key cache by channel, else by token; the value cache is always by
// token), group_size (by-token quantization group, 0: whole row), q_len (new tokens per sequence: 1 is the decode path,
// more the prefill path)
using SparsePALayerTestParams = std::tuple<std::string,
                                           ov::element::Type,
                                           size_t,
                                           size_t,
                                           size_t,
                                           size_t,
                                           size_t,
                                           ov::element::Type,
                                           bool,
                                           size_t,
                                           size_t>;

class SparsePALayerCPUTest : public testing::WithParamInterface<SparsePALayerTestParams>,
                             virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SparsePALayerTestParams>& obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    std::vector<ov::Tensor> calculate_refs() override;
    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override;

private:
    std::map<std::shared_ptr<ov::Node>, ov::Tensor> host_inputs;
    ov::element::Type m_data_precision = ov::element::f32;
    size_t m_num_selection_heads = 1;
    size_t m_num_sequences = 1;
    size_t m_k_head_size = 16;
    size_t m_key_block_size = 2;
    size_t m_page_size = 4;
    size_t m_seq_blocks = 4;
    ov::element::Type m_cache_precision = ov::element::dynamic;
    bool m_key_bychannel = false;
    size_t m_group_size = 0;
    size_t m_q_len = 8;
    std::array<int32_t, 3> m_past_lens = {0, 5, 13};
    // Effective (dequantized) values of the quantized history rows, per sequence, laid out [kv, past_len, head_size].
    std::vector<std::vector<float>> m_hist_key;
    std::vector<std::vector<float>> m_hist_value;
    [[nodiscard]] bool quantized() const {
        return m_cache_precision == ov::element::u8 || m_cache_precision == ov::element::u4;
    }
    // Padded key/value cache shapes: quantization parameters live inside the cache tensor (see SparsePAQuant).
    [[nodiscard]] ov::Shape cache_shape(bool is_key) const;
    float m_scale = 0.25F;
};

}  // namespace test
}  // namespace ov