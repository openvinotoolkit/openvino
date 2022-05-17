/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "gpu/jit/conv/gemm_schedule.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

layout_t bmnk_mapper_t::map_to_bmnk(abc_kind_t abc_kind,
        const std::vector<bmnk_kind_t> &bmnk_kinds, const view_t &view) const {
    auto layout = view.create_pseudo_vlayout();
    return map_to_bmnk(abc_kind, bmnk_kinds, layout);
}

layout_t bmnk_mapper_t::map_to_bmnk(abc_kind_t abc_kind,
        const std::vector<bmnk_kind_t> &bmnk_kinds,
        const layout_t &layout) const {
    std::vector<block_t> blocks;
    for (auto &b : layout.blocks()) {
        auto b_bmnk_kind = bmnk_kind(abc_kind, b.dim_idx);
        bool found = false;
        for (int i = 0; i < int(bmnk_kinds.size()); i++) {
            if (bmnk_kinds[i] == b_bmnk_kind) {
                blocks.emplace_back(i, b.block, b.stride);
                found = true;
                break;
            }
        }
        if (!found) ir_error_not_expected() << "MNK dimension not found.";
    }
    return layout_t(layout.type(), int(bmnk_kinds.size()), 0, blocks);
}

void bmnk_block_mapper_t::push_block(abc_kind_t abc_kind, const block_t &b) {
    auto bmnk_kind = bmnk_mapper_.bmnk_kind(abc_kind, b.dim_idx);
    switch (bmnk_kind) {
        case bmnk_kind_t::m: m_blocks_.emplace_back(abc_kind, b); break;
        case bmnk_kind_t::n: n_blocks_.emplace_back(abc_kind, b); break;
        case bmnk_kind_t::k: k_blocks_.emplace_back(abc_kind, b); break;
        default: ir_error_not_expected() << "Unknown MNK kind.";
    }
}

layout_t bmnk_block_mapper_t::map_from_bmnk(abc_kind_t abc_kind,
        const std::vector<bmnk_kind_t> &bmnk_kinds,
        const layout_t &bmnk_layout) const {
    ir_assert(bmnk_layout.ndims() <= 3);
    ir_assert(bmnk_layout.has_zero_offset());
    std::vector<block_t> blocks;
    std::vector<std::vector<block_t>> tmp_blocks(
            static_cast<int>(bmnk_kind_t::k) + 1);
    tmp_blocks[static_cast<int>(bmnk_kind_t::m)]
            = create_prb_blocks(abc_kind, m_blocks_);
    tmp_blocks[static_cast<int>(bmnk_kind_t::n)]
            = create_prb_blocks(abc_kind, n_blocks_);
    tmp_blocks[static_cast<int>(bmnk_kind_t::k)]
            = create_prb_blocks(abc_kind, k_blocks_);
    for (auto &b : bmnk_layout.blocks()) {
        auto &bmnk_blocks = tmp_blocks[static_cast<int>(bmnk_kinds[b.dim_idx])];
        bool ok = pop_block(bmnk_blocks, blocks, b);
        ir_assert(ok) << "Can't map from bmnk layout to problem layout.";
        MAYBE_UNUSED(ok);
    }
    for (auto bmnk_kind : bmnk_kinds) {
        auto &bmnk_blocks = tmp_blocks[static_cast<int>(bmnk_kind)];
        pop_size_1_blocks(bmnk_blocks);
        ir_assert(bmnk_blocks.empty());
    }

    // Fix strides to make them dense.
    dim_t dense_stride = 1;
    for (auto &b : blocks) {
        b.stride = stride_t(dense_stride);
        dense_stride *= b.block;
    }

    return layout_t(
            bmnk_layout.type(), bmnk_mapper_.ndims(abc_kind), 0, blocks);
}

bool bmnk_block_mapper_t::pop_block(std::vector<block_t> &bmnk_blocks,
        std::vector<block_t> &prb_blocks, const block_t &bmnk_block) const {
    if (bmnk_block.block == 1) return true;

    pop_size_1_blocks(bmnk_blocks);
    if (bmnk_blocks.empty()) return false;

    auto &next_block = bmnk_blocks.front();
    dim_t common_block = math::gcd(next_block.block, bmnk_block.block);
    if (common_block == bmnk_block.block) {
        prb_blocks.emplace_back(
                next_block.dim_idx, common_block, next_block.stride);
        next_block.block /= common_block;
        next_block.stride *= common_block;
        return true;
    } else if (common_block == next_block.block) {
        prb_blocks.emplace_back(
                next_block.dim_idx, common_block, next_block.stride);
        bmnk_blocks.erase(bmnk_blocks.begin());
        auto tmp_block = bmnk_block;
        tmp_block.block /= common_block;
        return pop_block(bmnk_blocks, prb_blocks, tmp_block);
    }
    return false;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
