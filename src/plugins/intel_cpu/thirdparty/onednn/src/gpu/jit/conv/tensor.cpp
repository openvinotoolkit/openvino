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

#include <cctype>
#include <sstream>
#include <thread>

#include "gpu/jit/conv/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

layout_t::layout_t(const type_t &type, const expr_t &offset,
        const std::string &format, const std::vector<dim_t> &dims,
        bool do_normalize)
    : type_(type), offset_(offset) {
    auto parts = parse_format(format, int(dims.size()));
    ndims_ = 0;
    for (auto &p : parts) {
        int dim_idx = p.first;
        dim_t block = p.second;
        ndims_ = std::max(ndims_, dim_idx + 1);
        if (block == 0 && dims.empty())
            ir_error_not_expected()
                    << "Dimensions are missing. Can't deduce them from "
                       "the format.";
    }
    if (!dims.empty() && ndims_ != int(dims.size())) {
        ir_error_not_expected() << "Format and dimensions do not match.";
    }

    dim_t stride = 1;
    // Iterate from right to left (innermost to outermost).
    for (auto it = parts.rbegin(); it != parts.rend(); ++it) {
        int dim_idx = it->first;
        dim_t block = it->second;
        if (block == 0) {
            dim_t full_block = 1;
            for (auto &b : blocks_)
                if (b.dim_idx == dim_idx) full_block *= b.block;

            block = utils::div_up(dims[dim_idx], full_block);
        }

        blocks_.emplace_back(dim_idx, block, stride);
        stride = block * stride;
    }

    if (do_normalize) blocks_ = normalize_blocks(ndims_, blocks_);
    sanity_check();
}

layout_t::layout_t(const memory_desc_wrapper &mdw, bool do_normalize)
    : type_(mdw.data_type()), offset_(mdw.offset0()) {
    ir_assert(mdw.is_blocking_desc()) << "Expected blocking memory descriptor.";

    ndims_ = mdw.ndims();
    auto &blocking = mdw.blocking_desc();
    auto *padded_dims = mdw.padded_dims();

    dim_t stride = 1;
    std::vector<dim_t> full_blocks(ndims_, 1);
    for (int i = blocking.inner_nblks - 1; i >= 0; i--) {
        int dim_idx = blocking.inner_idxs[i];
        dim_t block = blocking.inner_blks[i];
        blocks_.emplace_back(dim_idx, block, stride);
        stride *= block;
        full_blocks[dim_idx] *= block;
    }

    for (int i = 0; i < ndims_; i++) {
        dim_t block = padded_dims[i] / full_blocks[i];
        blocks_.emplace_back(i, block, blocking.strides[i]);
    }

    // Sort outer blocks by their stride.
    std::sort(blocks_.begin() + blocking.inner_nblks, blocks_.end(),
            [](const block_t &a, const block_t &b) {
                if (a.stride == b.stride) return a.dim_idx > b.dim_idx;
                return a.stride < b.stride;
            });

    if (do_normalize) blocks_ = normalize_blocks(ndims_, blocks_);
    sanity_check();
}

memory_desc_t layout_t::to_dnnl(const dim_t *dims_hint) const {
    memory_desc_t md = {};
    md.ndims = ndims();
    std::copy(dims_hint, dims_hint + ndims(), md.dims);
    md.data_type = jit::to_dnnl(type_);
    md.offset0 = to_cpp<dim_t>(offset_);
    md.format_kind = format_kind::blocked;

    auto &blk = md.format_desc.blocking;
    bool seen[DNNL_MAX_NDIMS] = {};

    bool in_inner_block = false;
    dim_t prev_stride = 0;

    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
        auto &b = *it;
        if (!seen[b.dim_idx]) {
            // Outer block.
            ir_assert(!in_inner_block);
            MAYBE_UNUSED(in_inner_block);
            blk.strides[b.dim_idx] = b.stride;
            md.padded_dims[b.dim_idx] = b.block;
        } else {
            // Inner block.
            md.padded_dims[b.dim_idx] *= b.block;
            blk.inner_idxs[blk.inner_nblks] = b.dim_idx;
            blk.inner_blks[blk.inner_nblks] = b.block;
            blk.inner_nblks++;
            if (prev_stride > 0) {
                // Inner block must be dense.
                ir_assert(prev_stride == b.block * b.stride);
            }
            prev_stride = b.stride;
            in_inner_block = true;
        }
        seen[b.dim_idx] = true;
    }

    return md;
}

layout_t layout_t::map(const tensor_t &tensor) const {
    if (ndims() != tensor.ndims())
        ir_error_not_expected() << "Dimensions do not match.";

    std::vector<dim_t> remaining_dims = tensor.dims();
    std::vector<block_t> mapped_blocks;

    for (auto &eb : enumerated_blocks()) {
        block_t &b = eb.second;
        bool b_is_outermost = is_outermost(eb);

        dim_t block = b.block;
        dim_t &rem_dim = remaining_dims[b.dim_idx];
        if (rem_dim == 1) {
            if (b_is_outermost) {
                // This is to have similarity between the current and
                // mapped layouts.
                mapped_blocks.emplace_back(b.dim_idx, 1, b.stride);
            }
            continue;
        }
        if (b_is_outermost) {
            block = rem_dim;
        } else if (rem_dim % block != 0) {
            // Try to split the current block and start mapping from
            // scratch.
            if (block % rem_dim == 0)
                return split_block(eb, rem_dim, block / rem_dim).map(tensor);

            ir_error_not_expected() << "Can't map tensor layout.";
        }
        rem_dim /= block;
        mapped_blocks.emplace_back(b.dim_idx, block, b.stride);
    }

    for (auto &d : remaining_dims) {
        ir_assert(d == 1) << "Can't map tensor layout.";
        MAYBE_UNUSED(d);
    }

    return layout_t(type(), ndims(), operator()(tensor.start()), mapped_blocks);
}

layout_t layout_t::reinterpret(
        const type_t &new_type, bool do_normalize) const {
    int old_size = type().size();
    int new_size = new_type.size();
    if (new_size == old_size) return *this;

    expr_t new_offset = 0;
    if (!has_zero_offset()) {
        ir_assert(is_const(offset_)) << "Expected constant offset.";
        int64_t off = to_cpp<int64_t>(offset_) * old_size;
        ir_assert(off % new_size == 0);
        new_offset = off / new_size;
    }

    if (old_size % new_size != 0 && new_size % old_size != 0) {
        ir_error_not_expected();
        return layout_t();
    }

    auto new_blocks = blocks_;
    if (new_blocks.empty()) {
        ir_error_not_expected() << "Can't reinterpret.";
        return layout_t();
    }

    if (new_size < old_size) {
        int factor = (old_size / new_size);
        auto &b0 = new_blocks.front();
        b0.block *= factor;
        // Recompute strides.
        for (auto &b : new_blocks) {
            if (&b == &b0) continue;
            b.stride *= factor;
        }
    } else {
        int factor = (new_size / old_size);
        auto &b0 = new_blocks.front();
        if (b0.block % factor != 0) {
            ir_error_not_expected();
            return layout_t();
        }
        b0.block /= factor;
        // Recompute strides.
        for (auto &b : new_blocks) {
            if (&b == &b0) continue;
            if (b.stride % factor != 0) {
                ir_error_not_expected();
                return layout_t();
            }
            b.stride /= factor;
        }
    }

    return layout_t(new_type, ndims(), new_offset, new_blocks, do_normalize);
}

layout_t layout_t::split_block(
        const std::pair<int, block_t> &eb, dim_t block0, dim_t block1) const {
    int block_idx = eb.first;
    auto &b = eb.second;
    ir_assert(b.block == block0 * block1) << "Incompatible block sizes.";
    MAYBE_UNUSED(b);

    auto new_blocks = blocks_;

    block_t &b0 = new_blocks[block_idx];
    block_t b1 = b0;

    b0.block = block0;
    b1.block = block1;
    b1.stride = b0.stride * block0;

    new_blocks.insert(new_blocks.begin() + block_idx + 1, b1);

    return layout_t(
            type(), ndims(), offset(), new_blocks, /*do_normalize=*/false);
}

layout_t layout_t::split_into_multi_blocks(
        const std::vector<dim_t> &multi_blocks) const {
    return split_into_multi_blocks_impl(multi_blocks, nullptr);
}

layout_t layout_t::split_into_multi_blocks_with_hint(
        std::vector<dim_t> &multi_blocks) const {
    return split_into_multi_blocks_impl(multi_blocks, &multi_blocks);
}

tensor_t layout_t::split_into_dense_tile(
        dim_t tile_elems, dim_t outer_block) const {
    stride_t dense_stride = 1;
    dim_t cur_tile_elems = 1;
    dim_t cur_outer_block = 1;
    bool in_tile = (tile_elems != 1);
    std::vector<dim_t> tile_dims(ndims(), 1);
    for (auto &b : blocks()) {
        if (b.block == 1) continue;
        if (in_tile) {
            if (b.stride.is_unknown()) return tensor_t();
            if (dense_stride != b.stride) return tensor_t();
            dense_stride = b.block * b.stride;
            cur_tile_elems *= b.block;
            tile_dims[b.dim_idx] *= b.block;
            ir_assert(cur_tile_elems <= tile_elems);
            if (cur_tile_elems == tile_elems) in_tile = false;
        } else {
            if (outer_block == 1) break;
            cur_outer_block *= b.block;
            tile_dims[b.dim_idx] *= b.block;
            ir_assert(cur_outer_block <= outer_block);
            if (cur_outer_block == outer_block) break;
        }
    }
    if (cur_tile_elems != tile_elems) return tensor_t();
    if (cur_outer_block != outer_block) return tensor_t();
    return tensor_t(tile_dims);
}

tensor_t layout_t::split_into_max_tile(
        dim_t max_tile_elems, bool is_dense_tile) const {
    stride_t dense_stride = 1;
    std::vector<dim_t> tile_dims(ndims(), 1);
    dim_t cur_elems = 1;
    for (auto &eb : enumerated_blocks()) {
        auto &b = eb.second;
        if (b.block == 1) continue;
        if (b.block * cur_elems <= max_tile_elems) {
            if (is_dense_tile) {
                if (b.stride.is_unknown()) break;
                if (dense_stride != b.stride) break;
                dense_stride = b.block * b.stride;
            }
            cur_elems *= b.block;
            tile_dims[b.dim_idx] *= b.block;
            continue;
        }
        dim_t max_block = utils::max_div(b.block, max_tile_elems / cur_elems);
        if (max_block == 1) break;
        auto tmp_layout = split_block(eb, max_block, b.block / max_block);
        return tmp_layout.split_into_max_tile(max_tile_elems, is_dense_tile);
    }
    return tensor_t(tile_dims);
}

void layout_t::align_layouts(layout_t &a, layout_t &b) {
    for (int i = 0; i < a.ndims(); i++) {
        auto a_blocks = a.blocks();
        auto b_blocks = b.blocks();

        int a_max = int(a_blocks.size());
        int b_max = int(b_blocks.size());
        int a_idx = 0;
        int b_idx = 0;

        for (;;) {
            while (a_idx < a_max && a_blocks[a_idx].dim_idx != i)
                a_idx++;
            while (b_idx < b_max && b_blocks[b_idx].dim_idx != i)
                b_idx++;

            if (a_idx >= a_max || b_idx >= b_max) break;

            auto &ab = a_blocks[a_idx];
            auto &bb = b_blocks[b_idx];
            dim_t common_block = math::gcd(ab.block, bb.block);
            if (ab.block == common_block && bb.block == common_block) {
                a_idx++;
                b_idx++;
                continue;
            }

            if (ab.block != common_block) {
                a = a.split_block(
                        {a_idx, ab}, common_block, ab.block / common_block);
            }
            if (bb.block != common_block) {
                b = b.split_block(
                        {b_idx, bb}, common_block, bb.block / common_block);
            }
            break;
        }
    }
}

std::vector<std::pair<char, dim_t>> layout_t::parse_letter_blocks(
        const std::string &format) {
    std::vector<std::pair<char, dim_t>> ret;

    std::stringstream ss(format);
    while (!ss.eof()) {
        int next = ss.peek();
        if (ss.eof()) break;
        dim_t block = 0;
        while (std::isdigit(next)) {
            block = 10 * block + (next - '0');
            ss.ignore(1);
            next = ss.peek();
        }
        char letter = char(ss.peek());
        ir_assert(!ss.eof()) << "EOF is unexpected.";
        ss.ignore(1);
        ret.emplace_back(letter, block);
    }
    return ret;
}

std::vector<std::pair<int, dim_t>> layout_t::parse_format(
        const std::string &format, int ndims_hint) {
    bool seen_letters[DNNL_MAX_NDIMS] = {};
    int letter_ndims = 0;
    for (char c = 'a'; c < 'a' + DNNL_MAX_NDIMS; c++) {
        if (format.find(c) != std::string::npos) {
            seen_letters[c - 'a'] = true;
            MAYBE_UNUSED(seen_letters);
            letter_ndims++;
        }
    }

    for (int i = 0; i < DNNL_MAX_NDIMS; i++) {
        ir_assert(seen_letters[i] == (i < letter_ndims));
    }

    auto letter_blocks = parse_letter_blocks(format);

    std::vector<std::pair<int, dim_t>> parts;
    for (auto &p : letter_blocks) {
        char letter = p.first;
        dim_t block = p.second;
        if (letter != 'x') {
            int dim_idx = std::tolower(letter) - 'a';
            parts.emplace_back(dim_idx, block);
        } else {
            ir_assert(ndims_hint >= letter_ndims);
            for (int i = letter_ndims; i < ndims_hint; i++) {
                parts.emplace_back(i, 0);
            }
        }
    }

    return parts;
}

void layout_t::sanity_check() const {
#ifdef NDEBUG
    return;
#endif
    if (is_empty()) return;

    for (auto &b : blocks_) {
        ir_assert(b.block > 0) << "Incorrect block size.";
        MAYBE_UNUSED(b);
    }
}

layout_t layout_t::split_into_multi_blocks_impl(
        const std::vector<dim_t> &multi_blocks,
        std::vector<dim_t> *out_multi_blocks) const {
    if (is_empty()) return *this;

    bool allow_smaller_blocks = bool(out_multi_blocks);
    layout_t tmp(*this);
    std::vector<dim_t> rem_elems = multi_blocks;
    std::vector<dim_t> cur_elems(rem_elems.size(), 1);
    for (auto &eb : tmp.enumerated_blocks()) {
        auto &b = eb.second;
        for (int i = 0; i < int(rem_elems.size()); i++) {
            auto &e = rem_elems[i];
            if (e == 1) continue;
            if (b.block > e) {
                // Try to split this block.
                int next_block = utils::max_div(b.block, e);
                if (next_block == 1) return layout_t();
                return tmp.split_block(eb, next_block, b.block / next_block)
                        .split_into_multi_blocks_impl(
                                multi_blocks, out_multi_blocks);
            }
            if (e % b.block != 0) {
                if (!allow_smaller_blocks) return layout_t();
            }
            e /= b.block;
            cur_elems[i] *= b.block;
            break;
        }
    }
    for (int i = 0; i < int(cur_elems.size()); i++) {
        if (cur_elems[i] != multi_blocks[i]) {
            if (!allow_smaller_blocks) return layout_t();
        }
        if (out_multi_blocks) (*out_multi_blocks)[i] = cur_elems[i];
    }
    return tmp;
}

expr_t grid_splitter_t::pop_block(int size) {
    ir_assert(size > 1);
    ir_assert(can_pop_block(size));

    int new_stride = cur_stride_ * size;

    auto idx_expr = grid_.idx(cur_idx_);
    if (cur_stride_ != 1) idx_expr /= cur_stride_;
    if (new_stride != grid_.dim(cur_idx_)) idx_expr %= size;

    cur_stride_ = new_stride;
    if (cur_stride_ == grid_.dim(cur_idx_)) {
        // Move to the next dimension.
        cur_idx_--;
        skip_size_1_dims();
        cur_stride_ = 1;
    }
    return idx_expr;
}

stride_t tdim_info_t::compute_stride(
        const expr_t &e, int idx, const expr_t &var) {
    // e == var -> fixed stride.
    if (e.is_same(var)) return stride_t(1);

    auto vars = find_objects<var_t>(e);

    auto e0 = e;
    auto e1 = substitute(e, var, var + 1);
    auto e_stride = simplify(e1 - e0);

    if (is_const(e_stride)) return stride_t(to_cpp<dim_t>(e_stride));

    // Stride is not a constant.
    return stride_t::unknown();
}

view_t view_t::create_sub_view(const tensor_t &sub_tensor) const {
    ir_assert(sub_tensor.ndims() == nvdims()) << "Dimensions don't match.";

    auto ret = *this;
    ret.vdims_ = sub_tensor.dims();
    for (int i = 0; i < nvdims(); i++) {
        auto &i_start = sub_tensor.start()[i];
        if (is_zero(i_start)) continue;
        auto &s = ret.vstart_[i];
        s += i_start;
        s = simplify(s);
    }
    return ret;
}

view_t view_t::substitute(const expr_t &from, const expr_t &to) const {
    view_t ret = *this;
    for (int i = 0; i < nvdims(); i++) {
        ret.vstart_[i] = jit::substitute(ret.vstart_[i], from, to);
        ret.vstart_[i] = simplify(ret.vstart_[i]);
    }
    return ret;
}

std::vector<expr_t> view_t::create_vvars(int nvdims) {
    static const int max_nvdims = 128;
    static std::vector<expr_t> _vvars;
    static std::once_flag initialized;
    std::call_once(initialized, [&]() {
        for (int i = 0; i < max_nvdims; i++)
            _vvars.push_back(
                    var_t::make(type_t::s32(), "_" + std::to_string(i)));
    });

    ir_assert(nvdims <= max_nvdims) << "Too many dimensions: " << nvdims;
    return std::vector<expr_t>(_vvars.begin(), _vvars.begin() + nvdims);
}

layout_t view_t::create_pseudo_vlayout(const layout_t &tlayout) const {
    ir_assert(!tlayout.is_empty());

    std::vector<dim_t> rem_vdims = vdims_;
    std::vector<block_t> blocks;

    for (auto &teb : tlayout.enumerated_blocks()) {
        block_t &tb = teb.second;
        bool tb_is_outermost = tlayout.is_outermost(teb);
        dim_t tblock = tb.block;

        auto &tinfo = tdims_[tb.dim_idx];
        if (tb_is_outermost) {
            bool is_first = true;
            for (int i = tinfo.nvargs() - 1; i >= 0; i--) {
                int vidx = tinfo.vidx(i);
                if (rem_vdims[vidx] == 1) continue;

                // When expression contains 2+ variables, use unknown
                // stride unless the view variable is the innermost.
                stride_t stride
                        = (is_first ? tinfo.vstride(i) : stride_t::unknown());
                blocks.emplace_back(
                        vidx, rem_vdims[vidx], stride * stride_t(tb.stride));
                rem_vdims[vidx] = 1;
                is_first = false;
            }
            continue;
        }

        ir_assert(tinfo.is_identity()) << "Can't create pseudo-layout.";

        int vidx = tinfo.vidx(0);
        dim_t &rem_vdim = rem_vdims[vidx];
        if (rem_vdim == 1) continue;

        if (tb_is_outermost) {
            tblock = rem_vdim;
            rem_vdim = 1;
        } else if (rem_vdim % tblock == 0) {
            rem_vdim /= tblock;
        } else if (rem_vdim % tblock != 0) {
            // Try to split the current block and start from scratch.
            if (tblock % rem_vdim == 0) {
                auto tmp_layout
                        = tlayout.split_block(teb, rem_vdim, tblock / rem_vdim);
                return create_pseudo_vlayout(tmp_layout);
            }

            ir_error_not_expected() << "Can't create pseudo-layout.";
        }
        blocks.emplace_back(tb.dim_idx, tblock, tb.stride);
    }

    for (auto &d : rem_vdims) {
        ir_assert(d == 1) << "Can't create pseudo-layout.";
        MAYBE_UNUSED(d);
    }

    return layout_t(tlayout.type(), nvdims(), 0, blocks);
}

layout_t dim_assignment_t::map(const layout_t &layout) const {
    std::vector<block_t> new_blocks;
    for (auto &b : layout.blocks()) {
        int new_idx = assignments_[b.dim_idx];
        if (new_idx == -1) continue; // Drop this block.
        auto new_b = b;
        new_b.dim_idx = new_idx;
        new_blocks.push_back(new_b);
    }
    new_blocks = layout_t::normalize_blocks(new_ndims(), new_blocks,
            /*keep_size_1_blocks=*/true);
    auto ret = layout_t(layout.type(), new_ndims(), layout.offset(), new_blocks,
            /*do_normalize=*/false);
    ir_assert(layout.elems() == ret.elems())
            << "Assignment doesn't preserve number of elements.";
    return ret;
}

// Adds size one spatial dimensions according to input parameters. Spatial
// dimensions are assumed to be the last dimensions.
layout_t normalize_conv_spatial(
        const layout_t &layout, int old_sp_ndims, bool reduced_to_1d) {
    int old_ndims = layout.ndims();
    int new_ndims = old_ndims - old_sp_ndims + 3;

    dim_assignment_t to_3d(old_ndims, new_ndims);
    for (int i = 0; i < old_ndims; i++) {
        if (i < old_ndims - old_sp_ndims) {
            // Non-spatial dimensions.
            to_3d.assign(i, i);
        } else {
            // Spatial dimensions.
            int sp_idx = 3 - (old_ndims - i);
            if (reduced_to_1d) sp_idx = 2;
            to_3d.assign(i, new_ndims - (3 - sp_idx));
        }
    }
    return to_3d.map(layout);
}

layout_t insert_dimension(const layout_t &layout, int dim_idx) {
    auto new_blocks = layout.blocks();
    for (auto &b : new_blocks) {
        if (b.dim_idx >= dim_idx) b.dim_idx++;
    }
    return layout_t(layout.type(), layout.ndims() + 1, layout.offset(),
            new_blocks,
            /*do_normalize=*/false);
}

layout_t remove_size_1_dimension(const layout_t &layout, int dim_idx) {
    ir_assert(0 <= dim_idx && dim_idx < layout.ndims());
    ir_assert(layout.dim(dim_idx) == 1);
    dim_assignment_t a(layout.ndims(), layout.ndims() - 1);
    for (int i = 0; i < layout.ndims(); i++) {
        if (i == dim_idx) continue;
        a.assign(i, i < dim_idx ? i : i - 1);
    }
    return a.map(layout);
}

layout_t split_dimension(
        const layout_t &_layout, int dim_idx, int outer_block) {
    int rem_inner_block
            = ir_utils::safe_divide(_layout.dim(dim_idx), outer_block);
    auto layout = insert_dimension(_layout, dim_idx);
    std::vector<block_t> new_blocks;
    for (auto &eb : layout.enumerated_blocks()) {
        auto &b = eb.second;
        if (b.dim_idx != dim_idx + 1) {
            new_blocks.push_back(b);
            continue;
        }
        if (b.block % rem_inner_block == 0) {
            new_blocks.emplace_back(dim_idx + 1, rem_inner_block, b.stride);
            new_blocks.emplace_back(dim_idx, b.block / rem_inner_block,
                    dim_t(b.stride) * rem_inner_block);
            rem_inner_block = 1;
        } else {
            new_blocks.push_back(b);
            rem_inner_block = ir_utils::safe_divide(rem_inner_block, b.block);
        }
    }

    // Remove inner blocks with size one.
    std::vector<block_t> _new_blocks;
    std::vector<bool> seen(layout.ndims());
    for (auto it = new_blocks.rbegin(); it != new_blocks.rend(); ++it) {
        if (it->block == 1 && seen[it->dim_idx]) continue;
        _new_blocks.push_back(*it);
        seen[it->dim_idx] = true;
    }
    std::reverse(_new_blocks.begin(), _new_blocks.end());
    return layout_t(layout.type(), layout.ndims(), layout.offset(), _new_blocks,
            /*do_normalize=*/false);
}

layout_t normalize_conv_groups(const layout_t &layout, bool with_groups,
        int groups, bool is_dw, bool add_groups, bool is_wei) {
    if (with_groups == add_groups) return layout;
    if (is_wei) {
        ir_assert(groups == 1)
                << "Adding/removing groups can be done only for single group.";
        if (add_groups) return insert_dimension(layout, 0);
        return remove_size_1_dimension(layout, 0);
    }

    ir_assert(!with_groups) << "Unexpected groups in source/destination.";
    if (is_dw) groups = layout.dim(1);
    return split_dimension(layout, /*dim_idx=*/1, groups);
}

layout_t normalize_conv_layout(const layout_t &_layout, bool with_groups,
        int groups, bool is_dw, bool reduced_to_1d, bool add_groups,
        bool is_wei) {
    int old_sp_ndims = _layout.ndims() - (with_groups ? 3 : 2);

    layout_t layout = _layout;
    layout = normalize_conv_spatial(layout, old_sp_ndims, reduced_to_1d);
    layout = normalize_conv_groups(
            layout, with_groups, groups, is_dw, add_groups, is_wei);

    return layout;
}

std::vector<dim_t> normalize_conv_dims(std::vector<dim_t> &dims,
        bool with_groups, int groups, bool is_dw, bool reduced_to_1d,
        bool add_groups, bool is_wei) {
    layout_t dummy_layout(type_t::u8(), 0, dims);
    return normalize_conv_layout(dummy_layout, with_groups, groups, is_dw,
            reduced_to_1d, add_groups, is_wei)
            .dims();
}

void normalize_conv_layouts(layout_t &src_layout, layout_t &wei_layout,
        layout_t &dst_layout, bool with_groups, int groups, bool is_dw,
        bool reduced_to_1d, bool add_groups) {
    src_layout = normalize_conv_layout(src_layout, /*with_groups=*/false,
            groups, is_dw, reduced_to_1d, add_groups, /*is_wei=*/false);
    wei_layout = normalize_conv_layout(wei_layout, with_groups, groups, is_dw,
            reduced_to_1d, add_groups, /*is_wei=*/true);
    dst_layout = normalize_conv_layout(dst_layout, /*with_groups=*/false,
            groups, is_dw, reduced_to_1d, add_groups, /*is_wei=*/false);
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
