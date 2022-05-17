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

#ifndef GPU_JIT_CONV_TENSOR_HPP
#define GPU_JIT_CONV_TENSOR_HPP

#include <algorithm>
#include <array>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>
#include <unordered_map>

#include "common/memory_desc_wrapper.hpp"
#include "gpu/jit/conv/ir.hpp"
#include "gpu/jit/conv/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class tensor_t {
public:
    tensor_t() = default;

    tensor_t(const std::vector<dim_t> &dims)
        : tensor_t(dims, std::vector<expr_t>()) {}

    tensor_t(const std::vector<dim_t> &dims, const std::vector<expr_t> &start)
        : dims_(dims), start_(start) {
        if (start_.empty()) start_.resize(dims.size(), 0);
    }

    tensor_t(const std::vector<dim_t> &dims, const std::vector<dim_t> &start)
        : tensor_t(dims) {
        start_.resize(start.size());
        for (size_t i = 0; i < start.size(); i++)
            start_[i] = start[i];
    }

    dim_t operator()(int idx) const { return dims_[idx]; }

    const expr_t &start(int idx) const { return start_[idx]; }

    int ndims() const { return int(dims_.size()); }

    dim_t elems() const {
        dim_t ret = 1;
        for (int i = 0; i < ndims(); i++)
            ret *= dims_[i];
        return ret;
    }

    const std::vector<dim_t> &dims() const { return dims_; }

    const std::vector<expr_t> &start() const { return start_; }

    bool is_empty() const { return dims_.empty(); }

    bool is_equal(const tensor_t &other) const {
        if (ndims() != other.ndims()) return false;
        for (int i = 0; i < ndims(); i++) {
            if (dims_[i] != other.dims_[i]) return false;
            if (!start_[i].is_equal(other.start_[i])) return false;
        }
        return true;
    }

    std::string str() const {
        using ir_utils::operator<<;

        if (is_empty()) return "(nil)";
        std::ostringstream oss;
        oss << ir_utils::make_seq_print_helper(dims_, "x");
        if (!has_zero_start()) oss << " start: [" << start_ << "]";
        return oss.str();
    }

    IR_DEFINE_DUMP()

    bool has_zero_start() const {
        for (int i = 0; i < ndims(); i++)
            if (!is_zero(start_[i])) return false;
        return true;
    }

    dim_t to_1d_offset(const std::vector<dim_t> &args) const {
        ir_assert(has_zero_start());

        dim_t off = 0;
        for (int i = 0; i < ndims(); i++) {
            off *= dims_[i];
            off += args[i];
        }
        return off;
    }

    tensor_t create_sub_tensor(const tensor_t &tile) const {
        ir_assert(ndims() == tile.ndims()) << "Incompatible sizes.";
        std::vector<expr_t> new_start = start_;
        for (int i = 0; i < ndims(); i++)
            new_start[i] += tile.start(i);
        return tensor_t(tile.dims(), new_start);
    }

    tensor_t substitute(const expr_t &from, const expr_t &to) const {
        tensor_t ret = *this;
        for (int i = 0; i < ndims(); i++) {
            ret.start_[i] = jit::substitute(ret.start_[i], from, to);
            ret.start_[i] = simplify(ret.start_[i]);
        }
        return ret;
    }

private:
    std::vector<dim_t> dims_;
    std::vector<expr_t> start_;
};

inline std::ostream &operator<<(std::ostream &out, const tensor_t &tensor) {
    out << tensor.str();
    return out;
}

class grid_info_t {
public:
    grid_info_t() = default;
    grid_info_t(int ndims) : dims_(ndims), offs_(ndims), idxs_(ndims) {}
    grid_info_t(const std::vector<int> &dims, const std::vector<expr_t> &idxs)
        : grid_info_t(dims, {}, idxs) {}
    grid_info_t(const std::vector<int> &dims, const std::vector<int> &offs,
            const std::vector<expr_t> &idxs)
        : dims_(dims), offs_(offs), idxs_(idxs) {
        if (offs_.empty()) offs_.resize(dims.size());
        ir_assert(dims_.size() == offs_.size());
        ir_assert(dims_.size() == idxs_.size());
    }

    bool operator==(const grid_info_t &other) const {
        if (ndims() != other.ndims()) return false;
        for (int i = 0; i < ndims(); i++) {
            if (dim(i) != other.dim(i)) return false;
            if (off(i) != other.off(i)) return false;
            if (!idx(i).is_equal(other.idx(i))) return false;
        }
        return true;
    }

    bool is_empty() const { return dims_.empty(); }

    int &dim(int dim_idx) { return dims_[dim_idx]; }
    int &off(int dim_idx) { return offs_[dim_idx]; }
    expr_t &idx(int dim_idx) { return idxs_[dim_idx]; }
    int dim_idx(const expr_t &idx_var) const {
        for (int i = 0; i < ndims(); i++) {
            if (idx(i).is_same(idx_var)) return i;
        }
        ir_error_not_expected() << "Index not found: " << idx_var;
        return -1;
    }

    const int &dim(int dim_idx) const { return dims_[dim_idx]; }
    const int &dim(const expr_t &idx_var) const {
        return dims_[dim_idx(idx_var)];
    }
    const int &off(int dim_idx) const { return offs_[dim_idx]; }
    const expr_t &idx(int dim_idx) const { return idxs_[dim_idx]; }

    int ndims() const { return int(dims_.size()); }
    int elems() const {
        return utils::array_product(dims_.data(), dims_.size());
    }

    grid_info_t sub_grid(std::initializer_list<int> old_dim_idxs) const {
        grid_info_t ret(int(old_dim_idxs.size()));
        int new_dim_idx = 0;
        for (auto old_dim_idx : old_dim_idxs) {
            ret.dim(new_dim_idx) = dim(old_dim_idx);
            ret.off(new_dim_idx) = off(old_dim_idx);
            ret.idx(new_dim_idx) = idx(old_dim_idx);
            new_dim_idx++;
        }
        return ret;
    }

    grid_info_t slice(int dim_idx, int new_off, int new_dim,
            const expr_t &new_idx, expr_t &new_idx_value) const {
        ir_assert(dim_idx >= 0 && dim_idx < ndims());
        ir_assert(new_dim > 0 && new_off >= 0);
        ir_assert(new_off + new_dim <= dims_[dim_idx]);

        grid_info_t ret = *this;
        ret.offs_[dim_idx] += new_off;
        ret.dims_[dim_idx] = new_dim;
        if (new_off > 0) {
            new_idx_value = ret.idxs_[dim_idx] - new_off;
            ret.idxs_[dim_idx] = new_idx;
        } else {
            new_idx_value = expr_t();
        }
        ret.parent_dims_ = (parent_dims_.empty() ? dims_ : parent_dims_);
        return ret;
    }

    grid_info_t halven(const expr_t &new_idx, int &dim_idx,
            expr_t &new_idx_value, bool first = true) const {
        for (int i = ndims() - 1; i >= 0; i--) {
            if (dim(i) == 1 || dim(i) % 2 != 0) continue;
            dim_idx = i;
            if (first) return slice(i, 0, dim(i) / 2, new_idx, new_idx_value);
            return slice(i, dim(i) / 2, dim(i) / 2, new_idx, new_idx_value);
        }
        return grid_info_t();
    }

    expr_t slice_condition() const {
        if (parent_dims_.empty()) return expr_t();
        expr_t ret(true);
        for (int i = 0; i < ndims(); i++) {
            auto &idx = idxs_[i];
            if (offs_[i] > 0) ret &= (idx >= 0);
            if (offs_[i] + dims_[i] < parent_dims_[i]) ret &= (idx < dims_[i]);
        }
        if (ret.is_equal(expr_t(true))) return expr_t();
        return ret;
    }

    std::string str() const {
        std::ostringstream oss;
        oss << ir_utils::make_seq_print_helper(dims_, "x");
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    std::vector<int> dims_;
    std::vector<int> offs_;
    std::vector<expr_t> idxs_;

    std::vector<int> parent_dims_;
};

inline std::ostream &operator<<(
        std::ostream &out, const grid_info_t &grid_info) {
    out << grid_info.str();
    return out;
}

class grid_splitter_t {
public:
    grid_splitter_t(const grid_info_t &grid)
        : grid_(grid), cur_idx_(grid.ndims() - 1), cur_stride_(1) {
        skip_size_1_dims();
        ir_assert(cur_idx_ >= 0);
    }

    int cur_block() const {
        if (is_empty()) return 1;

        return grid_.dim(cur_idx_) / cur_stride_;
    }

    bool is_empty() const { return cur_idx_ == -1; }

    bool can_pop_block(int size) const {
        if (is_empty()) return false;
        return cur_block() % size == 0;
    }

    expr_t pop_block(int size);

private:
    void skip_size_1_dims() {
        while (cur_idx_ >= 0 && grid_.dim(cur_idx_) == 1)
            cur_idx_--;
    }

    grid_info_t grid_;

    int cur_idx_;
    int cur_stride_;
};

enum class stride_kind_t {
    undef,
    fixed,
    unknown,
};

class stride_t {
public:
    stride_t() = default;

    stride_t(dim_t stride) : stride_t(stride_kind_t::fixed, stride) {}

    bool operator==(const stride_t &other) const {
        return (kind_ == other.kind_) && (stride_ == other.stride_);
    }

    bool operator!=(const stride_t &other) const { return !operator==(other); }

    size_t get_hash() const { return ir_utils::get_hash(kind_, stride_); }

    operator dim_t() const {
        ir_assert(kind_ == stride_kind_t::fixed);
        return stride_;
    }

    bool is_fixed() const { return kind_ == stride_kind_t::fixed; }

    bool is_unknown() const { return kind_ == stride_kind_t::unknown; }

    stride_t &operator*=(const stride_t &other) {
        if (is_fixed() && other.is_fixed()) {
            stride_ *= other.stride_;
        } else {
            set_unknown();
        }
        return *this;
    }

    stride_t &operator/=(const stride_t &other) {
        if (is_fixed() && other.is_fixed()) {
            stride_ /= other.stride_;
        } else {
            set_unknown();
        }
        return *this;
    }

    std::string str() const {
        std::ostringstream oss;
        if (is_fixed()) {
            oss << stride_;
        } else {
            oss << "(unknown)";
        }
        return oss.str();
    }

    IR_DEFINE_DUMP()

    static stride_t unknown() { return stride_t(stride_kind_t::unknown); }

private:
    stride_t(stride_kind_t kind, dim_t stride = 0)
        : kind_(kind), stride_(stride) {}

    void set_unknown() {
        kind_ = stride_kind_t::unknown;
        stride_ = 0;
    }

    stride_kind_t kind_ = stride_kind_t::undef;
    dim_t stride_ = 0;
};

inline std::ostream &operator<<(std::ostream &out, const stride_t &stride) {
    out << stride.str();
    return out;
}

inline stride_t operator*(const stride_t &a, const stride_t &b) {
    stride_t tmp = a;
    return tmp *= b;
}

inline stride_t operator*(const stride_t &a, dim_t b) {
    return a * stride_t(b);
}

inline stride_t operator*(dim_t a, const stride_t &b) {
    return stride_t(a) * b;
}

struct block_t {
    block_t() = default;

    block_t(int dim_idx, dim_t block, const stride_t &stride)
        : dim_idx(dim_idx), block(block), stride(stride) {}

    bool is_equal(const block_t &other) const {
        return (dim_idx == other.dim_idx) && (block == other.block)
                && (stride == other.stride);
    }

    size_t get_hash() const {
        return ir_utils::get_hash(dim_idx, block, stride);
    }

    std::string str() const {
        std::ostringstream oss;
        oss << "block_t(dim_idx = " << dim_idx;
        oss << ", block = " << block;
        oss << ", stride = " << stride;
        oss << ")";
        return oss.str();
    }

    IR_DEFINE_DUMP()

    int dim_idx; // Dimension index.
    dim_t block; // Block size.
    stride_t stride; // Stride between elements of the block.
};

inline std::ostream &operator<<(std::ostream &out, const block_t &b) {
    out << b.str();
    return out;
}

class layout_t {
public:
    static const int max_ndims = 6;

    layout_t() : type_(type_t::undef()), ndims_(0), offset_(0) {
        sanity_check();
    }

    layout_t(const type_t &type, const expr_t &offset,
            const std::string &format, const std::vector<dim_t> &dims = {},
            bool do_normalize = true);

    layout_t(const memory_desc_wrapper &mdw, const std::string &format,
            bool do_normalize = true)
        : layout_t(mdw.data_type(), mdw.offset0(), format,
                std::vector<dim_t>(
                        mdw.padded_dims(), mdw.padded_dims() + mdw.ndims()),
                do_normalize) {}

    layout_t(const memory_desc_wrapper &mdw, const char *format,
            bool do_normalize = true)
        : layout_t(mdw, std::string(format), do_normalize) {}

    layout_t(const memory_desc_wrapper &mdw, bool do_normalize = true);

    layout_t(const type_t &type, const expr_t &offset,
            const std::vector<dim_t> &dims, bool do_normalize = true)
        : type_(type), ndims_(int(dims.size())), offset_(offset) {
        dim_t stride = 1;
        for (int i = ndims_ - 1; i >= 0; i--) {
            blocks_.emplace_back(i, dims[i], stride);
            stride *= dims[i];
        }
        if (do_normalize) blocks_ = normalize_blocks(ndims_, blocks_);
        sanity_check();
    }

    layout_t(const type_t &type, int ndims, const expr_t &offset,
            const std::vector<block_t> &blocks, bool do_normalize = true)
        : type_(type), ndims_(ndims), offset_(offset), blocks_(blocks) {
        if (do_normalize) blocks_ = normalize_blocks(ndims_, blocks_);
        sanity_check();
    }

    layout_t(const type_t &type, const expr_t &offset, const layout_t &other,
            bool do_normalize)
        : layout_t(type, other.ndims(), offset, other.blocks(), do_normalize) {}

    bool is_empty() const { return ndims_ == 0; }

    int ndims() const { return ndims_; }

    dim_t elems() const {
        dim_t ret = 1;
        for (auto &b : blocks_)
            ret *= b.block;
        return ret;
    }

    // Storage size in bytes.
    dim_t size() const {
        if (is_empty()) return 0;
        dim_t max_stride = 1;
        for (auto &b : blocks_) {
            max_stride = std::max(max_stride, dim_t(b.block * b.stride));
        }
        return max_stride * type().size();
    }

    template <typename T = expr_t>
    T offset(
            const std::vector<T> &args = {}, bool ignore_offset = false) const {
        if (args.empty()) return expr_cast<T>(offset_);

        ir_assert(int(args.size()) == ndims()) << "Dimensions do not match.";

        T off = 0;
        auto _args = args;
        for (auto &eb : enumerated_blocks()) {
            auto &b = eb.second;
            auto &idx = _args[b.dim_idx];
            if (ir_utils::is_equal(idx, T(0))) continue;

            // Do not use modulus for outermost blocks.
            auto i = is_outermost(eb) ? idx : (idx % b.block);
            off = i * dim_t(b.stride) + off;
            idx /= b.block;
        }
        if (ignore_offset) return off;

        T off0 = expr_cast<T>(offset_);
        return off0 + off;
    }

    const type_t &type() const { return type_; }

    std::vector<dim_t> dims() const {
        std::vector<dim_t> dims(ndims(), 1);
        for (auto &b : blocks_) {
            dims[b.dim_idx] *= b.block;
        }
        return dims;
    }

    dim_t dim(int dim_idx) const {
        dim_t ret = 1;
        for (auto &b : blocks_) {
            if (b.dim_idx == dim_idx) ret *= b.block;
        }
        return ret;
    }

    const std::vector<block_t> &blocks() const { return blocks_; }

    void set_offset(const expr_t &offset) { offset_ = offset; }

    bool is_strictly_equal(
            const layout_t &other, bool compare_offset = true) const {
        if (!type_.is_equal(other.type_)) return false;
        if (compare_offset && !offset_.is_equal(other.offset_)) return false;
        if (!ir_utils::is_equal(blocks_, other.blocks_)) return false;
        return true;
    }

    bool operator==(const layout_t &other) const { return is_equal(other); }

    bool operator!=(const layout_t &other) const { return !operator==(other); }

    bool is_equal(const layout_t &other, bool compare_offset = true) const {
        return normalize().is_strictly_equal(other.normalize(), compare_offset);
    }

    size_t get_hash() const {
        return ir_utils::get_hash(type_, ndims_, offset_, blocks_);
    }

    template <typename T>
    T operator()(const std::vector<T> &args) const {
        return offset(args);
    }

    template <typename T = expr_t>
    T offset_in_bytes(
            const std::vector<T> &args = {}, bool ignore_offset = false) const {
        return offset(args, ignore_offset) * type().size();
    }

    std::string desc_str(bool dnnl_style = false) const {
        if (is_empty()) return "(nil)";
        if (!dnnl_style && blocks_.empty()) return "(scalar)";
        std::string ret;
        stride_t dense_stride(1);
        std::vector<bool> seen(ndims());
        for (auto &eb : enumerated_blocks()) {
            auto &b = eb.second;
            std::string b_str;
            if (dnnl_style && is_outermost(eb)) {
                b_str.append(1, (seen[b.dim_idx] ? 'A' : 'a') + b.dim_idx);
            } else {
                b_str = std::to_string(b.block);
                b_str.append(1, 'a' + b.dim_idx);
            }
            if (!dnnl_style) {
                if (b.stride.is_unknown()) {
                    b_str.append(1, '?');
                } else if (b.stride != dense_stride) {
                    b_str.append(1, '*');
                }
            }
            ret = b_str + ret;
            dense_stride = b.stride * b.block;
            seen[b.dim_idx] = true;
        }
        return ret;
    }

    std::string str() const {
        if (is_empty()) return "(nil)";
        std::ostringstream oss;
        oss << desc_str();
        if (!has_zero_offset()) oss << " offset: " << offset_;
        return oss.str();
    }

    IR_DEFINE_DUMP()

    memory_desc_t to_dnnl(const dim_t *dims_hint) const;

    // Returns a vector of <block index, block> pairs.
    // The innermost block (first) has index 0.
    std::vector<std::pair<int, block_t>> enumerated_blocks() const {
        std::vector<std::pair<int, block_t>> ret;
        for (int i = 0; i < int(blocks_.size()); i++) {
            ret.emplace_back(i, blocks_[i]);
        }
        return ret;
    }

    std::vector<dim_t> strides(int dim_idx) const {
        std::vector<dim_t> ret;
        for (auto &b : blocks_)
            if (b.dim_idx == dim_idx) ret.push_back(b.stride);
        return ret;
    }

    // eb is <block index, block> pair, see enumerated_blocks().
    bool is_outermost(const std::pair<int, block_t> &eb) const {
        return is_outermost(eb, blocks_);
    }

    bool is_plain() const {
        std::vector<bool> seen(ndims());
        for (auto &b : blocks_) {
            if (seen[b.dim_idx]) return false;
            seen[b.dim_idx] = true;
        }
        return true;
    }

    bool has_zero_offset() const { return offset_.is_equal(expr_t(0)); }

    bool has_unknown_strides() const {
        for (auto &b : blocks_)
            if (b.stride.is_unknown()) return true;
        return false;
    }

    // Returns a canonical representation of the layout:
    // - Consecutive dense blocks are merged
    // - Size one blocks are:
    //   - Removed (if keep_size_1_blocks is false)
    //   - Reordered according to the heuristic (if keep_size_1_blocks is true)
    // Optionally removes size one blocks and merges consecutive dense blocks
    // representing the same dimension.
    layout_t normalize(bool keep_size_1_blocks = false) const {
        auto blocks = normalize_blocks(ndims(), blocks_, keep_size_1_blocks);
        return layout_t(type(), ndims(), offset(), blocks);
    }

    layout_t transpose() const {
        if (ndims() != 2) ir_error_not_expected();

        // Flip: 0 -> 1, 1 -> 0.
        auto blocks = blocks_;
        for (auto &b : blocks)
            b.dim_idx ^= 1;

        return layout_t(type(), ndims(), offset(), blocks);
    }

    // Returns a new (sub-)layout that fully contains the passed sub-tensor.
    // Strides are kept unchanged.
    // Assumption: the original layout can be tiled by the passed sub-tensor.
    // For example: XaYb4a2b can be tiled into 2x2 sub-tensors but it's not
    // possible to tile it into 3x2 sub-tensors.
    layout_t map(const tensor_t &tensor) const;

    layout_t reinterpret(
            const type_t &new_type, bool do_normalize = true) const;

    layout_t retype(const type_t &new_type) const {
        auto ret = *this;
        ret.type_ = new_type;
        return ret;
    }

    bool is_dense() const {
        stride_t stride = 1;
        for (auto &b : blocks_) {
            if (b.stride != stride) return false;
            stride *= b.block;
        }
        return true;
    }

    // Returns a packed layout where all blocks are contiguous, without gaps.
    layout_t make_dense() const {
        dim_t stride = 1;
        auto new_blocks = blocks_;
        for (auto &b : new_blocks) {
            b.stride = stride;
            stride *= b.block;
        }
        return layout_t(type(), ndims(), 0, new_blocks);
    }

    layout_t make_strided(int _stride) const {
        stride_t stride = _stride;
        auto new_blocks = blocks_;
        for (auto &b : new_blocks) {
            b.stride = stride;
            stride *= b.block;
        }
        return layout_t(type(), ndims(), 0, new_blocks);
    }

    // Returns an equivalent layout where the specified block is split into two.
    // block0 - inner block size.
    // block1 - outer block size.
    layout_t split_block(const std::pair<int, block_t> &eb, dim_t block0,
            dim_t block1) const;

    // Splits blocks so that they can be used to form `multi_blocks` without
    // crossing the block boundaries. `multi_blocks` are ordered from innermost
    // to outermost. Returns an empty layout if such a split is not possible.
    // Example (all blocks are ordered from innermost to outermost):
    //     Input blocks:  [4, 4, 2]
    //     Multi-blocks:  [8, 2]
    //     Output blocks: [4, 2, 2, 2]
    layout_t split_into_multi_blocks(
            const std::vector<dim_t> &multi_blocks) const;

    layout_t split_into_multi_blocks_with_hint(
            std::vector<dim_t> &multi_blocks) const;

    layout_t add_outer_block(
            int dim_idx, dim_t block, dim_t stride = -1) const {
        if (stride == -1) stride = elems();
        ir_assert(stride >= elems());
        ir_assert(dim_idx < ndims());
        auto new_blocks = blocks();
        new_blocks.emplace_back(dim_idx, block, stride);
        return layout_t(type(), ndims(), offset(), new_blocks);
    }

    tensor_t split_into_dense_tile(dim_t tile_elems, dim_t outer_block) const;

    // Returns a tensor corresponding to the biggest innermost sub-layout so that
    // 1) It consists of consecutive blocks only.
    // 2) It contains less or equal than max_tile_elems elements.
    // 3) It is dense if is_dense_tile is true.
    tensor_t split_into_max_tile(
            dim_t max_tile_elems, bool is_dense_tile) const;

    tensor_t split(const grid_info_t &grid) const {
        std::vector<dim_t> tile_dims(ndims(), 1);
        ir_assert(elems() % grid.elems() == 0) << "Can't split across grid.";

        dim_t cur_elems_per_tile = 1;
        dim_t elems_per_tile = elems() / grid.elems();
        for (auto &b : blocks()) {
            dim_t block
                    = std::min(b.block, elems_per_tile / cur_elems_per_tile);
            tile_dims[b.dim_idx] *= block;
            cur_elems_per_tile *= block;
        }
        ir_assert(cur_elems_per_tile == elems_per_tile)
                << "Can't split across grid.";

        return split(tensor_t(tile_dims), grid);
    }

    tensor_t split(const tensor_t &tile, const grid_info_t &grid,
            std::vector<block_t> *outer_blocks = nullptr) const {
        ir_assert(ndims() == tile.ndims())
                << "Number of dimensions doesn't match.";
        ir_assert(tile.has_zero_start());

        if (outer_blocks) outer_blocks->resize(0);

        if (grid.elems() == 1) return tile;

        dim_t total_elems = elems();
        dim_t tile_elems = tile.elems();

        grid_splitter_t grid_splitter(grid);
        ir_assert(tile_elems * grid.elems() == total_elems)
                << "Tile/grid dimensions do not match.";
        MAYBE_UNUSED(total_elems);
        MAYBE_UNUSED(tile_elems);

        std::vector<dim_t> dims(tile.ndims(), 1);
        std::vector<expr_t> start(tile.ndims(), 0);
        std::vector<dim_t> rem_dims = tile.dims();
        for (auto &eb : enumerated_blocks()) {
            auto &b = eb.second;
            if (b.block == 1) continue;

            dim_t &e = rem_dims[b.dim_idx];
            if (e > 1) {
                if (e % b.block == 0) {
                    e /= b.block;
                } else if (b.block % e == 0) {
                    auto tmp_layout = split_block(eb, e, b.block / e);
                    return tmp_layout.split(tile, grid, outer_blocks);
                } else {
                    ir_error_not_expected() << "Can't split across grid.";
                }
            } else {
                dim_t next_chunk
                        = math::gcd(b.block, grid_splitter.cur_block());
                if (b.block == next_chunk) {
                    auto idx = grid_splitter.pop_block(next_chunk);
                    start[b.dim_idx] += idx * dims[b.dim_idx];
                    if (outer_blocks) outer_blocks->push_back(b);
                } else if (b.block % next_chunk == 0) {
                    auto tmp_layout
                            = split_block(eb, next_chunk, b.block / next_chunk);
                    return tmp_layout.split(tile, grid, outer_blocks);
                } else {
                    ir_error_not_expected() << "Can't split across grid.";
                }
            }
            dims[b.dim_idx] *= b.block;
        }
        return tensor_t(tile.dims(), start);
    }

    // Iterates through tiles of the layout, calling `f` with relative offsets
    // for each tile. The iteration order is defined by the layout blocks -
    // absolute 1D offsets are increasing between callback calls.
    template <typename F>
    void for_each_tile(const tensor_t &tile, const F &f) const {
        ir_assert(tile.ndims() == ndims());
        ir_assert(tile.has_zero_start());
        for (int i = 0; i < ndims(); i++) {
            ir_assert(dim(i) % tile.dims()[i] == 0);
        }

        int nblocks = int(blocks().size());
        std::vector<dim_t> sub_blocks(nblocks);
        for (int i = 0; i < nblocks; i++)
            sub_blocks[i] = blocks()[i].block;

        for (int i = 0; i < ndims(); i++) {
            dim_t dim = tile.dims()[i];
            for (auto &eb : enumerated_blocks()) {
                auto &b = eb.second;
                if (b.dim_idx != i) continue;
                int block_idx = eb.first;
                if (b.block >= dim) {
                    ir_assert(b.block % dim == 0);
                    sub_blocks[block_idx] = b.block / dim;
                    break;
                }
                sub_blocks[block_idx] = 1;
                ir_assert(dim % b.block == 0);
                dim /= b.block;
            }
        }

        int ntiles = int(elems() / tile.elems());

        std::vector<dim_t> sub_block_idxs(nblocks);
        for (int i = 0; i < ntiles; i++) {
            // Convert sub-block indices to dimension indices.
            std::vector<dim_t> dims(ndims(), 1);
            std::vector<dim_t> start(ndims());
            for (int j = 0; j < nblocks; j++) {
                auto &b = blocks()[j];
                dim_t k = sub_block_idxs[j]
                        * (blocks()[j].block / sub_blocks[j]);
                start[b.dim_idx] += dims[b.dim_idx] * k;
                dims[b.dim_idx] *= b.block;
            }

            // Pass dimension offsets to the callback.
            f(start);

            // Move to the next vector of indices.
            for (int j = 0; j < nblocks; j++) {
                auto &idx = sub_block_idxs[j];
                if (idx + 1 < sub_blocks[j]) {
                    idx++;
                    break;
                }
                idx = 0;
            }
        }
    }

    // eb is <block index, block> pair, see enumerated_blocks().
    static bool is_outermost(const std::pair<int, block_t> &eb,
            const std::vector<block_t> &blocks) {
        int dim_idx = eb.second.dim_idx;
        for (int i = 0; i < int(blocks.size()); i++) {
            if (blocks[i].dim_idx == dim_idx && i > eb.first) return false;
        }
        return true;
    }

    // Assume that layouts are normalized.
    static void align_layouts(layout_t &a, layout_t &b);

    static std::vector<block_t> normalize_blocks(int ndims,
            const std::vector<block_t> &blocks,
            bool keep_size_1_blocks = false) {
        auto new_blocks = blocks;

        // Remove blocks of size 1.
        for (auto it = new_blocks.begin(); it != new_blocks.end();) {
            if (it->block == 1) {
                it = new_blocks.erase(it);
            } else {
                ++it;
            }
        }
        // Merge same dimension blocks.
        block_t prev_b;
        prev_b.dim_idx = -1;
        for (auto it = new_blocks.begin(); it != new_blocks.end();) {
            if (it->dim_idx == prev_b.dim_idx
                    && it->stride == (prev_b.stride * prev_b.block)) {
                auto &b = *(it - 1);
                b.block *= it->block;
                prev_b = b;
                it = new_blocks.erase(it);
            } else {
                prev_b = *it;
                ++it;
            }
        }
        // No need to keep size one blocks, return.
        if (!keep_size_1_blocks) return new_blocks;

        bool seen[max_ndims] = {false};
        for (auto &b : new_blocks)
            seen[b.dim_idx] = true;

        stride_t stride = (new_blocks.empty()
                        ? stride_t(1)
                        : new_blocks.back().stride * new_blocks.back().block);

        // Insert size one blocks according to the following heuristic:
        // TODO: Add documentation.
        for (int i = ndims - 1; i >= 0; i--) {
            if (seen[i]) continue;
            new_blocks.emplace_back(i, 1, stride);
        }

        return new_blocks;
    }

private:
    // Returns vector of <dimension index, block size> pairs.
    static std::vector<std::pair<int, dim_t>> parse_format(
            const std::string &format, int ndims_hint);

    // Returns vector of <dimension letter, block size> pairs.
    static std::vector<std::pair<char, dim_t>> parse_letter_blocks(
            const std::string &format);

    void sanity_check() const;

    layout_t split_into_multi_blocks_impl(
            const std::vector<dim_t> &multi_blocks,
            std::vector<dim_t> *out_multi_blocks) const;

    // Data type of the layout.
    type_t type_;

    // Number of dimensions.
    int ndims_;

    // Offset to the start of the layout (in elements of type).
    expr_t offset_;

    // Blocks ordered from innermost to outermost.
    std::vector<block_t> blocks_;
};

inline std::ostream &operator<<(std::ostream &out, const layout_t &layout) {
    out << layout.str();
    return out;
}

class mask_tensor_t {
public:
    mask_tensor_t() = default;

    mask_tensor_t(const layout_t &layout)
        : layout_(layout), masks_(layout.elems(), -1) {
        ir_assert(layout.is_dense());
    }

    mask_tensor_t(const layout_t &layout, const std::vector<int> &masks,
            const object_eq_map_t<expr_t, int> &mask2ids,
            const std::vector<expr_t> &id2masks)
        : layout_(layout)
        , masks_(masks)
        , mask2ids_(mask2ids)
        , id2masks_(id2masks) {
        ir_assert(int(masks.size()) == elems()) << "Incompatible size.";
    }

    const type_t &type() const { return layout_.type(); }

    const layout_t &layout() const { return layout_; }

    dim_t elems() const { return layout_.elems(); }

    void set_mask(dim_t off, const expr_t &mask) {
        ir_assert(0 <= off && off < elems()) << "Incorrect offset.";
        if (mask.is_empty()) return;

        auto ret = mask2ids_.insert({mask, int(mask2ids_.size())});
        int id = ret.first->second;
        masks_[off] = id;

        if (ret.second) id2masks_.push_back(mask);
    }

    const expr_t &mask(dim_t off) const {
        ir_assert(0 <= off && off < elems());
        return id2masks_[masks_[off]];
    }

    void simplify(const constraint_set_t &cset) {
        for (auto &mask : id2masks_) {
            auto new_mask = jit::simplify(mask, cset);
            // Some complex expressions need more than one simplify() call.
            int max_tries = 5;
            for (int i = 0; i < max_tries; i++) {
                mask = new_mask;
                new_mask = jit::simplify(new_mask, cset);
                if (new_mask.is_equal(mask)) break;
            }
        }
        mask2ids_.clear();
        for (int i = 0; i < int(id2masks_.size()); i++) {
            auto ret = mask2ids_.insert({id2masks_[i], i});
            if (!ret.second) {
                for (auto &m : masks_)
                    if (m == i) m = ret.first->second;
            }
        }
    }

    mask_tensor_t map(const tensor_t &tile) const {
        auto tile_start = expr_cast<dim_t>(tile.start());
        auto sub_layout = layout_.map(tensor_t(tile.dims()));
        mask_tensor_t sub_mask(sub_layout);
        ir_utils::for_each(
                tile.dims(), [&](const std::vector<dim_t> &sub_start) {
                    dim_t sub_off = sub_layout(sub_start);
                    dim_t off = layout_(tile_start) + layout_(sub_start);
                    sub_mask.set_mask(sub_off, mask(off));
                });
        return sub_mask;
    }

    mask_tensor_t reinterpret(const type_t &new_type) const {
        ir_assert(!is_empty()) << "Can't reinterpret.";
        dim_t bytes = elems() * type().size();
        if (bytes % new_type.size() != 0 && bytes > new_type.size())
            return mask_tensor_t();
        int new_mask_size = std::max((int)(bytes / new_type.size()), 1);
        std::vector<int> new_masks(new_mask_size);
        for (dim_t i = 0; i < bytes; i += new_type.size()) {
            int mask_id = std::numeric_limits<int>::max();
            for (int j = 0; j < new_type.size() && j < bytes; j++) {
                int cur_mask_id = masks_[(i + j) / type().size()];
                if (mask_id >= int(masks_.size())) {
                    mask_id = cur_mask_id;
                } else if (mask_id != cur_mask_id) {
                    // Mask is not consistent, can't reinterpret.
                    return mask_tensor_t();
                }
            }
            ir_assert(0 <= mask_id && mask_id < int(masks_.size()));
            new_masks[i / new_type.size()] = mask_id;
        }
        dim_t new_elmes = utils::div_up(bytes, new_type.size());
        layout_t _1d_layout(new_type, 0, std::vector<dim_t> {new_elmes});
        return mask_tensor_t(_1d_layout, new_masks, mask2ids_, id2masks_);
    }

    expr_t to_expr(int nmasks) const {
        if (elems() % nmasks != 0) return expr_t();

        std::vector<expr_t> vec(nmasks);
        for (int i = 0; i < elems(); i++) {
            auto &channel_mask = vec[i % nmasks];
            auto &cur_mask = id2masks_[masks_[i]];
            if (channel_mask.is_empty()) {
                channel_mask = cur_mask;
                continue;
            }
            if (!channel_mask.is_equal(cur_mask)) return expr_t();
        }
        auto e = shuffle_t::make(vec);
        e = jit::simplify(e);
        e = jit::simplify_propagate_shuffle(e);
        return e;
    }

    bool is_empty() const { return layout_.is_empty(); }

    std::string str() const {
        std::ostringstream oss;
        for (int i = 0; i < int(elems()); i++) {
            if (i != 0) oss << std::endl;
            oss << "mask #" << i << ": ";
            if (masks_[i] == -1) {
                oss << "(nil)";
            } else {
                oss << id2masks_[masks_[i]];
            }
        }
        return oss.str();
    }

    IR_DEFINE_DUMP()

private:
    layout_t layout_;
    std::vector<int> masks_;

    object_eq_map_t<expr_t, int> mask2ids_;
    std::vector<expr_t> id2masks_;
};

inline std::ostream &operator<<(
        std::ostream &out, const mask_tensor_t &mask_tensor) {
    out << mask_tensor.str();
    return out;
}

class tdim_info_t {
public:
    tdim_info_t() = default;

    tdim_info_t(const expr_t &expr, const expr_t &mask)
        : expr_(expr), mask_(mask) {}

    int nvargs() const { return nvargs_; }

    const expr_t &expr() const { return expr_; }

    const expr_t &mask() const { return mask_; }

    expr_t mask(const expr_t &tvalue, const std::vector<expr_t> &vvars,
            const std::vector<expr_t> &vvalues) const {
        auto ret = substitute(mask_, placeholder_var(), tvalue);
        for (int i = 0; i < int(vvars.size()); i++) {
            if (contains_object(ret, vvars[i])) {
                ret = substitute(ret, vvars[i], vvalues[i]);
            }
        }
        return ret;
    }

    int vidx(int arg_idx) const {
        ir_assert(arg_idx < nvargs());
        return vidxs_[arg_idx];
    }

    stride_t vstride(int arg_idx) const {
        ir_assert(arg_idx < nvargs());
        return vstrides_[arg_idx];
    }

    bool is_empty() const { return expr_.is_empty(); }

    bool is_identity() const { return is_var(expr_); }

    bool is_fixed_stride(int arg_idx) const {
        ir_assert(arg_idx < nvargs());
        return vstrides_[arg_idx].is_fixed();
    }

    void add_vvar(int vidx, const expr_t &varg) {
        ir_assert(nvargs_ + 1 <= max_nvargs);
        vidxs_[nvargs_] = vidx;
        vstrides_[nvargs_] = compute_stride(expr_, nvargs_, varg);
        nvargs_++;
    }

    static const expr_t &placeholder_var() {
        static expr_t ph_var = var_t::make(type_t::s32(), "_ph");
        return ph_var;
    }

private:
    static const int max_nvargs = 2;

    static stride_t compute_stride(const expr_t &e, int idx, const expr_t &var);

    expr_t expr_;

    int nvargs_ = 0;
    std::array<stride_t, max_nvargs> vstrides_;
    std::array<int, max_nvargs> vidxs_;
    expr_t mask_;
};

class view_t {
public:
    view_t() = default;

    view_t(const std::vector<expr_t> &vvars, int ntdims)
        : vvars_(vvars)
        , vdims_(vvars.size())
        , vstart_(vvars.size())
        , tdims_(ntdims) {}

    // Constructs view from a layout.
    explicit view_t(const layout_t &layout,
            const std::vector<expr_t> &_vvars = {},
            uint32_t bound_check_mask = 0)
        : vvars_(_vvars)
        , vdims_(layout.dims())
        , vstart_(layout.ndims(), 0)
        , tdims_(layout.ndims())
        , tlayout_(layout) {
        if (vvars_.empty()) vvars_ = create_vvars(layout.ndims());
        for (int i = 0; i < nvdims(); i++) {
            expr_t i_mask;
            if ((bound_check_mask & (1 << i)) != 0)
                i_mask = (placeholder_var() < layout.dim(i));
            set_tdim(i, vvars_[i], i_mask);
        }
    }

    const std::vector<expr_t> &vvars() const { return vvars_; }

    const std::vector<dim_t> &vdims() const { return vdims_; }

    expr_t vstart(int vidx) const { return vstart_[vidx]; }

    const layout_t tlayout() const { return tlayout_; }

    int nvdims() const { return int(vdims_.size()); }

    int ntdims() const { return int(tdims_.size()); }

    dim_t velems() const {
        dim_t ret = 1;
        for (int i = 0; i < nvdims(); i++)
            ret *= vdims_[i];
        return ret;
    }

    const expr_t &vvar(int idx) const {
        ir_assert(idx < nvdims());
        return vvars_[idx];
    }

    const tdim_info_t &tdim(int idx) const {
        ir_assert(idx < ntdims());
        return tdims_[idx];
    }

    void set_tdim(int tidx, const expr_t &_texpr, expr_t mask = {}) {
        ir_assert(tdims_[tidx].is_empty());

        auto texpr = simplify(_texpr);
        ir_assert(!is_const(texpr)) << "Tensor dimension can't be a constant.";

        tdim_info_t tdim(texpr, mask);
        for (int i = 0; i < nvdims(); i++) {
            if (contains_object(texpr, vvars_[i])) tdim.add_vvar(i, vvars_[i]);
        }
        ir_assert(tdim.nvargs() > 0)
                << "Tensor dimension must have at least one "
                   "view dimension that maps to it.";
        tdims_[tidx] = tdim;
    }

    void set_vdim(
            const expr_t &varg, dim_t vdim, const expr_t &vstart = expr_t(0)) {
        int vidx = vvar_index(varg);
        ir_assert(vstart_[vidx].is_empty());
        vstart_[vidx] = vstart;
        vdims_[vidx] = vdim;
    }

    void set_tlayout(const layout_t &tlayout) { tlayout_ = tlayout; }

    std::string str() const {
        using ir_utils::operator<<;

        if (is_empty()) return "(nil)";
        std::ostringstream oss;
        oss << ir_utils::make_seq_print_helper(vdims_, "x");
        if (!has_zero_vstart()) oss << " vstart: [" << vstart_ << "]";
        oss << " tlayout: " << tlayout_;
        return oss.str();
    }

    IR_DEFINE_DUMP()

    bool is_empty() const { return vdims_.empty(); }

    bool has_zero_vstart() const {
        for (int i = 0; i < nvdims(); i++)
            if (!is_zero(vstart_[i])) return false;
        return true;
    }

    bool has_tmask(int tidx) const {
        ir_assert(tidx >= 0 && tidx < ntdims());
        return !tdims_[tidx].mask().is_empty();
    }

    const type_t &type() const { return tlayout_.type(); }

    expr_t offset(const std::vector<expr_t> &vargs = {},
            bool ignore_offset = false) const {
        auto targs = cvt_vargs_to_targs(vargs);
        return tlayout_.offset(targs, ignore_offset);
    }

    expr_t offset_in_bytes(const std::vector<expr_t> &vargs = {},
            bool ignore_offset = false) const {
        return offset(vargs, ignore_offset) * type().size();
    }

    int vvar_index(const expr_t &vvar) const {
        for (size_t i = 0; i < vvars_.size(); i++)
            if (vvar.is_same(vvars_[i])) return int(i);
        ir_error_not_expected() << "Can't find view dimension.";
        return -1;
    }

    template <typename T>
    T operator()(const std::vector<T> &vargs) const {
        auto targs = cvt_vargs_to_targs(vargs);
        return tlayout_(targs);
    }

    view_t create_sub_view(const tensor_t &sub_tensor) const;

    view_t retype(const type_t &new_type) const {
        auto ret = *this;
        ret.tlayout_ = tlayout_.retype(new_type);
        return ret;
    }

    view_t make_dense() const {
        auto ret = *this;
        ret.tlayout_ = tlayout_.make_dense();
        return ret;
    }

    bool can_convert_to_vlayout() const {
        if (nvdims() != ntdims()) return false;
        for (int i = 0; i < nvdims(); i++) {
            if (!tdims_[i].expr().is_same(vvars_[i])) return false;
            if (!tdims_[i].is_fixed_stride(0)) return false;
        }
        return true;
    }

    // FIXME: Offset of the returned layout is always 0.
    layout_t create_pseudo_vlayout() const {
        return create_pseudo_vlayout(tlayout_);
    }

    layout_t create_dense_vlayout() const {
        return create_pseudo_vlayout().make_dense();
    }

    layout_t create_vlayout(bool force_zero_offset = false) const {
        ir_assert(can_convert_to_vlayout()) << "Can't convert view to layout.";
        if (force_zero_offset) return tlayout_.map(tensor_t(vdims_));
        return tlayout_.map(tensor_t(vdims_, vstart_));
    }

    dim_t vlayout_size() const { return create_vlayout().size(); }

    bool has_same_vlayout(
            const view_t &other, bool compare_offset = true) const {
        return create_vlayout().is_equal(
                other.create_vlayout(), compare_offset);
    }

    view_t split(const grid_info_t &grid, tensor_t &vtile) const {
        auto vlayout = create_pseudo_vlayout();
        vtile = vlayout.split(grid);
        return create_sub_view(vtile);
    }

    view_t split(const grid_info_t &grid) const {
        tensor_t vtile;
        return split(grid, vtile);
    }

    // Tile is assumed to be dense.
    tensor_t split_into_dense_tile(
            dim_t &tile_elems, dim_t &outer_block) const {
        auto vlayout = create_pseudo_vlayout();
        std::vector<dim_t> blocks = {tile_elems, outer_block};
        vlayout = vlayout.split_into_multi_blocks_with_hint(blocks);
        if (vlayout.is_empty()) return tensor_t();
        tile_elems = blocks[0];
        outer_block = blocks[1];
        return vlayout.split_into_dense_tile(tile_elems, outer_block);
    }

    // Returns a tensor corresponding to the biggest innermost sub-layout so that
    // 1) It consists of consecutive blocks only.
    // 2) It contains less or equal than max_tile_elems elements.
    // 3) It is dense if is_dense_tile is true.
    tensor_t split_into_max_tile(
            dim_t max_tile_elems, bool is_dense_tile) const {
        auto vlayout = create_pseudo_vlayout();
        return vlayout.split_into_max_tile(max_tile_elems, is_dense_tile);
    }

    template <typename F>
    void for_each_tile(const tensor_t &tile, const F &f) const {
        auto vlayout = create_dense_vlayout();
        vlayout.for_each_tile(tile, f);
    }

    view_t substitute(const expr_t &from, const expr_t &to) const;

    mask_tensor_t create_mask_tensor(const constraint_set_t &cset) const {
        auto _vlayout = create_dense_vlayout();
        mask_tensor_t mask_tensor(_vlayout);
        std::vector<dim_t> vargs(nvdims());
        create_mask_tensor(mask_tensor, _vlayout, 0, vargs);
        mask_tensor.simplify(cset);
        return mask_tensor;
    }

    bool try_create_buffer_view(view_t &buf_view, view_t &inv_view) const {
        buf_view = view_t(create_vvars(ntdims()), ntdims());
        inv_view = view_t(vvars(), ntdims());
        for (int i = 0; i < nvdims(); i++) {
            inv_view.set_vdim(vvars()[i], vdims()[i]);
        }
        for (int i = 0; i < ntdims(); i++) {
            auto &tdim = tdims_[i];
            auto &buf_vvar = buf_view.vvars()[i];
            if (tdim.is_identity()) {
                int vidx = tdim.vidx(0);
                buf_view.set_vdim(buf_vvar, vdims()[vidx], vstart(vidx));
                buf_view.set_tdim(i, buf_vvar, tdim.mask());
                inv_view.set_tdim(i, tdim.expr());
                continue;
            }
            int buf_vdim = 0;
            bool ok = true;
            for (int j = 0; j < tdim.nvargs(); j++) {
                int vidx = tdim.vidx(j);
                auto &vvar = vvars()[vidx];
                int vdim = vdims()[vidx];
                if (vdim == 1) continue;
                auto A = tdim.expr();
                auto B = jit::substitute(A, vvar, vvar + 1);
                auto C = simplify(B - A);
                if (!is_const(C)) {
                    ok = false;
                    break;
                }
                buf_vdim += to_cpp<int>(C) * (vdim - 1);
            }
            buf_vdim++;

            if (!ok) return false;

            auto buf_vstart = tdim.expr();
            auto inv_vstart = tdim.expr();
            for (int j = 0; j < tdim.nvargs(); j++) {
                int vidx = tdim.vidx(j);
                buf_vstart = jit::substitute(
                        buf_vstart, vvars()[vidx], vstart(vidx));
                inv_vstart
                        = jit::substitute(inv_vstart, vvars()[vidx], expr_t(0));
            }
            buf_vstart = simplify(buf_vstart);
            inv_vstart = simplify(inv_vstart);

            if (!is_const(inv_vstart)) return false;

            buf_view.set_vdim(buf_vvar, buf_vdim, buf_vstart);
            // TODO: Check that mask doesn't contain vvars.
            buf_view.set_tdim(i, buf_vvar, tdim.mask());
            inv_view.set_tdim(i, tdim.expr() - inv_vstart);
        }
        buf_view.set_tlayout(tlayout_);
        return true;
    }

    static const expr_t &placeholder_var() {
        return tdim_info_t::placeholder_var();
    }

    static std::vector<expr_t> create_vvars(int nvdims);

private:
    template <typename SrcT = expr_t, typename DstT = SrcT>
    std::vector<DstT> cvt_vargs_to_targs(
            const std::vector<SrcT> &_vargs = {}) const {
        std::vector<expr_t> vargs = expr_cast<expr_t>(_vargs);
        if (vargs.empty()) vargs.resize(nvdims(), 0);

        for (int i = 0; i < nvdims(); i++) {
            if (!is_zero(vstart_[i])) vargs[i] += vstart_[i];
        }

        std::vector<expr_t> targs(ntdims());
        for (int i = 0; i < ntdims(); i++) {
            targs[i] = tdims_[i].expr();
            for (int j = 0; j < nvdims(); j++) {
                targs[i] = jit::substitute(targs[i], vvars_[j], vargs[j]);
            }
        }
        for (int i = 0; i < ntdims(); i++) {
            targs[i] = const_fold(targs[i]);
        }
        return expr_cast<DstT>(targs);
    }

    layout_t create_pseudo_vlayout(const layout_t &tlayout) const;

    void create_mask_tensor(mask_tensor_t &mask_tensor,
            const layout_t &_vlayout, int vidx,
            std::vector<dim_t> &vargs) const {
        if (vidx == _vlayout.ndims()) {
            std::vector<expr_t> vvalues = vstart_;
            for (int i = 0; i < nvdims(); i++)
                vvalues[i] += vargs[i];
            auto targs = cvt_vargs_to_targs<dim_t, expr_t>(vargs);
            expr_t mask = bool_imm_t::make(true);
            for (int i = 0; i < ntdims(); i++) {
                auto &tdim = tdims_[i];
                if (tdim.mask().is_empty()) continue;
                mask &= tdim.mask(targs[i], vvars_, vvalues);
            }
            mask_tensor.set_mask(_vlayout(vargs), mask);
            return;
        }

        for (int i = 0; i < vdims()[vidx]; i++) {
            vargs[vidx] = i;
            create_mask_tensor(mask_tensor, _vlayout, vidx + 1, vargs);
        }
    }

    std::vector<expr_t> vvars_;
    std::vector<dim_t> vdims_;
    std::vector<expr_t> vstart_;

    std::vector<tdim_info_t> tdims_;
    layout_t tlayout_;
};

inline std::ostream &operator<<(std::ostream &out, const view_t &view) {
    out << view.str();
    return out;
}

class dim_assignment_t {
public:
    dim_assignment_t() = default;

    dim_assignment_t(int old_ndims, int new_ndims)
        : old_ndims_(old_ndims)
        , new_ndims_(new_ndims)
        , assignments_(old_ndims, -1) {}

    void assign(int old_idx, int new_idx) {
        ir_assert(0 <= old_idx && old_idx < old_ndims_);
        ir_assert(0 <= new_idx && new_idx < new_ndims_);
        assignments_[old_idx] = new_idx;
    }

    void assign(const std::vector<int> &old_idxes, int new_idx) {
        for (auto old_idx : old_idxes) {
            assign(old_idx, new_idx);
        }
    }

    int operator[](int old_idx) const {
        ir_assert(old_idx >= 0 && old_idx < old_ndims());
        return assignments_[old_idx];
    }

    int old_ndims() const { return old_ndims_; }

    int new_ndims() const { return new_ndims_; }

    bool is_empty() const { return old_ndims_ == 0 && new_ndims_ == 0; }

    layout_t map(const layout_t &layout) const;

private:
    int old_ndims_ = 0;
    int new_ndims_ = 0;

    // assignments_[old_idx] = new_idx.
    std::vector<int> assignments_;
};

std::vector<dim_t> normalize_conv_dims(std::vector<dim_t> &dims,
        bool with_groups, int groups, bool is_dw, bool reduced_to_1d,
        bool add_groups, bool is_wei);

layout_t normalize_conv_layout(const layout_t &_layout, bool with_groups,
        int groups, bool is_dw, bool reduced_to_1d, bool add_groups,
        bool is_wei);

void normalize_conv_layouts(layout_t &src_layout, layout_t &wei_layout,
        layout_t &dst_layout, bool with_groups, int groups, bool is_dw,
        bool reduced_to_1d, bool add_groups);

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
