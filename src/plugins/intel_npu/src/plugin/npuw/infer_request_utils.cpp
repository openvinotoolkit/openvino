// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_request_utils.hpp"

#include "logging.hpp"
#include "openvino/runtime/make_tensor.hpp"  // get_tensor_impl
#include "util_xarch.hpp"

// FIXME: Use ov::npuw::util::view instead
ov::SoPtr<ov::ITensor> ov::npuw::util::make_tensor_slice(ov::SoPtr<ov::ITensor> tensor,
                                                         uint32_t dim,
                                                         uint32_t start_pos,
                                                         uint32_t end_pos) {
    // Sub-byte element types (i4/u4) are not supported by make_tensor ROI path.
    // Use copy_tensor_slice_i4() instead.
    const auto& et = tensor->get_element_type();
    if (et.bitwidth() < 8u) {
        OPENVINO_ASSERT(false,
                        "make_tensor_slice: sub-byte tensor type ",
                        et,
                        " is not supported by ROI tensor creation. "
                        "Use copy_tensor_slice_i4() for i4/u4 slices.");
    }
    ov::Shape start_shape(std::vector<size_t>(tensor->get_shape().size(), 0u));
    start_shape[dim] = start_pos;
    ov::Shape end_shape = tensor->get_shape();
    end_shape[dim] = end_pos;
    return ov::get_tensor_impl(ov::Tensor(ov::make_tensor(tensor), start_shape, end_shape));
}

void ov::npuw::util::copy_to_right(const ov::SoPtr<ov::ITensor>& src, const ov::SoPtr<ov::ITensor>& dst) {
    OPENVINO_ASSERT(src->get_byte_size() <= dst->get_byte_size());
    std::copy_n(reinterpret_cast<uint8_t*>(src->data()),
                src->get_byte_size(),
                reinterpret_cast<uint8_t*>(dst->data()) + dst->get_byte_size() - src->get_byte_size());
}

void ov::npuw::util::copy_by_planes(ov::SoPtr<ov::ITensor> src_tensor, ov::SoPtr<ov::ITensor> dst_tensor) {
    // [1, H, S1, E] -> [1, H, S2, E]
    const int N = 0;
    const int H = 1;
    const int S = 2;
    const int E = 3;

    OPENVINO_ASSERT(src_tensor->get_shape()[N] == dst_tensor->get_shape()[N]);
    OPENVINO_ASSERT(src_tensor->get_shape()[H] == dst_tensor->get_shape()[H]);
    OPENVINO_ASSERT(src_tensor->get_shape()[E] == dst_tensor->get_shape()[E]);
    OPENVINO_ASSERT(src_tensor->get_element_type() == dst_tensor->get_element_type());
    OPENVINO_ASSERT(src_tensor->get_shape()[N] == 1u);
    OPENVINO_ASSERT(src_tensor->get_shape().size() == 4u);

    const auto* src_tensor_data = reinterpret_cast<uint8_t*>(src_tensor->data());
    auto* dst_tensor_data = reinterpret_cast<uint8_t*>(dst_tensor->data());

    const auto num_planes = src_tensor->get_shape()[H];
    const auto src_plane_stride = src_tensor->get_strides()[H];
    const auto dst_plane_stride = dst_tensor->get_strides()[H];
    const auto plane_size_in_bytes = src_tensor->get_strides()[S] * src_tensor->get_shape()[S];

    for (size_t i = 0; i < num_planes; ++i) {
        std::copy_n(src_tensor_data, plane_size_in_bytes, dst_tensor_data);
        dst_tensor_data += dst_plane_stride;
        src_tensor_data += src_plane_stride;
    }
}

void ov::npuw::util::copy_columns_by_row_chunks(ov::SoPtr<ov::ITensor> src, ov::SoPtr<ov::ITensor>& dst) {
    /*
      src/dst layout: [1, heads, emb_size, seq_len]

      X[*,i] - embedding for i-th token,
      Instead of copy columns, copy rows X[i,*]

      [[X00 X01 ... X0n]      [[X00 X01 ... X0n]
       [X10 X11 ... X1n]       [X10 X11 ... X1n]
       [X20 X21 ... X2n]  ...  [X20 X21 ... X2n]
             ...                     ...
       [Xm0 Xm1 ... Xmn]]      [Xm0 Xm1 ... Xmn]]
    */

    const auto& src_shape = src->get_shape();

    OPENVINO_ASSERT(src_shape.size() == 4u);
    OPENVINO_ASSERT(src_shape == dst->get_shape());
    OPENVINO_ASSERT(src->get_byte_size() == dst->get_byte_size());

    const auto& src_strides = src->get_strides();
    const auto& dst_strides = dst->get_strides();
    const auto elem_size = src->get_byte_size() / src->get_size();

    const auto C = src_shape[1];
    const auto H = src_shape[2];
    const auto W = src_shape[3];

    const auto IS_H = src_strides[2];
    const auto OS_H = dst_strides[2];

    const size_t chunk_byte_size = W * elem_size;

    const auto* src_p = static_cast<uint8_t*>(src->data());
    auto* dst_p = static_cast<uint8_t*>(dst->data());

    for (size_t i = 0; i < C * H; ++i) {
        const size_t src_offset = i * IS_H;
        const size_t dst_offset = i * OS_H;
        std::copy_n(src_p + src_offset, chunk_byte_size, dst_p + dst_offset);
    }
}

void ov::npuw::util::copy_tensor_by_dim(ov::SoPtr<ov::ITensor> src_tensor,
                                        ov::SoPtr<ov::ITensor> dst_tensor,
                                        uint32_t kv_dim_src,
                                        uint32_t kv_dim_dst) {
    if (kv_dim_src != kv_dim_dst) {
        // new case - do a generic copy for now (in fact it is a permute)
        // Example:
        //   kv_dim_src         kv_dim_dst
        //       v                     v
        // [1,8,256,128] --> [1,8,128,256]
        const auto& src_shape = src_tensor->get_shape();
        const auto& dst_shape = dst_tensor->get_shape();
        NPUW_ASSERT(src_shape.size() == 4);
        NPUW_ASSERT(dst_shape.size() == 4);
        NPUW_ASSERT(kv_dim_src < 4);
        NPUW_ASSERT(kv_dim_dst < 4);
        NPUW_ASSERT(src_shape[kv_dim_src] == dst_shape[kv_dim_dst]);

        std::array<int, 4> axis = {0, 1, 2, 3};
        // Remap like 0,1,2,3 => 0,1,3,2 (see example)
        std::swap(axis[kv_dim_src], axis[kv_dim_dst]);
        ov::npuw::util::permute_i4d(src_tensor, dst_tensor, axis);
        return;
    }
    // Old behavior
    NPUW_ASSERT(kv_dim_src == kv_dim_dst);
    if (kv_dim_src == 3u) {
        // Asserting that we work with last dimenston here:
        const auto& src_shape = src_tensor->get_shape();
        OPENVINO_ASSERT(src_shape.size() == 4);
        // If last dimenstion of src_tensor is equal to 1, then we can squeeze
        // src_shape from [1, heads, d_v, seq_len=1] to [heads, d_v].
        // We can then treat src_tensor as a continuous tensor of row value vectors
        // for multiple heads, while dst_tensor will still have [1, heads, d_v, seq_len!=1],
        // shape, awaiting updates at column dimension, as value vectors are columns now.
        if (src_shape[kv_dim_src] == 1 && src_tensor->is_continuous()) {
            // FIXME: ov::npuw::util::XARCH::copy_row_as_column(src_tensor, dst_tensor) throws when used here
            copy_columns_by_row_chunks(src_tensor, dst_tensor);
        } else {
            copy_columns_by_row_chunks(src_tensor, dst_tensor);
        }
    } else if (kv_dim_src == 2u) {
        copy_by_planes(src_tensor, dst_tensor);
    } else {
        src_tensor->copy_to(dst_tensor._ptr);
    }
}

std::optional<ov::Output<const ov::Node>> ov::npuw::util::find_port_by_name(
    const std::vector<ov::Output<const ov::Node>>& ports,
    const std::string& name) {
    auto it = std::find_if(ports.begin(), ports.end(), [&](const auto& port) {
        return port.get_names().count(name) != 0;
    });
    if (it == ports.end()) {
        return std::nullopt;
    }
    return std::make_optional(*it);
}

std::optional<ov::Output<const ov::Node>> ov::npuw::util::find_port_by_names(
    const std::vector<ov::Output<const ov::Node>>& ports,
    const std::unordered_set<std::string>& names) {
    for (const auto& port : ports) {
        const auto& port_names = port.get_names();
        for (const auto& port_name : port_names) {
            if (names.count(port_name)) {
                return std::make_optional(port);
            }
        }
    }
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// copy_tensor_slice_i4
// ---------------------------------------------------------------------------
// Copies [src_start, src_end) along src_dim in i4/u4 tensors without creating
// ROI tensors. Works directly on packed bytes and handles odd (nibble-aligned)
// offsets via boundary nibble arithmetic.
// ---------------------------------------------------------------------------

namespace {

static inline uint8_t get_i4_elem(const uint8_t* data, size_t elem_idx) {
    const uint8_t b = data[elem_idx / 2u];
    return (elem_idx & 1u) ? static_cast<uint8_t>((b >> 4u) & 0x0Fu) : static_cast<uint8_t>(b & 0x0Fu);
}

static inline void set_i4_elem(uint8_t* data, size_t elem_idx, uint8_t val) {
    uint8_t& b = data[elem_idx / 2u];
    const uint8_t nibble = static_cast<uint8_t>(val & 0x0Fu);
    if (elem_idx & 1u) {
        b = static_cast<uint8_t>((b & 0x0Fu) | static_cast<uint8_t>(nibble << 4u));
    } else {
        b = static_cast<uint8_t>((b & 0xF0u) | nibble);
    }
}

// Copy i4/u4 scalar elements from source logical scalar index range into
// destination scalar index range. Uses byte copy for aligned bulk and nibble
// operations for boundary/unaligned tails.
static void copy_i4_range(const uint8_t* src_data,
                          size_t src_elem_begin,
                          uint8_t* dst_data,
                          size_t dst_elem_begin,
                          size_t elem_count) {
    if (elem_count == 0u)
        return;

    size_t src_idx = src_elem_begin;
    size_t dst_idx = dst_elem_begin;
    size_t left = elem_count;

    // If parity differs, no direct byte-to-byte mapping is possible.
    if ((src_idx & 1u) != (dst_idx & 1u)) {
        for (size_t i = 0; i < left; ++i) {
            set_i4_elem(dst_data, dst_idx + i, get_i4_elem(src_data, src_idx + i));
        }
        return;
    }

    // Same parity: peel one leading element if odd nibble-aligned, then copy
    // full bytes, then one trailing element if needed.
    if (src_idx & 1u) {
        set_i4_elem(dst_data, dst_idx, get_i4_elem(src_data, src_idx));
        ++src_idx;
        ++dst_idx;
        --left;
    }

    const size_t full_bytes = left / 2u;
    std::copy_n(src_data + src_idx / 2u, full_bytes, dst_data + dst_idx / 2u);

    if (left & 1u) {
        set_i4_elem(dst_data, dst_idx + full_bytes * 2u, get_i4_elem(src_data, src_idx + full_bytes * 2u));
    }
}

}  // anonymous namespace

void ov::npuw::util::copy_tensor_slice_i4(const ov::SoPtr<ov::ITensor>& src,
                                          uint32_t src_dim,
                                          uint32_t src_start,
                                          uint32_t src_end,
                                          const ov::SoPtr<ov::ITensor>& dst,
                                          uint32_t dst_dim,
                                          uint32_t dst_start) {
    OPENVINO_ASSERT(src->get_element_type().bitwidth() == 4u,
                    "copy_tensor_slice_i4: source element type must be i4 or u4, got ",
                    src->get_element_type());
    OPENVINO_ASSERT(src->get_element_type() == dst->get_element_type(),
                    "copy_tensor_slice_i4: source and destination element types must match.");

    const auto& src_shape = src->get_shape();
    const auto& dst_shape = dst->get_shape();
    const size_t rank = src_shape.size();
    OPENVINO_ASSERT(rank == dst_shape.size() && rank >= 1u,
                    "copy_tensor_slice_i4: rank mismatch or empty shape.");
    OPENVINO_ASSERT(src_dim < rank && dst_dim < rank,
                    "copy_tensor_slice_i4: invalid slice dimension.");
    OPENVINO_ASSERT(src_dim == dst_dim,
                    "copy_tensor_slice_i4: src_dim != dst_dim is not supported.");

    const uint32_t count = src_end - src_start;
    OPENVINO_ASSERT(count > 0u, "copy_tensor_slice_i4: empty slice.");

    const size_t slice_dim = static_cast<size_t>(src_dim);
    const size_t src_dim_size = src_shape[slice_dim];
    const size_t dst_dim_size = dst_shape[slice_dim];
    OPENVINO_ASSERT(src_end <= src_dim_size,
                    "copy_tensor_slice_i4: src_end exceeds source slice dimension size.");
    OPENVINO_ASSERT(dst_start + count <= dst_dim_size,
                    "copy_tensor_slice_i4: destination range exceeds destination slice dimension size.");

    // Number of outer blocks = product of dimensions before sliced dim.
    size_t num_rows = 1u;
    for (size_t d = 0; d < slice_dim; ++d)
        num_rows *= src_shape[d];

    // Elements inside one position of sliced dim.
    size_t inner_elems = 1u;
    for (size_t d = slice_dim + 1u; d < rank; ++d)
        inner_elems *= src_shape[d];

    // Full tensor scalar element counts.
    size_t src_total_elems = 1u;
    for (const auto s : src_shape)
        src_total_elems *= s;
    size_t dst_total_elems = 1u;
    for (const auto s : dst_shape)
        dst_total_elems *= s;

    const auto* src_bytes = static_cast<const uint8_t*>(src->data());
    auto* dst_bytes = static_cast<uint8_t*>(dst->data());

    for (size_t row = 0; row < num_rows; ++row) {
        const size_t src_elem_begin = (row * src_dim_size + static_cast<size_t>(src_start)) * inner_elems;
        const size_t dst_elem_begin = (row * dst_dim_size + static_cast<size_t>(dst_start)) * inner_elems;
        const size_t elem_count = static_cast<size_t>(count) * inner_elems;
        OPENVINO_ASSERT(src_elem_begin + elem_count <= src_total_elems,
                        "copy_tensor_slice_i4: source range out of bounds.");
        OPENVINO_ASSERT(dst_elem_begin + elem_count <= dst_total_elems,
                        "copy_tensor_slice_i4: destination range out of bounds.");
        copy_i4_range(src_bytes, src_elem_begin, dst_bytes, dst_elem_begin, elem_count);
    }
}

void ov::npuw::util::pad_position_ids(const ov::SoPtr<ov::ITensor>& padded_position_ids,
                                      const ov::SoPtr<ov::ITensor>& position_ids) {
    // NB: Regular LLM uses 2D position_ids [BATCH, SEQ_LEN], Qwen2.5 VL/Omni uses 3D position_ids [3, BATCH, SEQ_LEN]
    // The first dimension (3) represents the three components of position encoding: time, height, and width
    // enabling alignment across multimodal inputs like text, audio, and video
    auto padded_shape = padded_position_ids->get_shape();
    auto position_shape = position_ids->get_shape();

    OPENVINO_ASSERT(position_shape.size() <= 3);

    size_t diff_dim = position_shape.size() - 1;
    for (size_t i = 0; i < diff_dim; ++i) {
        OPENVINO_ASSERT(padded_shape[i] == position_shape[i]);
    }

    size_t keep_elements = padded_shape[diff_dim] - position_shape[diff_dim];

    size_t batch_size = 1;
    for (size_t i = 0; i < padded_shape.size(); ++i) {
        if (i != diff_dim) {
            batch_size *= padded_shape[i];
        }
    }

    int64_t* padded_data = padded_position_ids->data<int64_t>();
    const int64_t* position_data = position_ids->data<int64_t>();

    for (size_t batch = 0; batch < batch_size; ++batch) {
        size_t padded_offset = batch * padded_shape[diff_dim];
        size_t position_offset = batch * position_shape[diff_dim];
        std::copy_n(position_data + position_offset,
                    position_shape[diff_dim],
                    padded_data + padded_offset + keep_elements);
    }
}
