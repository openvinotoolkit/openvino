// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_request_utils.hpp"

#include <limits>

#include "logging.hpp"
#include "openvino/runtime/make_tensor.hpp"  // get_tensor_impl
#include "util.hpp"
#include "util_xarch.hpp"

// FIXME: Use ov::npuw::util::view instead
ov::SoPtr<ov::ITensor> ov::npuw::util::make_tensor_slice(ov::SoPtr<ov::ITensor> tensor,
                                                         uint32_t dim,
                                                         uint32_t start_pos,
                                                         uint32_t end_pos) {
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
        NPUW_ASSERT(dst_tensor._ptr &&
                    "null tensor view passed to copy — check that the source tensor is valid and non-empty");
        src_tensor->copy_to(dst_tensor._ptr);
    }
}

void ov::npuw::util::write_swa_kv_slice_circular(ov::SoPtr<ov::ITensor> dst_tensor,
                                                 ov::SoPtr<ov::ITensor> src_new_kv,
                                                 uint32_t dst_kv_dim,
                                                 uint32_t src_kv_dim,
                                                 uint32_t num_stored_tokens_before,
                                                 uint32_t num_new_tokens) {
    const uint32_t capacity = static_cast<uint32_t>(dst_tensor->get_shape()[dst_kv_dim]);
    const uint32_t old_total = num_stored_tokens_before;
    const uint32_t new_total = old_total + num_new_tokens;
    const uint32_t new_valid = std::min(new_total, capacity);

    // Clamp by source length as well: source may already be capacity-limited.
    const uint32_t src_len = static_cast<uint32_t>(src_new_kv->get_shape()[src_kv_dim]);
    const uint32_t tokens_to_write = std::min({num_new_tokens, new_valid, src_len});

    if (tokens_to_write == 0) {
        return;
    }
    const uint32_t first_new_abs_pos = num_stored_tokens_before + (num_new_tokens - tokens_to_write);
    const uint32_t dst_start = first_new_abs_pos % capacity;

    auto src_slice = (src_len > tokens_to_write)
                         ? ov::npuw::util::make_tensor_slice(src_new_kv, src_kv_dim, src_len - tokens_to_write, src_len)
                         : src_new_kv;

    if (dst_start + tokens_to_write <= capacity) {
        auto dst_slice =
            ov::npuw::util::make_tensor_slice(dst_tensor, dst_kv_dim, dst_start, dst_start + tokens_to_write);
        ov::npuw::util::copy_tensor_by_dim(src_slice, dst_slice, src_kv_dim, dst_kv_dim);
    } else {
        const uint32_t first_leg_len = capacity - dst_start;
        const uint32_t second_leg_len = tokens_to_write - first_leg_len;

        auto src_first_leg = ov::npuw::util::make_tensor_slice(src_slice, src_kv_dim, 0u, first_leg_len);
        auto dst_first_leg = ov::npuw::util::make_tensor_slice(dst_tensor, dst_kv_dim, dst_start, capacity);
        ov::npuw::util::copy_tensor_by_dim(src_first_leg, dst_first_leg, src_kv_dim, dst_kv_dim);

        auto src_second_leg = ov::npuw::util::make_tensor_slice(src_slice, src_kv_dim, first_leg_len, tokens_to_write);
        auto dst_second_leg = ov::npuw::util::make_tensor_slice(dst_tensor, dst_kv_dim, 0u, second_leg_len);
        ov::npuw::util::copy_tensor_by_dim(src_second_leg, dst_second_leg, src_kv_dim, dst_kv_dim);
    }
}

void ov::npuw::util::write_swa_kv_slice_left_aligned(ov::SoPtr<ov::ITensor> dst_tensor,
                                                     ov::SoPtr<ov::ITensor> src_new_kv,
                                                     uint32_t dst_kv_dim,
                                                     uint32_t src_kv_dim,
                                                     uint32_t num_stored_tokens_before,
                                                     uint32_t num_new_tokens) {
    const uint32_t capacity = static_cast<uint32_t>(dst_tensor->get_shape()[dst_kv_dim]);
    const uint32_t old_total = num_stored_tokens_before;
    const uint32_t new_total = old_total + num_new_tokens;
    const uint32_t old_valid = std::min(old_total, capacity);
    const uint32_t new_valid = std::min(new_total, capacity);

    // Clamp against source length too. Some source tensors can hold fewer tokens
    // than num_new_tokens when they were capacity-limited earlier.
    const uint32_t src_len = static_cast<uint32_t>(src_new_kv->get_shape()[src_kv_dim]);
    const uint32_t tokens_to_write = std::min({num_new_tokens, new_valid, src_len});

    const uint32_t keep = new_valid - tokens_to_write;
    const bool needs_shift = (keep > 0 && keep < old_valid);

    if (needs_shift && dst_kv_dim == 3u) {
        // Transposed-V (dim=3) partial-slice shifts degrade into many small remote
        // transfers. Use a full-buffer round-trip to keep
        // transfer count low and improve stability on remote tensors.
        LOG_DEBUG("[SWA] Bulk-shifting KV buffer (dim=3): keeping last " << keep << " of " << old_valid
                                                                         << " old token(s), capacity=" << capacity);
        auto whole_tmp =
            ov::npuw::util::allocMem(dst_tensor->get_element_type(), dst_tensor->get_shape(), "CPU", nullptr);
        dst_tensor->copy_to(whole_tmp._ptr);  // single bulk contiguous transfer

        auto old_tail_cpu = ov::npuw::util::make_tensor_slice(whole_tmp, dst_kv_dim, old_valid - keep, old_valid);
        auto shift_tmp =
            ov::npuw::util::allocMem(dst_tensor->get_element_type(), old_tail_cpu->get_shape(), "CPU", nullptr);
        old_tail_cpu->copy_to(shift_tmp._ptr);  // CPU-to-CPU, cheap regardless of iteration count
        auto dst_front_cpu = ov::npuw::util::make_tensor_slice(whole_tmp, dst_kv_dim, 0u, keep);
        ov::npuw::util::copy_tensor_by_dim(shift_tmp, dst_front_cpu, dst_kv_dim, dst_kv_dim);  // CPU-to-CPU

        if (tokens_to_write > 0) {
            auto src_slice =
                (src_len > tokens_to_write)
                    ? ov::npuw::util::make_tensor_slice(src_new_kv, src_kv_dim, src_len - tokens_to_write, src_len)
                    : src_new_kv;
            auto dst_back_cpu = ov::npuw::util::make_tensor_slice(whole_tmp, dst_kv_dim, keep, keep + tokens_to_write);
            ov::npuw::util::copy_tensor_by_dim(src_slice, dst_back_cpu, src_kv_dim, dst_kv_dim);
        }

        whole_tmp->copy_to(dst_tensor._ptr);  // single bulk contiguous transfer back
        return;
    }

    if (needs_shift) {
        // Window saturated: move the surviving old tail to the front.
        // Use a temporary buffer to avoid overlapping in-place copy.
        LOG_DEBUG("[SWA] Shifting KV buffer: keeping last "
                  << keep << " of " << old_valid << " old token(s), dim=" << dst_kv_dim << ", capacity=" << capacity);
        auto old_tail = ov::npuw::util::make_tensor_slice(dst_tensor, dst_kv_dim, old_valid - keep, old_valid);
        auto tmp = ov::npuw::util::allocMem(dst_tensor->get_element_type(), old_tail->get_shape(), "CPU", nullptr);
        old_tail->copy_to(tmp._ptr);
        auto dst_front = ov::npuw::util::make_tensor_slice(dst_tensor, dst_kv_dim, 0u, keep);
        ov::npuw::util::copy_tensor_by_dim(tmp, dst_front, dst_kv_dim, dst_kv_dim);
    }

    if (tokens_to_write == 0) {
        return;
    }
    auto src_slice = (src_len > tokens_to_write)
                         ? ov::npuw::util::make_tensor_slice(src_new_kv, src_kv_dim, src_len - tokens_to_write, src_len)
                         : src_new_kv;
    auto dst_back = ov::npuw::util::make_tensor_slice(dst_tensor, dst_kv_dim, keep, keep + tokens_to_write);
    ov::npuw::util::copy_tensor_by_dim(src_slice, dst_back, src_kv_dim, dst_kv_dim);
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

void ov::npuw::util::copy_per_layer_inputs_chunk_to_right(const ov::SoPtr<ov::ITensor>& src,
                                                          const ov::SoPtr<ov::ITensor>& dst,
                                                          uint32_t src_offset_tokens,
                                                          uint32_t chunk_tokens) {
    // Gemma4 26B A4B has dangling per_layer_inputs with zero-sized tensors.
    if (src->get_byte_size() == 0u || dst->get_byte_size() == 0u) {
        OPENVINO_ASSERT(src->get_byte_size() == 0u && dst->get_byte_size() == 0u,
                        "per_layer_inputs zero-byte mismatch between src and dst. src_bytes=",
                        src->get_byte_size(),
                        ", dst_bytes=",
                        dst->get_byte_size());
        return;
    }

    const auto src_seq_len = src->get_shape().at(1);
    const auto dst_seq_len = dst->get_shape().at(1);
    OPENVINO_ASSERT(chunk_tokens > 0u, "chunk_tokens must be > 0");
    OPENVINO_ASSERT(src_offset_tokens <= src_seq_len,
                    "src_offset_tokens exceeds source seq_len. src_offset_tokens=",
                    src_offset_tokens,
                    ", src_seq_len=",
                    src_seq_len);
    OPENVINO_ASSERT(chunk_tokens <= src_seq_len - src_offset_tokens,
                    "chunk range exceeds source seq_len by given offset. src_offset_tokens=",
                    src_offset_tokens,
                    ", chunk_tokens=",
                    chunk_tokens,
                    ", src_seq_len=",
                    src_seq_len);
    OPENVINO_ASSERT(chunk_tokens <= dst_seq_len,
                    "chunk_tokens exceeds destination seq_len. chunk_tokens=",
                    chunk_tokens,
                    ", dst_seq_len=",
                    dst_seq_len);

    const auto src_seq_len_bytes = src->get_byte_size();
    const auto dst_seq_len_bytes = dst->get_byte_size();
    OPENVINO_ASSERT(src_seq_len > 0u, "per_layer_inputs src has zero seq_len");
    OPENVINO_ASSERT(dst_seq_len > 0u, "per_layer_inputs dst has zero seq_len");
    OPENVINO_ASSERT(src_seq_len_bytes % src_seq_len == 0u,
                    "per_layer_inputs src byte size is not divisible by seq_len. byte_size=",
                    src_seq_len_bytes,
                    ", seq_len=",
                    src_seq_len);
    OPENVINO_ASSERT(dst_seq_len_bytes % dst_seq_len == 0u,
                    "per_layer_inputs dst byte size is not divisible by seq_len. byte_size=",
                    dst_seq_len_bytes,
                    ", seq_len=",
                    dst_seq_len);
    const auto src_per_token_bytes = src_seq_len_bytes / src_seq_len;
    const auto dst_per_token_bytes = dst_seq_len_bytes / dst_seq_len;
    OPENVINO_ASSERT(src_per_token_bytes == dst_per_token_bytes,
                    "per-token byte size mismatch between src and dst. src=",
                    src_per_token_bytes,
                    ", dst=",
                    dst_per_token_bytes);

    OPENVINO_ASSERT(static_cast<size_t>(chunk_tokens) <= std::numeric_limits<size_t>::max() / src_per_token_bytes,
                    "chunk byte size overflow. chunk_tokens=",
                    chunk_tokens,
                    ", per_token_bytes=",
                    src_per_token_bytes);
    OPENVINO_ASSERT(static_cast<size_t>(src_offset_tokens) <= std::numeric_limits<size_t>::max() / src_per_token_bytes,
                    "offset byte size overflow. src_offset_tokens=",
                    src_offset_tokens,
                    ", per_token_bytes=",
                    src_per_token_bytes);

    const size_t chunk_bytes = static_cast<size_t>(chunk_tokens) * src_per_token_bytes;
    const size_t offset_bytes = static_cast<size_t>(src_offset_tokens) * src_per_token_bytes;

    std::copy_n(reinterpret_cast<const uint8_t*>(src->data()) + offset_bytes,
                chunk_bytes,
                reinterpret_cast<uint8_t*>(dst->data()) + dst->get_byte_size() - chunk_bytes);
}
