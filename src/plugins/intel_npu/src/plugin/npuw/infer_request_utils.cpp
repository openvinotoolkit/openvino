// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_request_utils.hpp"

#include <algorithm>
#include <limits>
#include <string>

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

void ov::npuw::util::fill_causal_sliding_mask(ov::SoPtr<ov::ITensor> mask_tensor,
                                              uint32_t num_stored_tokens_before,
                                              uint32_t num_real_new_tokens,
                                              uint32_t window_size) {
    OPENVINO_ASSERT(mask_tensor->get_element_type() == ov::element::f32,
                    "Attention mask tensor is expected to be f32, got: ",
                    mask_tensor->get_element_type());
    const auto& shape = mask_tensor->get_shape();
    OPENVINO_ASSERT(shape.size() >= 2, "Attention mask tensor rank must be >= 2, got shape: ", shape);
    const uint32_t row_dim = static_cast<uint32_t>(shape[shape.size() - 2]);
    const uint32_t col_dim = static_cast<uint32_t>(shape[shape.size() - 1]);
    OPENVINO_ASSERT(col_dim >= row_dim,
                    "Attention mask's key axis (",
                    col_dim,
                    ") must be >= its query axis (",
                    row_dim,
                    ")");
    OPENVINO_ASSERT(num_real_new_tokens <= row_dim,
                    "num_real_new_tokens (",
                    num_real_new_tokens,
                    ") exceeds the mask's query axis (",
                    row_dim,
                    ")");
    const uint32_t past_width = col_dim - row_dim;
    const uint32_t row_pad = row_dim - num_real_new_tokens;  // rows/current-cols < row_pad are pad.
    const uint32_t P = num_stored_tokens_before;

    constexpr float kAttend = 0.0f;
    // For NPU execution - use fp16 lowest value to represent masked positions
    float kMasked = static_cast<float>(std::numeric_limits<ov::float16>::lowest());

    float* data = mask_tensor->data<float>();

    for (uint32_t row = 0; row < row_dim; ++row) {
        float* row_ptr = data + static_cast<size_t>(row) * col_dim;

        // Past columns: c in [0, past_width). The past K/V buffer is maintained by
        // write_kv_slice_sliding(..., SlidingBufferLayout::Circular): physical slot c always
        // holds whichever absolute token position last landed there via `p % past_width` - no
        // data is ever shifted (see kv_cache_sliding_window_manager.hpp). While the window has
        // not yet saturated (P < past_width), writes fill physical slots strictly in arrival
        // order 0, 1, 2, ... - so the valid prefix is LEFT-aligned at [0, P) (column c holds
        // absolute position c), and columns >= P are still-uninitialized garbage. Once
        // P >= past_width (saturated at least once), every physical slot has been written at
        // least once (all valid), but which absolute position slot c currently holds depends on
        // how far the wrap-around write cursor `r = P % past_width` has progressed: slots >= r
        // hold the most recently *completed* lap (abs = P - r + c - past_width), slots < r hold
        // the lap currently in progress (abs = P - r + c).
        const int64_t row_local = static_cast<int64_t>(row) - static_cast<int64_t>(row_pad);
        const int64_t q = static_cast<int64_t>(P) + row_local;  // this row's own absolute position
        const uint32_t r = P % past_width;
        for (uint32_t c = 0; c < past_width; ++c) {
            bool valid;
            int64_t abs_pos;
            if (P >= past_width) {
                valid = true;
                abs_pos = (c < r) ? (static_cast<int64_t>(P) - r + c)
                                  : (static_cast<int64_t>(P) - r + c - past_width);
            } else {
                valid = c < P;
                abs_pos = c;
            }
            const bool causal = abs_pos <= q;
            const bool window_ok = (q - abs_pos) < static_cast<int64_t>(window_size);
            const bool attend = valid && causal && window_ok;
            row_ptr[c] = attend ? kAttend : kMasked;
        }

        // Current-chunk diagonal columns: local_c in [0, row_dim), mapped to c = past_width +
        // local_c. Both axes share the same row_pad right-alignment offset, so it cancels
        // identically in both the causal and window comparisons below - raw indices suffice.
        for (uint32_t local_c = 0; local_c < row_dim; ++local_c) {
            const bool valid_key = local_c >= row_pad;
            const bool causal = local_c <= row;
            const bool window_ok = causal && (row - local_c) < window_size;
            const bool attend = valid_key && causal && window_ok;
            row_ptr[past_width + local_c] = attend ? kAttend : kMasked;
        }
    }
}

void ov::npuw::util::overlay_vision_bidirectional_mask(ov::SoPtr<ov::ITensor> mask_tensor,
                                                        const int64_t* token_type_ids_real,
                                                        uint32_t num_real_new_tokens) {
    if (num_real_new_tokens == 0) {
        return;
    }
    OPENVINO_ASSERT(mask_tensor->get_element_type() == ov::element::f32,
                    "Attention mask tensor is expected to be f32, got: ",
                    mask_tensor->get_element_type());
    const auto& shape = mask_tensor->get_shape();
    OPENVINO_ASSERT(shape.size() >= 2, "Attention mask tensor rank must be >= 2, got shape: ", shape);
    const uint32_t row_dim = static_cast<uint32_t>(shape[shape.size() - 2]);
    const uint32_t col_dim = static_cast<uint32_t>(shape[shape.size() - 1]);
    OPENVINO_ASSERT(num_real_new_tokens <= row_dim,
                    "num_real_new_tokens (",
                    num_real_new_tokens,
                    ") exceeds the mask's query axis (",
                    row_dim,
                    ")");
    const uint32_t past_width = col_dim - row_dim;
    const uint32_t row_pad = row_dim - num_real_new_tokens;

    // Assign each real token a group id: bumped every time a new contiguous run of
    // token_type_ids == 1 starts; -1 ("no group") for text tokens (token_type_ids == 0).
    std::vector<int32_t> group_id(num_real_new_tokens, -1);
    int32_t current_group = -1;
    bool in_run = false;
    for (uint32_t i = 0; i < num_real_new_tokens; ++i) {
        if (token_type_ids_real[i] == 1) {
            if (!in_run) {
                ++current_group;
                in_run = true;
            }
            group_id[i] = current_group;
        } else {
            in_run = false;
        }
    }

    constexpr float kAttend = 0.0f;
    float* data = mask_tensor->data<float>();
    for (uint32_t i = 0; i < num_real_new_tokens; ++i) {
        if (group_id[i] < 0) {
            continue;
        }
        float* row_ptr = data + static_cast<size_t>(row_pad + i) * col_dim;
        for (uint32_t j = 0; j < num_real_new_tokens; ++j) {
            if (group_id[j] == group_id[i]) {
                row_ptr[past_width + row_pad + j] = kAttend;
            }
        }
    }
}
