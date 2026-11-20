// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_request_utils.hpp"

#include "logging.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/runtime/make_tensor.hpp"  // get_tensor_impl
#include "util.hpp"
#include "util_xarch.hpp"

// FIXME: Use ov::npuw::util::view instead
ov::SoPtr<ov::ITensor> ov::npuw::util::make_tensor_slice(ov::SoPtr<ov::ITensor> tensor,
                                                         uint32_t dim,
                                                         uint32_t start_pos,
                                                         uint32_t end_pos) {
    // `dim` may originate from a deserialized KVCacheDesc and is used below to subscript
    // freshly-allocated ov::Shape vectors. Validate it against the actual tensor rank before
    // the stores, otherwise a malformed axis is an out-of-bounds write on the shape vectors.
    const auto& shape = tensor->get_shape();
    OPENVINO_ASSERT(dim < shape.size(), "KV slice axis ", dim, " is out of range for a rank-", shape.size(), " tensor");
    OPENVINO_ASSERT(start_pos <= end_pos && end_pos <= shape[dim],
                    "KV slice [",
                    start_pos,
                    ", ",
                    end_pos,
                    ") is out of range for extent ",
                    shape[dim]);
    ov::Shape start_shape(std::vector<size_t>(shape.size(), 0u));
    start_shape[dim] = start_pos;
    ov::Shape end_shape = shape;
    end_shape[dim] = end_pos;
    return ov::get_tensor_impl(ov::Tensor(ov::make_tensor(tensor), start_shape, end_shape));
}

void ov::npuw::util::copy_to_right(const ov::SoPtr<ov::ITensor>& src, const ov::SoPtr<ov::ITensor>& dst) {
    if (src->get_element_type() == dst->get_element_type()) {
        OPENVINO_ASSERT(src->get_byte_size() <= dst->get_byte_size());
        std::copy_n(reinterpret_cast<uint8_t*>(src->data()),
                    src->get_byte_size(),
                    reinterpret_cast<uint8_t*>(dst->data()) + dst->get_byte_size() - src->get_byte_size());
        return;
    }
    OPENVINO_ASSERT(src->get_size() <= dst->get_size());
    copy_tensor_elements(src, dst, 0, dst->get_size() - src->get_size(), src->get_size());
}

void ov::npuw::util::copy_tensor_elements(const ov::SoPtr<ov::ITensor>& src,
                                          const ov::SoPtr<ov::ITensor>& dst,
                                          size_t src_offset,
                                          size_t dst_offset,
                                          size_t count) {
    OPENVINO_ASSERT(src_offset <= src->get_size() && count <= src->get_size() - src_offset,
                    "Source element range exceeds tensor size");
    OPENVINO_ASSERT(dst_offset <= dst->get_size() && count <= dst->get_size() - dst_offset,
                    "Destination element range exceeds tensor size");
    const auto src_type = src->get_element_type();
    const auto dst_type = dst->get_element_type();
    OPENVINO_ASSERT(src_type == dst_type || (src_type == ov::element::f32 && dst_type == ov::element::f16),
                    "Unsupported staging conversion from ",
                    src_type,
                    " to ",
                    dst_type);
    if (count == 0) {
        return;
    }
    if (src_type == dst_type) {
        std::copy_n(static_cast<const uint8_t*>(src->data()) + src_offset * src_type.size(),
                    count * src_type.size(),
                    static_cast<uint8_t*>(dst->data()) + dst_offset * dst_type.size());
    } else {
        OPENVINO_ASSERT(src->is_continuous() && dst->is_continuous(), "Staging conversion requires continuous tensors");
        const auto* src_data = src->data<float>() + src_offset;
        auto* dst_data = dst->data<ov::float16>() + dst_offset;
        ov::npuw::util::XARCH::to_f16_buffer(src_data, dst_data, count);
    }
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
        copy_tensor_elements(src, dst, 0, 0, 0);
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

    const auto src_elements = src->get_size();
    const auto dst_elements = dst->get_size();
    OPENVINO_ASSERT(src_seq_len > 0u, "per_layer_inputs src has zero seq_len");
    OPENVINO_ASSERT(dst_seq_len > 0u, "per_layer_inputs dst has zero seq_len");
    OPENVINO_ASSERT(src_elements % src_seq_len == 0u,
                    "per_layer_inputs src size is not divisible by seq_len. size=",
                    src_elements,
                    ", seq_len=",
                    src_seq_len);
    OPENVINO_ASSERT(dst_elements % dst_seq_len == 0u,
                    "per_layer_inputs dst size is not divisible by seq_len. size=",
                    dst_elements,
                    ", seq_len=",
                    dst_seq_len);
    const auto src_per_token_elements = src_elements / src_seq_len;
    const auto dst_per_token_elements = dst_elements / dst_seq_len;
    OPENVINO_ASSERT(src_per_token_elements == dst_per_token_elements,
                    "per-token element count mismatch between src and dst. src=",
                    src_per_token_elements,
                    ", dst=",
                    dst_per_token_elements);

    const size_t chunk_elements = static_cast<size_t>(chunk_tokens) * src_per_token_elements;
    const size_t offset_elements = static_cast<size_t>(src_offset_tokens) * src_per_token_elements;
    copy_tensor_elements(src, dst, offset_elements, dst_elements - chunk_elements, chunk_elements);
}
