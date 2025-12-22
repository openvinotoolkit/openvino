// Copyright (C) 2025 Intel Corporation
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

void ov::npuw::util::copy_inplace_columns_by_row_chunks(ov::SoPtr<ov::ITensor> src, ov::SoPtr<ov::ITensor>& dst) {
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

    const size_t num_chunks = C * H;
    if (num_chunks == 0 || chunk_byte_size == 0) {
        return;
    }

    for (size_t i = num_chunks; i-- > 0;) {
        const size_t src_offset = i * IS_H;
        const size_t dst_offset = i * OS_H;
        std::memmove(dst_p + dst_offset, src_p + src_offset, chunk_byte_size);
    }
}

void ov::npuw::util::copy_inplace_by_planes(ov::SoPtr<ov::ITensor> src_tensor, ov::SoPtr<ov::ITensor> dst_tensor) {
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

    const auto* src_base = reinterpret_cast<uint8_t*>(src_tensor->data());
    auto* dst_base = reinterpret_cast<uint8_t*>(dst_tensor->data());

    const auto num_planes = src_tensor->get_shape()[H];
    const auto src_plane_stride = src_tensor->get_strides()[H];
    const auto dst_plane_stride = dst_tensor->get_strides()[H];
    const auto plane_size_in_bytes = src_tensor->get_strides()[S] * src_tensor->get_shape()[S];

    if (num_planes == 0 || plane_size_in_bytes == 0) {
        return;
    }

    for (size_t i = num_planes; i-- > 0;) {
        const auto* src_ptr = src_base + i * src_plane_stride;
        auto* dst_ptr = dst_base + i * dst_plane_stride;
        std::memmove(dst_ptr, src_ptr, plane_size_in_bytes);
    }
}

void ov::npuw::util::copy_inplace(ov::SoPtr<ov::ITensor> src_tensor, ov::SoPtr<ov::ITensor> dst_tensor) {
    const auto& shape = src_tensor->get_shape();

    auto* base = static_cast<uint8_t*>(src_tensor->data());

    auto src_strides = src_tensor->get_strides();
    auto dst_strides = dst_tensor->get_strides();

    const size_t total_elems = src_tensor->get_size();
    const size_t elem_size = src_tensor->get_byte_size() / total_elems;

    if (src_strides == dst_strides) {
        LOG_INFO("identical strides, skip");
        return;
    }

    for (size_t d = 0; d < shape.size(); ++d) {
        if (shape[d] == 0) {
            LOG_INFO("zero-sized dimension, nothing to move");
            return;
        }
    }

    auto rank = shape.size();

    ov::Shape cur_pos{0};
    ov::Shape max_pos{1};

    if (src_tensor->get_element_type().bitwidth() < 8 || (is_scalar(shape))) {
        // Doesn't support strides for LP types
        // or both tensors have default strides
        // Strides and positions already initialized
    } else {
        ov::Strides src_str, dst_str;
        // Calculate src and dst shapes
        bool found_step = false;
        for (size_t inverted_idx = rank - 1; inverted_idx < rank; --inverted_idx) {
            if (!found_step) {
                if (src_strides[inverted_idx] == dst_strides[inverted_idx]) {
                    continue;
                } else {
                    found_step = true;
                    size_t strides_size = inverted_idx + 1;
                    // Set right size
                    src_str.resize(strides_size + 1);
                    dst_str.resize(strides_size + 1);
                    max_pos.resize(strides_size + 1);
                    cur_pos.resize(strides_size + 1);
                    // In case of default continuous strides we can copy several elements
                    // In other case only one element
                    size_t dim = 1;
                    size_t strides = elem_size;

                    if (strides_size < src_strides.size()) {
                        strides = src_strides[strides_size];
                        dim = shape[strides_size];
                    }
                    src_str[strides_size] = strides;
                    dst_str[strides_size] = strides;
                    max_pos[strides_size] = dim;
                    cur_pos[strides_size] = max_pos[strides_size] - 1;
                }
            }
            src_str[inverted_idx] = src_strides[inverted_idx];
            dst_str[inverted_idx] = dst_strides[inverted_idx];
            max_pos[inverted_idx] = shape[inverted_idx];
            cur_pos[inverted_idx] = max_pos[inverted_idx] - 1;
        }
        src_strides = std::move(src_str);
        dst_strides = std::move(dst_str);
    }

    size_t src_off = 0;
    size_t dst_off = 0;
    for (size_t d = 0; d < max_pos.size(); ++d) {
        src_off += cur_pos[d] * src_strides[d];
        dst_off += cur_pos[d] * dst_strides[d];
    }

    auto dec_index_and_update_offsets = [&]() -> bool {
        for (int d = static_cast<int>(max_pos.size()) - 1; d >= 0; --d) {
            const size_t old = cur_pos[static_cast<size_t>(d)];
            if (old > 0) {
                cur_pos[static_cast<size_t>(d)] = old - 1;
                src_off -= src_strides[static_cast<size_t>(d)];
                dst_off -= dst_strides[static_cast<size_t>(d)];
                return true;
            } else {
                cur_pos[static_cast<size_t>(d)] = max_pos[static_cast<size_t>(d)] - 1;
                src_off += src_strides[static_cast<size_t>(d)] * (max_pos[static_cast<size_t>(d)] - 1);
                dst_off += dst_strides[static_cast<size_t>(d)] * (max_pos[static_cast<size_t>(d)] - 1);
            }
        }
        return false;
    };

    while (true) {
        uint8_t* src_ptr = base + src_off;
        uint8_t* dst_ptr = base + dst_off;

        if (src_ptr != dst_ptr) {
            std::memmove(dst_ptr, src_ptr, src_strides[src_strides.size() - 1]);
        }

        if (!dec_index_and_update_offsets()) {
            break;
        }
    }
}

// In-place move along kv_dim when src/dst share the same buffer.
// Requirements:
//   - kv_dim_src == kv_dim_dst, otherwise throws
//   - src_tensor->data() == dst_tensor->data()
void ov::npuw::util::copy_tensor_inplace_by_dim(ov::SoPtr<ov::ITensor> src_tensor,
                                                ov::SoPtr<ov::ITensor> dst_tensor,
                                                uint32_t kv_dim_src,
                                                uint32_t kv_dim_dst) {
    OPENVINO_ASSERT(src_tensor);
    OPENVINO_ASSERT(dst_tensor);

    if (kv_dim_src != kv_dim_dst) {
        OPENVINO_THROW("move_tensor_inplace_by_dim currently supports only kv_dim_src == kv_dim_dst");
    }

    void* base_data = src_tensor->data();
    void* dst_data = dst_tensor->data();
    OPENVINO_ASSERT(base_data);
    OPENVINO_ASSERT(dst_data);
    OPENVINO_ASSERT(base_data == dst_data);

    const auto& src_shape = src_tensor->get_shape();
    const auto& dst_shape = dst_tensor->get_shape();
    OPENVINO_ASSERT(src_shape.size() == dst_shape.size());
    OPENVINO_ASSERT(src_shape == dst_shape);
    OPENVINO_ASSERT(kv_dim_src < src_shape.size());

    if (kv_dim_src == 3u) {
        copy_inplace_columns_by_row_chunks(src_tensor, dst_tensor);
    } else if (kv_dim_src == 2u) {
        copy_inplace_by_planes(src_tensor, dst_tensor);
    } else {
        copy_inplace(src_tensor, dst_tensor);
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
