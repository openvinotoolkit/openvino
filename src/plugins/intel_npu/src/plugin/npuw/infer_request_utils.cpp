// // Copyright (C) 2025 Intel Corporation
// // SPDX-License-Identifier: Apache-2.0
// //

// #include "infer_request_utils.hpp"

// #include "logging.hpp"
// #include "openvino/runtime/make_tensor.hpp"  // get_tensor_impl
// #include "util_xarch.hpp"

// // FIXME: Use ov::npuw::util::view instead
// ov::SoPtr<ov::ITensor> ov::npuw::util::make_tensor_slice(ov::SoPtr<ov::ITensor> tensor,
//                                                          uint32_t dim,
//                                                          uint32_t start_pos,
//                                                          uint32_t end_pos) {
//     ov::Shape start_shape(std::vector<size_t>(tensor->get_shape().size(), 0u));
//     start_shape[dim] = start_pos;
//     ov::Shape end_shape = tensor->get_shape();
//     end_shape[dim] = end_pos;
//     return ov::get_tensor_impl(ov::Tensor(ov::make_tensor(tensor), start_shape, end_shape));
// }

// void ov::npuw::util::copy_to_right(const ov::SoPtr<ov::ITensor>& src, const ov::SoPtr<ov::ITensor>& dst) {
//     OPENVINO_ASSERT(src->get_byte_size() <= dst->get_byte_size());
//     std::copy_n(reinterpret_cast<uint8_t*>(src->data()),
//                 src->get_byte_size(),
//                 reinterpret_cast<uint8_t*>(dst->data()) + dst->get_byte_size() - src->get_byte_size());
// }

// void ov::npuw::util::copy_by_planes(ov::SoPtr<ov::ITensor> src_tensor, ov::SoPtr<ov::ITensor> dst_tensor) {
//     // [1, H, S1, E] -> [1, H, S2, E]
//     const int N = 0;
//     const int H = 1;
//     const int S = 2;
//     const int E = 3;

//     OPENVINO_ASSERT(src_tensor->get_shape()[N] == dst_tensor->get_shape()[N]);
//     OPENVINO_ASSERT(src_tensor->get_shape()[H] == dst_tensor->get_shape()[H]);
//     OPENVINO_ASSERT(src_tensor->get_shape()[E] == dst_tensor->get_shape()[E]);
//     OPENVINO_ASSERT(src_tensor->get_element_type() == dst_tensor->get_element_type());
//     OPENVINO_ASSERT(src_tensor->get_shape()[N] == 1u);
//     OPENVINO_ASSERT(src_tensor->get_shape().size() == 4u);

//     const auto* src_tensor_data = reinterpret_cast<uint8_t*>(src_tensor->data());
//     auto* dst_tensor_data = reinterpret_cast<uint8_t*>(dst_tensor->data());

//     const auto num_planes = src_tensor->get_shape()[H];
//     const auto src_plane_stride = src_tensor->get_strides()[H];
//     const auto dst_plane_stride = dst_tensor->get_strides()[H];
//     const auto plane_size_in_bytes = src_tensor->get_strides()[S] * src_tensor->get_shape()[S];

//     for (size_t i = 0; i < num_planes; ++i) {
//         std::copy_n(src_tensor_data, plane_size_in_bytes, dst_tensor_data);
//         dst_tensor_data += dst_plane_stride;
//         src_tensor_data += src_plane_stride;
//     }
// }

// void ov::npuw::util::copy_columns_by_row_chunks(ov::SoPtr<ov::ITensor> src, ov::SoPtr<ov::ITensor>& dst) {
//     /*
//       src/dst layout: [1, heads, emb_size, seq_len]

//       X[*,i] - embedding for i-th token,
//       Instead of copy columns, copy rows X[i,*]

//       [[X00 X01 ... X0n]      [[X00 X01 ... X0n]
//        [X10 X11 ... X1n]       [X10 X11 ... X1n]
//        [X20 X21 ... X2n]  ...  [X20 X21 ... X2n]
//              ...                     ...
//        [Xm0 Xm1 ... Xmn]]      [Xm0 Xm1 ... Xmn]]
//     */

//     const auto& src_shape = src->get_shape();

//     OPENVINO_ASSERT(src_shape.size() == 4u);
//     OPENVINO_ASSERT(src_shape == dst->get_shape());
//     OPENVINO_ASSERT(src->get_byte_size() == dst->get_byte_size());

//     const auto& src_strides = src->get_strides();
//     const auto& dst_strides = dst->get_strides();
//     const auto elem_size = src->get_byte_size() / src->get_size();

//     const auto C = src_shape[1];
//     const auto H = src_shape[2];
//     const auto W = src_shape[3];

//     const auto IS_H = src_strides[2];
//     const auto OS_H = dst_strides[2];

//     const size_t chunk_byte_size = W * elem_size;

//     const auto* src_p = static_cast<uint8_t*>(src->data());
//     auto* dst_p = static_cast<uint8_t*>(dst->data());

//     for (size_t i = 0; i < C * H; ++i) {
//         const size_t src_offset = i * IS_H;
//         const size_t dst_offset = i * OS_H;
//         std::copy_n(src_p + src_offset, chunk_byte_size, dst_p + dst_offset);
//     }
// }

// void ov::npuw::util::copy_tensor_by_dim(ov::SoPtr<ov::ITensor> src_tensor,
//                                         ov::SoPtr<ov::ITensor> dst_tensor,
//                                         uint32_t kv_dim_src,
//                                         uint32_t kv_dim_dst) {
//     if (kv_dim_src != kv_dim_dst) {
//         // new case - do a generic copy for now (in fact it is a permute)
//         // Example:
//         //   kv_dim_src         kv_dim_dst
//         //       v                     v
//         // [1,8,256,128] --> [1,8,128,256]
//         const auto& src_shape = src_tensor->get_shape();
//         const auto& dst_shape = dst_tensor->get_shape();
//         NPUW_ASSERT(src_shape.size() == 4);
//         NPUW_ASSERT(dst_shape.size() == 4);
//         NPUW_ASSERT(kv_dim_src < 4);
//         NPUW_ASSERT(kv_dim_dst < 4);
//         NPUW_ASSERT(src_shape[kv_dim_src] == dst_shape[kv_dim_dst]);

//         std::array<int, 4> axis = {0, 1, 2, 3};
//         // Remap like 0,1,2,3 => 0,1,3,2 (see example)
//         std::swap(axis[kv_dim_src], axis[kv_dim_dst]);
//         ov::npuw::util::permute_i4d(src_tensor, dst_tensor, axis);
//         return;
//     }
//     // Old behavior
//     NPUW_ASSERT(kv_dim_src == kv_dim_dst);
//     if (kv_dim_src == 3u) {
//         // Asserting that we work with last dimenston here:
//         const auto& src_shape = src_tensor->get_shape();
//         OPENVINO_ASSERT(src_shape.size() == 4);
//         // If last dimenstion of src_tensor is equal to 1, then we can squeeze
//         // src_shape from [1, heads, d_v, seq_len=1] to [heads, d_v].
//         // We can then treat src_tensor as a continuous tensor of row value vectors
//         // for multiple heads, while dst_tensor will still have [1, heads, d_v, seq_len!=1],
//         // shape, awaiting updates at column dimension, as value vectors are columns now.
//         if (src_shape[kv_dim_src] == 1 && src_tensor->is_continuous()) {
//             // FIXME: ov::npuw::util::XARCH::copy_row_as_column(src_tensor, dst_tensor) throws when used here
//             copy_columns_by_row_chunks(src_tensor, dst_tensor);
//         } else {
//             copy_columns_by_row_chunks(src_tensor, dst_tensor);
//         }
//     } else if (kv_dim_src == 2u) {
//         copy_by_planes(src_tensor, dst_tensor);
//     } else {
//         src_tensor->copy_to(dst_tensor._ptr);
//     }
// }

// std::optional<ov::Output<const ov::Node>> ov::npuw::util::find_port_by_name(
//     const std::vector<ov::Output<const ov::Node>>& ports,
//     const std::string& name) {
//     auto it = std::find_if(ports.begin(), ports.end(), [&](const auto& port) {
//         return port.get_names().count(name) != 0;
//     });
//     if (it == ports.end()) {
//         return std::nullopt;
//     }
//     return std::make_optional(*it);
// }

//////////////////////////////////////////////////////////////////////
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

void ov::npuw::util::copy_inplace_generic_rows(const ov::SoPtr<ov::ITensor> src_tensor,
                                               ov::SoPtr<ov::ITensor> dst_tensor) {
    OPENVINO_ASSERT(src_tensor);
    OPENVINO_ASSERT(dst_tensor);

    void* base_data = src_tensor->data();
    void* dst_data = dst_tensor->data();
    OPENVINO_ASSERT(base_data && dst_data);
    OPENVINO_ASSERT(base_data == dst_data);

    const auto& shape0 = src_tensor->get_shape();
    const auto& dst_shape0 = dst_tensor->get_shape();
    OPENVINO_ASSERT(shape0 == dst_shape0);

    const size_t rank0 = shape0.size();
    if (rank0 == 0) {
        return;
    }

    for (size_t d = 0; d < rank0; ++d) {
        if (shape0[d] == 0) {
            return;
        }
    }

    const size_t total_elems = src_tensor->get_size();
    OPENVINO_ASSERT(total_elems != 0);
    const size_t elem_size = src_tensor->get_byte_size() / total_elems;

    ov::Strides src_strides0 = src_tensor->get_strides();
    ov::Strides dst_strides0 = dst_tensor->get_strides();
    OPENVINO_ASSERT(src_strides0.size() == rank0);
    OPENVINO_ASSERT(dst_strides0.size() == rank0);

    // Build default byte strides for given shape (same as ov::ITensor::copy_to logic).
    ov::Strides default_strides(rank0, 0);
    default_strides[rank0 - 1] = elem_size;
    for (size_t i = rank0 - 1; i > 0; --i) {
        default_strides[i - 1] = default_strides[i] * shape0[i];
    }

    // Your explicit preconditions:
    OPENVINO_ASSERT(src_strides0[rank0 - 1] == elem_size);
    OPENVINO_ASSERT(dst_strides0[rank0 - 1] == elem_size);
    OPENVINO_ASSERT(default_strides[rank0 - 1] == elem_size);

    if (rank0 >= 2) {
        const size_t packed = shape0[rank0 - 1] * elem_size;
        OPENVINO_ASSERT(src_strides0[rank0 - 2] == packed);
        OPENVINO_ASSERT(dst_strides0[rank0 - 2] == packed);
        OPENVINO_ASSERT(default_strides[rank0 - 2] == packed);
    }

    // Find the COMMON trailing segment where src_stride == dst_stride == default_stride.
    // This is the only part eligible for flattening.
    size_t cut = rank0 - 1;  // at worst, we can always copy along last dim
    for (size_t inverted_idx = rank0 - 1; inverted_idx < rank0; --inverted_idx) {
        const bool ok = (src_strides0[inverted_idx] == default_strides[inverted_idx]) &&
                        (dst_strides0[inverted_idx] == default_strides[inverted_idx]) &&
                        (src_strides0[inverted_idx] == dst_strides0[inverted_idx]);
        if (ok) {
            cut = inverted_idx;
            if (inverted_idx == 0) {
                break;
            }
            continue;
        }
        break;
    }

    // Fold [cut..rank0-1] into a single last dimension.
    ov::Shape shape;
    ov::Strides src_strides;
    ov::Strides dst_strides;

    shape.reserve(cut + 1);
    src_strides.reserve(cut + 1);
    dst_strides.reserve(cut + 1);

    for (size_t d = 0; d < cut; ++d) {
        shape.push_back(shape0[d]);
        src_strides.push_back(src_strides0[d]);
        dst_strides.push_back(dst_strides0[d]);
    }

    size_t folded_last = 1;
    for (size_t d = cut; d < rank0; ++d) {
        folded_last *= shape0[d];
    }
    shape.push_back(folded_last);

    // For the folded last dim, the step is element-size (bytes per element).
    // (Since the whole folded tail is default-contiguous, this holds.)
    src_strides.push_back(elem_size);
    dst_strides.push_back(elem_size);

    const size_t rank = shape.size();
    OPENVINO_ASSERT(rank >= 1);

    const size_t row_elems = shape[rank - 1];
    const size_t row_bytes = row_elems * elem_size;
    if (row_bytes == 0) {
        return;
    }

    // Iterate outer coordinates in reverse lexicographic order for overlap-safe memmove.
    size_t num_rows = 1;
    for (size_t d = 0; d + 1 < rank; ++d) {
        num_rows *= shape[d];
    }
    if (num_rows == 0) {
        return;
    }

    auto* base = static_cast<uint8_t*>(base_data);

    ov::Shape idx(rank - 1, 0);
    for (size_t d = 0; d + 1 < rank; ++d) {
        idx[d] = shape[d] - 1;
    }

    auto compute_offset = [&](const ov::Shape& outer, const ov::Strides& strides_bytes) -> size_t {
        size_t off = 0;
        for (size_t d = 0; d < outer.size(); ++d) {
            off += outer[d] * strides_bytes[d];
        }
        return off;
    };

    auto dec_outer = [&]() -> bool {
        for (int d = static_cast<int>(rank) - 2; d >= 0; --d) {
            const size_t ud = static_cast<size_t>(d);
            if (idx[ud] > 0) {
                --idx[ud];
                return true;
            }
            idx[ud] = shape[ud] - 1;
        }
        return false;
    };

    while (true) {
        const size_t src_off = compute_offset(idx, src_strides);
        const size_t dst_off = compute_offset(idx, dst_strides);

        uint8_t* src_ptr = base + src_off;
        uint8_t* dst_ptr = base + dst_off;
        if (src_ptr != dst_ptr) {
            std::memmove(dst_ptr, src_ptr, row_bytes);
        }

        if (!dec_outer()) {
            break;
        }
    }
}

// In-place move along kv_dim when src/dst share the same buffer.
// Requirements:
//   - kv_dim_src == kv_dim_dst, otherwise throws
//   - src_tensor->data() == dst_tensor->data()
void ov::npuw::util::copy_tensor_inplace_by_dim(const ov::SoPtr<ov::ITensor> src_tensor,
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

    // One generic implementation for all kv_dim.
    // We rely on row-wise memmove on the (possibly flattened) last dimension and stride-based addressing.
    copy_inplace_generic_rows(src_tensor, dst_tensor);
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