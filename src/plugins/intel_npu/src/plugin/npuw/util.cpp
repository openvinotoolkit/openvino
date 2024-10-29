// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "util.hpp"

#include <intel_npu/config/config.hpp>
#include <iomanip>
#include <openvino/core/parallel.hpp>
#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/float16.hpp>
#include <sstream>

#include "logging.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/runtime/make_tensor.hpp"  // get_tensor_impl
#include "util_xarch.hpp"

bool ov::npuw::util::is_set(const std::size_t sub_idx, const std::string& opt, const std::size_t end_idx) {
    if (opt.empty() || opt == "NO") {
        return false;
    }
    if (opt == "YES") {
        return true;
    }

    std::string str(opt);
    std::size_t last_pos = str.find("last");
    if (last_pos != std::string::npos) {
        str.erase(last_pos, 4);
        if (end_idx != SIZE_MAX && sub_idx == end_idx - 1) {
            return true;
        }
    }

    std::vector<std::size_t> sub_inds{};
    sub_inds = ::intel_npu ::OptionParser<std::vector<std::size_t>>::parse(str);
    if (std::find(sub_inds.begin(), sub_inds.end(), sub_idx) != sub_inds.end()) {
        return true;
    }
    return false;
}

namespace {
inline uint8_t hi4(uint8_t x) {
    return x >> 4;
}

inline uint8_t lo4(uint8_t x) {
    return x & 0xF;
}
}  // namespace

ov::Tensor ov::npuw::util::tensor_from_const(const std::shared_ptr<ov::Node>& node) {
    NPUW_ASSERT(ov::op::util::is_constant(node));
    NPUW_ASSERT(node->outputs().size() == 1);
    const auto port = node->output(0);
    auto cnst_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(node);
    return ov::Tensor(port.get_element_type(), port.get_shape(), const_cast<void*>(cnst_node->get_data_ptr()));
}

bool ov::npuw::util::starts_with(const std::string& str, const std::string& prefix) {
    return str.substr(0, prefix.size()) == prefix;
}

std::string ov::npuw::util::fmt(std::size_t number, std::size_t total) {
    std::size_t regs = 1;
    while (total /= 10) {
        regs++;
    }
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(regs) << number;
    return ss.str();
}

void ov::npuw::util::unpack(const ov::SoPtr<ov::ITensor>& from,
                            const ov::SoPtr<ov::ITensor>& to,
                            const UnpackOptions& unpack_options) {
    // This is in fact a weight decompression procedure
    auto type_from = from->get_element_type();
    auto type_to = to->get_element_type();

    namespace ove = ov::element;
#define CAST(x)    static_cast<int>((x).operator ove::Type_t())
#define PAIR(f, t) (CAST(f) << 16 | CAST(t))
#define HNDL(f, t)                                                      \
    case PAIR(ove::f, ove::t):                                          \
        ov::npuw::util::XARCH::unpack_##f##t(from, to, unpack_options); \
        break;
    switch (PAIR(type_from, type_to)) {
        HNDL(i4, i8);
        HNDL(i4, f16);
        HNDL(u4, i8);
        HNDL(u4, f16);
        HNDL(u4, f32);
        HNDL(i8, f16);
    default:
        OPENVINO_THROW("Unknown unpack combination ", type_from, " -> ", type_to);
    }
#undef HNDL
#undef PAIR
#undef CAST
}

void ov::npuw::util::unpack(const ov::SoPtr<ov::ITensor>& from,
                            const ov::SoPtr<ov::ITensor>& scale,
                            const ov::SoPtr<ov::ITensor>& to,
                            const UnpackOptions& unpack_options) {
    // This is in fact a weight decompression procedure
    const auto type_from = from->get_element_type();
    const auto type_to = to->get_element_type();
    NPUW_ASSERT(type_to == ov::element::f16);

    const auto& from_shape = from->get_shape();
    const auto& scale_shape = scale->get_shape();

    if (type_from == ov::element::i4) {
        if (from_shape.size() == 3) {
            if (scale_shape[2] == from_shape[2]) {
                ov::npuw::util::XARCH::unpack_i4f16_z(from, scale, to, unpack_options);
            } else {
                ov::npuw::util::XARCH::unpack_i4f16_scale(from, scale, to, unpack_options);
            }
        } else {
            NPUW_ASSERT(from_shape.size() == 2);
            ov::npuw::util::XARCH::unpack_i4f16_scale(from, scale, to, unpack_options);
        }
    } else if (type_from == ov::element::i8) {
        ov::npuw::util::XARCH::unpack_i8f16_scale(from, scale, to, unpack_options);
    } else {
        NPUW_ASSERT(false && "Unsupported combination");
    }
}

void ov::npuw::util::unpack(const ov::SoPtr<ov::ITensor>& from,
                            const ov::SoPtr<ov::ITensor>& zerop,
                            const ov::SoPtr<ov::ITensor>& scale,
                            const ov::SoPtr<ov::ITensor>& to,
                            const UnpackOptions& unpack_options) {
    const auto type_from = from->get_element_type();
    const auto type_zerop = zerop->get_element_type();
    const auto type_scale = scale->get_element_type();
    const auto type_to = to->get_element_type();

    if (type_from == ov::element::u4) {
        NPUW_ASSERT(type_zerop == ov::element::u4 || type_zerop == ov::element::f16 || type_zerop == ov::element::f32);
        NPUW_ASSERT(type_scale == ov::element::f16 || type_scale == ov::element::f32);
        NPUW_ASSERT(type_to == ov::element::f16);
    } else if (type_from == ov::element::u8) {
        NPUW_ASSERT(type_zerop == ov::element::u8);
        NPUW_ASSERT(type_scale == ov::element::f16);
        NPUW_ASSERT(type_to == ov::element::f16);
    } else {
        NPUW_ASSERT(false && "Unsupported combination");
    }

    // This function determines the appropriate unpacking strategy for tensor multiplication
    // based on the 'scale' shape and 'from' shape.
    // Example tensors -> (scale.*from):
    // unpack_u4f16:
    //     - [4096, 1].*[4096, 4096]
    //     - [11008, 1].*[11008, 4096]
    //     - [4096, 32, 1].*[4096, 32, 128]
    // unpack_u4f16_z:
    //     - [32, 1, 4096].*[32, 128, 4096]
    //     - [32, 1, 11008].*[32, 128, 11008]
    //     - [86, 1, 4096].*[86, 128, 4096]
    // unpack_u4f16_asymm_zp:
    //     - [256, 16, 1].*[256, 16, 128]
    //     - [2048, 16, 1].*[2048, 16, 128]
    //     - [5632, 16, 1].*[5632, 16, 128]
    //      Zero Point Shapes: [256, 16, 1], [2048, 16, 1], [5632, 16, 1]
    // Unsupported Case for scale tensor:
    //     - [s1, 1, s2, 1, s3]

    const auto& from_shape = from->get_shape();
    const auto& scale_shape = scale->get_shape();
    const auto& zerop_shape = zerop->get_shape();

    if (type_from == ov::element::u4) {
        if (scale_shape.size() == 3 && scale_shape[0] == from_shape[0] && scale_shape[1] == 1 &&
            scale_shape[2] == from_shape[2]) {
            ov::npuw::util::XARCH::unpack_u4f16_z(from, zerop, scale, to, unpack_options);
        } else if (scale_shape.size() == 3 && scale_shape[0] == from_shape[0] && scale_shape[1] == from_shape[1] &&
                   scale_shape[2] == 1) {
            if (zerop->get_size() == 1) {
                ov::npuw::util::XARCH::unpack_u4f16_scale_zp(from, zerop, scale, to, unpack_options);
            } else {
                ov::npuw::util::XARCH::unpack_u4f16_asymm_zp(from, zerop, scale, to, unpack_options);
            }
        } else if (scale_shape.size() == 2 && scale_shape[0] == from_shape[0] && scale_shape[1] == 1) {
            ov::npuw::util::XARCH::unpack_u4f16_scale_zp(from, zerop, scale, to, unpack_options);
        } else {
            NPUW_ASSERT(false);
        }
    } else if (type_from == ov::element::u8) {
        if (scale_shape.size() == 3 && scale_shape[1] == 1 && scale_shape[2] == 1) {
            // Special case for broadcasting vocab by 2 dimensions
            // FIXME: all this logic probably should be in some specific unpack or another util function
            const auto& from_strides = from->get_strides();
            const auto& zerop_strides = zerop->get_strides();
            const auto& scale_strides = scale->get_strides();
            ov::Tensor wraped_from(from->get_element_type(),
                                   ov::Shape{from_shape[0], from_shape[1] * from_shape[2]},
                                   from->data(),
                                   ov::Strides{from_strides[0], from_strides[2]});
            ov::Tensor wraped_zerop(zerop->get_element_type(),
                                    ov::Shape{zerop_shape[0], zerop_shape[1] * zerop_shape[2]},
                                    zerop->data(),
                                    ov::Strides{zerop_strides[0], zerop_strides[2]});
            ov::Tensor wraped_scale(scale->get_element_type(),
                                    ov::Shape{scale_shape[0], scale_shape[1] * scale_shape[2]},
                                    scale->data(),
                                    ov::Strides{scale_strides[0], scale_strides[2]});

            ov::npuw::util::XARCH::unpack_u8f16(ov::get_tensor_impl(wraped_from),
                                                ov::get_tensor_impl(wraped_zerop),
                                                ov::get_tensor_impl(wraped_scale),
                                                to,
                                                unpack_options);
        } else if (scale_shape.size() == 2 && scale_shape[0] == from_shape[0] && scale_shape[1] == 1) {
            ov::npuw::util::XARCH::unpack_u8f16(from, zerop, scale, to, unpack_options);
        } else {
            NPUW_ASSERT(false);
        }
    }
}

void ov::npuw::util::gather(const ov::SoPtr<ov::ITensor>& src,
                            const ov::SoPtr<ov::ITensor>& idx,
                            const ov::SoPtr<ov::ITensor>& dst) {
    const auto src_type = src->get_element_type();
    const auto dst_type = dst->get_element_type();
    NPUW_ASSERT(idx->get_element_type() == ov::element::i64);
    NPUW_ASSERT(src_type == ov::element::f16 || src_type == ov::element::f32);
    NPUW_ASSERT(src_type == dst_type);

    const auto& idx_shape = idx->get_shape();
    NPUW_ASSERT(idx_shape.size() == 2);
    NPUW_ASSERT(idx_shape[0] == 1);

    const auto& src_shape = src->get_shape();
    NPUW_ASSERT(src_shape.size() == 2);

    const auto& dst_shape = dst->get_shape();
    NPUW_ASSERT(dst_shape.size() == 3);
    NPUW_ASSERT(src_shape[1] == dst_shape[2]);

    const int64_t* pIdx = idx->data<int64_t>();
    const uint8_t* pSrc = static_cast<uint8_t*>(src->data());
    uint8_t* pDst = static_cast<uint8_t*>(dst->data());

    for (std::size_t r = 0; r < idx_shape[1]; r++) {
        auto srcRowIdx = pIdx[r];
        auto pSrcRow = pSrc + src_shape[1] * srcRowIdx * src_type.size();
        std::copy_n(pSrcRow, src_shape[1] * src_type.size(), pDst);
        pDst += dst_shape[2] * dst_type.size();
    }
}

ov::SoPtr<ov::ITensor> ov::npuw::util::view(const ov::SoPtr<ov::ITensor>& src,
                                            const ov::npuw::util::View& from,
                                            const ov::npuw::util::View& to) {
    const auto type = src->get_element_type();
    NPUW_ASSERT(from.size() == to.size());

    // Sub-byte views are not supported here
    NPUW_ASSERT(type != ov::element::u4 && type != ov::element::i4);

    const auto num_dims = from.size();
    ov::Shape view_shape;
    for (auto d = 0u; d < num_dims; d++) {
        view_shape.push_back(to[d] - from[d]);
    }

    const auto strides = src->get_strides();
    uint8_t* ptr = static_cast<uint8_t*>(src->data());

    // Shift PTR according to the strides
    for (auto d = 0u; d < num_dims; d++) {
        ptr += strides[d] * from[d];
    }

    ov::Tensor viewt(type, view_shape, ptr, strides);
    return ov::get_tensor_impl(viewt);
}

ov::SoPtr<ov::ITensor> ov::npuw::util::view(const ov::SoPtr<ov::ITensor>& src,
                                            std::size_t dim,
                                            std::size_t offset,
                                            std::size_t len) {
    const auto shape = src->get_shape();
    View view_start = View(shape.size(), 0u);
    View view_end = shape;
    view_start[dim] = offset;
    view_end[dim] = offset + len;
    return ov::npuw::util::view(src, view_start, view_end);
}

template <typename InT>
void to_f32(const ov::Tensor& in, ov::Tensor& out) {
    NPUW_ASSERT(in.is_continuous());
    NPUW_ASSERT(out.is_continuous());
    NPUW_ASSERT(in.get_shape() == out.get_shape());

    if (ov::element::Type_t::f32 == in.get_element_type()) {
        in.copy_to(out);
        return;
    }

    const InT* in_buffer = in.data<InT>();
    NPUW_ASSERT(in_buffer != nullptr);
    const auto out_buffer = out.data<float>();
    NPUW_ASSERT(out_buffer != nullptr);

    // NOTE: ov::parallel_for takes care of splitting the work among threads such way,
    //       that the passed lambda function will be called sequentially
    //       on some part of "in.get_size()" range inside the each thread
    ov::parallel_for(in.get_size(), [in_buffer, out_buffer](int64_t index) {
        out_buffer[index] = static_cast<float>(in_buffer[index]);
    });
}

void ov::npuw::util::to_f32(const ov::Tensor& in, ov::Tensor& out) {
    switch (in.get_element_type()) {
    case ov::element::Type_t::f32:
        ::to_f32<float>(in, out);
        break;
    case ov::element::Type_t::u64:
        ::to_f32<uint64_t>(in, out);
        break;
    case ov::element::Type_t::i64:
        ::to_f32<int64_t>(in, out);
        break;
    case ov::element::Type_t::u32:
        ::to_f32<uint32_t>(in, out);
        break;
    case ov::element::Type_t::i32:
        ::to_f32<int32_t>(in, out);
        break;
    case ov::element::Type_t::u16:
        ::to_f32<uint16_t>(in, out);
        break;
    case ov::element::Type_t::i16:
        ::to_f32<int16_t>(in, out);
        break;
    case ov::element::Type_t::u8:
        ::to_f32<uint8_t>(in, out);
        break;
    case ov::element::Type_t::i8:
        ::to_f32<int8_t>(in, out);
        break;
    case ov::element::Type_t::f16:
        ::to_f32<ov::float16>(in, out);
        break;
    case ov::element::Type_t::bf16:
        ::to_f32<ov::bfloat16>(in, out);
        break;
    default:
        OPENVINO_THROW("Unsupported precision {0}", in.get_element_type().get_type_name());
        break;
    }
}

ov::Tensor ov::npuw::util::to_f16(const ov::Tensor& t) {
    return ov::npuw::util::XARCH::to_f16(t);
}

inline uint8_t tread_4b(const ov::Tensor& t, std::size_t r, std::size_t c, std::size_t COLS) {
    const uint8_t* tdata = static_cast<uint8_t*>(t.data());
    const uint8_t* trow = tdata + r * COLS / 2;
    const uint8_t* telem = trow + c / 2;
    if (c % 2 == 0) {
        return lo4(*telem);
    }
    return hi4(*telem);
}

inline void twrite_4b(ov::Tensor& t, uint8_t value, std::size_t r, std::size_t c, std::size_t COLS) {
    uint8_t* tdata = static_cast<uint8_t*>(t.data());
    uint8_t* trow = tdata + r * COLS / 2;
    uint8_t* telem = trow + c / 2;
    if (c % 2 == 0) {
        *telem = (hi4(*telem) << 4) | lo4(value);
    } else {
        *telem = (lo4(value) << 4) | lo4(*telem);
    }
}

ov::Tensor ov::npuw::util::transpose(const ov::Tensor& t) {
    ov::Shape shape = t.get_shape();
    NPUW_ASSERT(shape.size() == 3);  // Yes, so far only transpose 3D tensors
    NPUW_ASSERT(t.get_element_type() == ov::element::i4);

    ov::Shape tshape = {shape[2], shape[0], shape[1]};
    ov::Tensor tnew(t.get_element_type(), tshape);

    const auto IN_ROWS = shape[0] * shape[1];
    const auto IN_COLS = shape[2];
    for (std::size_t i = 0; i < IN_ROWS; i++) {
        for (std::size_t j = 0; j < IN_COLS; j++) {
            uint8_t value = tread_4b(t, i, j, IN_COLS);
            twrite_4b(tnew, value, j, i, IN_ROWS);
        }
    }
    return tnew;
}

template <typename T>
void permute120(const ov::Tensor& src, ov::Tensor& dst) {
    const ov::Shape src_shape = src.get_shape();
    const ov::Shape dst_shape = dst.get_shape();
    NPUW_ASSERT(src_shape.size() == 3);  // Yes, so far only transpose 3D tensors

    const T* pSrc = static_cast<T*>(src.data());
    T* pDst = static_cast<T*>(dst.data());

    // DSTs [b,r,c] map to SRC's [r,c,b]

    for (std::size_t b = 0; b < dst_shape[0]; b++) {
        for (std::size_t r = 0; r < dst_shape[1]; r++) {
            for (std::size_t c = 0; c < dst_shape[2]; c++) {
                auto dst_idx = b * dst_shape[1] * dst_shape[2] + r * dst_shape[2] + c;
                auto src_idx = r * src_shape[1] * src_shape[2] + c * src_shape[1] + b;
                pDst[dst_idx] = pSrc[src_idx];
            }
        }
    }
}

ov::Tensor ov::npuw::util::permute(const ov::Tensor& t, const std::vector<std::size_t>& axes) {
    ov::Shape shape = t.get_shape();
    NPUW_ASSERT(shape.size() == 3);  // Yes, so far only transpose 3D tensors

    if (axes[0] == 2 && axes[1] == 0 && axes[2] == 1) {
        return transpose(t);
    } else if (axes[0] == 0 && axes[1] == 2 && axes[2] == 1) {
        NPUW_ASSERT(t.get_element_type() == ov::element::i4);  // 4bit only here
        ov::Shape tshape = {shape[0], shape[2], shape[1]};
        ov::Tensor tnew(t.get_element_type(), tshape);

        for (std::size_t p = 0; p < shape[0]; p++) {
            for (std::size_t r = 0; r < shape[1]; r++) {
                for (std::size_t c = 0; c < shape[2]; c++) {
                    uint8_t value = tread_4b(t, p * shape[1] + r, c, shape[2]);
                    twrite_4b(tnew, value, p * shape[2] + c, r, shape[1]);
                }
            }
        }
        return tnew;
    } else if (axes[0] == 1 && axes[1] == 0 && axes[2] == 2) {
        NPUW_ASSERT(t.get_element_type() == ov::element::i4);  // 4bit only here too
        ov::Shape tshape = {shape[1], shape[0], shape[2]};
        ov::Tensor tnew(t.get_element_type(), tshape);

        // Iterate over output tensor coordinates
        for (std::size_t p = 0; p < tshape[0]; p++) {
            for (std::size_t r = 0; r < tshape[1]; r++) {
                for (std::size_t c = 0; c < tshape[2]; c++) {
                    uint8_t value = tread_4b(t, r, p * shape[2] + c, shape[1] * shape[2]);
                    twrite_4b(tnew, value, p * tshape[1] + r, c, tshape[2]);
                }
            }
        }
        return tnew;
    } else if (axes[0] == 1 && axes[1] == 2 && axes[2] == 0) {
        ov::Shape tshape = {shape[1], shape[2], shape[0]};
        ov::Tensor tnew(t.get_element_type(), tshape);
        switch (t.get_element_type()) {
        case ov::element::f32:
            permute120<uint32_t>(t, tnew);
            break;
        case ov::element::f16:
            permute120<uint16_t>(t, tnew);
            break;
        default:
            NPUW_ASSERT("Element type is not supported yet");
        }
        return tnew;
    } else {
        NPUW_ASSERT(false && "Not supported yet");
    }
}

ov::Tensor ov::npuw::util::concat(const std::vector<ov::Tensor>& tt, std::size_t axis) {
    NPUW_ASSERT(axis == 0 || axis == 2);

    const auto type = tt.front().get_element_type();
    auto shape = tt.front().get_shape();
    std::size_t new_dim = 0;

    std::vector<std::size_t> offsets;
    std::vector<std::size_t> lens;
    for (auto&& t : tt) {
        NPUW_ASSERT(tt.front().get_element_type() == t.get_element_type());
        NPUW_ASSERT(t.is_continuous());

        auto tshape = t.get_shape();
        for (std::size_t d = 0; d < tshape.size(); d++) {
            if (d != axis) {
                NPUW_ASSERT(shape[d] == tshape[d]);
            } else {
                offsets.push_back(new_dim);
                lens.push_back(tshape[d]);
                new_dim += tshape[d];
            }
        }
    }
    shape[axis] = new_dim;

    if (axis == 0) {
        ov::Tensor tnew(tt.front().get_element_type(), shape);
        uint8_t* pDst = static_cast<uint8_t*>(tnew.data());

        const bool is_4bit = (type == ov::element::i4 || type == ov::element::u4);
        for (std::size_t t_idx = 0; t_idx < tt.size(); t_idx++) {
            const uint8_t* pSrc = static_cast<uint8_t*>(tt[t_idx].data());

            const auto copy_size = lens[t_idx] * shape[1] * shape[2];
            const auto copy_len = is_4bit ? copy_size / 2 : copy_size * type.size();

            std::copy_n(pSrc, copy_len, pDst);
            pDst += copy_len;
        }
        return tnew;
    } else if (axis == 2) {
        ov::Tensor tnew(tt.front().get_element_type(), shape);
        uint8_t* pDst = static_cast<uint8_t*>(tnew.data());

        const bool is_4bit = (type == ov::element::i4 || type == ov::element::u4);
        for (std::size_t t_idx = 0; t_idx < tt.size(); t_idx++) {
            const auto& t_src = tt[t_idx];

            for (std::size_t r = 0; r < shape[0] * shape[1]; r++) {
                const auto r_offset = is_4bit ? new_dim * r / 2 : new_dim * r * type.size();
                const auto c_offset = is_4bit ? offsets[t_idx] / 2 : offsets[t_idx] * type.size();
                const auto copy_len = is_4bit ? lens[t_idx] / 2 : lens[t_idx] * type.size();
                uint8_t* pDstRow = pDst + r_offset + c_offset;

                const auto r_offset_src = is_4bit ? lens[t_idx] * r / 2 : lens[t_idx] * r * type.size();
                const uint8_t* pSrc = static_cast<uint8_t*>(t_src.data());
                const uint8_t* pSrcRow = pSrc + r_offset_src;

                std::copy_n(pSrcRow, copy_len, pDstRow);
            }
        }
        return tnew;
    } else {
        NPUW_ASSERT(false && "Not supported yet");
    }
}

namespace {
template <typename T>
ov::npuw::util::range_1d validMaskRange(const T* data, std::size_t len) {
    using R = ov::npuw::util::range_1d;
    std::size_t range_begin = 0u;
    bool was_set = false;

    for (std::size_t idx = 0u; idx < len; idx++) {
        const bool is_set = static_cast<std::size_t>(data[idx] > 0);

        if (is_set && !was_set) {
            was_set = true;
            range_begin = idx;
        } else if (!is_set && was_set) {
            return R{range_begin, idx};
        }
    }
    return was_set ? R{range_begin, len} : R{0u, 0u};
}
}  // namespace

ov::npuw::util::range_1d ov::npuw::util::validMaskRange(const ov::SoPtr<ov::ITensor>& src) {
    NPUW_ASSERT(src->is_continuous());

    namespace ove = ov::element;
#define HNDL(t, T) \
    case ove::t:   \
        return ::validMaskRange(static_cast<const T*>(src->data()), src->get_size());
    switch (src->get_element_type()) {
        HNDL(i64, int64_t);
        HNDL(i32, int32_t);
    default:
        OPENVINO_THROW("Unsupported type ", src->get_element_type());
    }
#undef HNDL
}
