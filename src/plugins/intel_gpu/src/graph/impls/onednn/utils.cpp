// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"
#include <oneapi/dnnl/dnnl_debug.h>
#include <numeric>
#include <oneapi/dnnl/dnnl_ocl.hpp>

namespace cldnn {
namespace onednn {

template <typename T>
cldnn::memory::ptr convert_zp_data_to_s32(const memory::ptr zp_memory) {
    auto engine = zp_memory->get_engine();
    auto& stream = engine->get_service_stream();

    auto zp_s32_layout = zp_memory->get_layout();
    zp_s32_layout.data_type = data_types::i32;
    auto zp_s32_memory = engine->allocate_memory(zp_s32_layout, false);

    mem_lock<T, mem_lock_type::read> zp_data(zp_memory, stream);
    mem_lock<int32_t, mem_lock_type::write> zp_s32_data(zp_s32_memory, stream);
    for (size_t i = 0; i < zp_data.size(); i++) {
        zp_s32_data.data()[i] = static_cast<int32_t>(zp_data.data()[i]);
    }

    return zp_s32_memory;
}

template cldnn::memory::ptr convert_zp_data_to_s32<int8_t>(const memory::ptr zp_memory);
template cldnn::memory::ptr convert_zp_data_to_s32<uint8_t>(const memory::ptr zp_memory);
template cldnn::memory::ptr convert_zp_data_to_s32<int32_t>(const memory::ptr zp_memory);

cldnn::format default_fmt_for_dims(size_t dims, bool is_grouped) {
    switch (dims) {
    case 6: return is_grouped ? cldnn::format::goizyx : cldnn::format::bfwzyx;
    case 5: return is_grouped ? cldnn::format::goiyx : cldnn::format::bfzyx;
    default: return cldnn::format::bfyx;
    }
}

dnnl::memory::dims convert_tensor(cldnn::tensor t, size_t dims, bool is_grouped) {
    auto sizes = t.sizes(default_fmt_for_dims(dims, is_grouped));
    dnnl::memory::dims res(sizes.begin(), sizes.end());
    return res;
}

dnnl::memory::dims convert_gemm_tensor(cldnn::tensor t, size_t dims, bool batched_dims_can_be_removed) {
    auto sizes = t.sizes(default_fmt_for_dims(dims, false));
    dnnl::memory::dims res(sizes.begin(), sizes.end());
    if (dims > 4) {
        for (size_t i = 0; i < dims - 4; i++) {
            res[i + 1] *= res[i];
        }
        res.erase(res.begin(), res.begin() + dims - 4);
    }
    if (res.size() == 4 && batched_dims_can_be_removed) {
        res.erase(res.begin(), res.begin() + 2);
    }
    return res;
}

dnnl::memory::dims convert_gemm_dims(const std::vector<ov::Dimension::value_type> &sizes, size_t dims, bool batched_dims_can_be_removed) {
    dnnl::memory::dims res(sizes.begin(), sizes.end());
    if (dims > 4) {
        for (size_t i = 0; i < dims - 4; i++) {
            res[i + 1] *= res[i];
        }
        res.erase(res.begin(), res.begin() + dims - 4);
    }
    if (res.size() == 4 && batched_dims_can_be_removed) {
        res.erase(res.begin(), res.begin() + 2);
    }
    return res;
}

dnnl::memory::format_tag get_default_data_format(const cldnn::layout& l) {
    switch (l.get_partial_shape().size()) {
    case 2: return dnnl::memory::format_tag::ab;
    case 3: return dnnl::memory::format_tag::abc;
    case 4: return dnnl::memory::format_tag::abcd;
    case 5: return dnnl::memory::format_tag::abcde;
    case 6: return dnnl::memory::format_tag::abcdef;
    default: OPENVINO_THROW("[GPU] Unsupported conversion from "+ l.to_short_string() + " to onednn format_tag");
    }
}

dnnl::memory::format_tag convert_gemm_data_format(dnnl::memory::dims dims, format target) {
    if (dims.size() == target.dimension()) {
        auto tag = convert_data_format(target);
        if (tag != dnnl::memory::format_tag::undef) {
            return tag;
        } else {
            throw std::invalid_argument("[clDNN] Unsupported conversion from "+ target.to_string() + " to onednn format_tag");
        }
    } else {
        switch (dims.size()) {
        case 2: return dnnl::memory::format_tag::ab;
        case 3: return dnnl::memory::format_tag::abc;
        case 4: return dnnl::memory::format_tag::abcd;
        default: throw std::invalid_argument("[clDNN] Unsupported conversion from "+ std::to_string(dims.size()) + " to onednn format_tag");
        }
    }
}

dnnl::memory::dims convert_spatials(cldnn::tensor t, size_t dims) {
    auto spatials = t.spatial;
    dnnl::memory::dims res(dims);
    for (size_t i = 0; i < dims; i++) {
        res[i] = spatials[dims - i - 1];
    }
    return res;
}

dnnl::memory::dims flatten_tensor(cldnn::tensor t) {
    return {static_cast<int64_t>(t.count())};
}

dnnl::memory::dims get_strides(dnnl::memory::dims dims) {
    dnnl::memory::dims strides(dims.size(), dnnl::memory::dim(1));
    std::partial_sum(dims.rbegin(), dims.rend() - 1, strides.rbegin() + 1, std::multiplies<dnnl::memory::dim>());
    return strides;
}

dnnl::memory::data_type convert_data_type(cldnn::data_types dt) {
    switch (dt) {
        case cldnn::data_types::f32: return dnnl::memory::data_type::f32;
        case cldnn::data_types::f16: return dnnl::memory::data_type::f16;
        case cldnn::data_types::i8: return dnnl::memory::data_type::s8;
        case cldnn::data_types::u8: return dnnl::memory::data_type::u8;
        case cldnn::data_types::i32: return dnnl::memory::data_type::s32;
        case cldnn::data_types::i4: return dnnl::memory::data_type::s4;
        case cldnn::data_types::u4: return dnnl::memory::data_type::u4;
        default: throw std::invalid_argument("[clDNN] Unsupported conversion from cldnn to onednn type");
    }
}

std::vector<std::pair<cldnn::format, dnnl::memory::format_tag>> format_map = {
        /// weights format for onednnn
        { cldnn::format::oiyx,  dnnl::memory::format_tag::oihw },
        { cldnn::format::ioyx,  dnnl::memory::format_tag::iohw },
        { cldnn::format::yxio,  dnnl::memory::format_tag::hwio },
        { cldnn::format::oizyx, dnnl::memory::format_tag::oidhw },
        { cldnn::format::iozyx, dnnl::memory::format_tag::iodhw },
        { cldnn::format::iyxo,  dnnl::memory::format_tag::ihwo },
        { cldnn::format::oyxi,  dnnl::memory::format_tag::ohwi },
        { cldnn::format::oyix,  dnnl::memory::format_tag::acbd },
        { cldnn::format::oxiy,  dnnl::memory::format_tag::adbc },
        { cldnn::format::goiyx,  dnnl::memory::format_tag::goihw },
        { cldnn::format::gioyx,  dnnl::memory::format_tag::giohw },
        { cldnn::format::gyxio,  dnnl::memory::format_tag::ghwio },
        { cldnn::format::giozyx, dnnl::memory::format_tag::giodhw },
        { cldnn::format::goizyx,  dnnl::memory::format_tag::goidhw },

        { cldnn::format::os_iyx_osv16,  dnnl::memory::format_tag::Oihw16o },
        { cldnn::format::gs_oiyx_gsv16,  dnnl::memory::format_tag::Goihw16g },
        { cldnn::format::gs_oiyx_gsv32,  dnnl::memory::format_tag::Goihw32g },
        { cldnn::format::gs_oizyx_gsv16,  dnnl::memory::format_tag::Goidhw16g },
        { cldnn::format::g_os_iyx_osv16,  dnnl::memory::format_tag::gOihw16o },

        { cldnn::format::os_is_yx_osv16_isv16,  dnnl::memory::format_tag::OIhw16o16i },
        { cldnn::format::os_is_yx_isv16_osv16,  dnnl::memory::format_tag::OIhw16i16o },
        { cldnn::format::os_is_zyx_isv16_osv16,  dnnl::memory::format_tag::OIdhw16i16o },
        { cldnn::format::is_os_zyx_isv16_osv16,  dnnl::memory::format_tag::IOdhw16i16o },
        { cldnn::format::is_os_yx_isv16_osv16,  dnnl::memory::format_tag::IOhw16i16o },

        { cldnn::format::g_os_is_zyx_isv16_osv16,  dnnl::memory::format_tag::gIOdhw16i16o },

        { cldnn::format::bfyx,  dnnl::memory::format_tag::nchw },
        { cldnn::format::bfxy,  dnnl::memory::format_tag::abdc },
        { cldnn::format::byxf,  dnnl::memory::format_tag::nhwc },
        { cldnn::format::byfx,  dnnl::memory::format_tag::acbd },
        { cldnn::format::bxfy,  dnnl::memory::format_tag::adbc },
        { cldnn::format::fyxb,  dnnl::memory::format_tag::bcda },
        { cldnn::format::xbfy,  dnnl::memory::format_tag::dabc },
        { cldnn::format::fybx,  dnnl::memory::format_tag::bcad },
        { cldnn::format::ybfx,  dnnl::memory::format_tag::cabd },
        { cldnn::format::fbyx,  dnnl::memory::format_tag::bacd },
        { cldnn::format::bfzyx, dnnl::memory::format_tag::ncdhw },
        { cldnn::format::bzyxf, dnnl::memory::format_tag::ndhwc },
        { cldnn::format::bfwzyx, dnnl::memory::format_tag::abcdef },
        { cldnn::format::b_fs_yx_fsv2, dnnl::memory::format_tag::undef },
        { cldnn::format::b_fs_yx_fsv4, dnnl::memory::format_tag::aBcd4b },
        { cldnn::format::b_fs_yx_fsv8, dnnl::memory::format_tag::aBcd8b },
        { cldnn::format::b_fs_yx_fsv16, dnnl::memory::format_tag::nChw16c },
        { cldnn::format::b_fs_yx_fsv32, dnnl::memory::format_tag::aBcd32b },
        { cldnn::format::b_fs_zyx_fsv4, dnnl::memory::format_tag::aBcde4b },
        { cldnn::format::b_fs_zyx_fsv8, dnnl::memory::format_tag::aBcde8b },
        { cldnn::format::b_fs_zyx_fsv16, dnnl::memory::format_tag::nCdhw16c },
        { cldnn::format::b_fs_zyx_fsv32, dnnl::memory::format_tag::aBcde32b },
        { cldnn::format::bs_fs_yx_bsv16_fsv16, dnnl::memory::format_tag::NChw16n16c },
        { cldnn::format::bs_fs_yx_bsv16_fsv32, dnnl::memory::format_tag::NChw16n32c },
        { cldnn::format::bs_fs_yx_bsv32_fsv32, dnnl::memory::format_tag::NChw32n32c },
        { cldnn::format::bs_fs_yx_bsv4_fsv4, dnnl::memory::format_tag::ABcd4a4b },
        { cldnn::format::bs_fs_yx_bsv8_fsv4, dnnl::memory::format_tag::ABcd8a4b },
        { cldnn::format::bs_fs_yx_bsv8_fsv2, dnnl::memory::format_tag::ABcd8a2b },
        { cldnn::format::bs_fs_yx_bsv4_fsv2, dnnl::memory::format_tag::ABcd4a2b },
        { cldnn::format::bs_fs_yx_bsv32_fsv16, dnnl::memory::format_tag::NChw32n16c },
        { cldnn::format::bs_fs_zyx_bsv32_fsv16, dnnl::memory::format_tag::NCdhw32n16c },
        { cldnn::format::bs_fs_zyx_bsv32_fsv32, dnnl::memory::format_tag::NCdhw32n32c },
        { cldnn::format::bs_fs_zyx_bsv16_fsv16, dnnl::memory::format_tag::NCdhw16n16c },
        // { cldnn::format::bs_fs_zyx_bsv16_fsv32, dnnl::memory::format_tag::NCdhw16n32c }, // TODO onednn3.0: Request NCdhw16n32c format
        { cldnn::format::bs_fs_zyx_bsv8_fsv4, dnnl::memory::format_tag::ABcde8a4b },
        { cldnn::format::bs_fs_zyx_bsv8_fsv2, dnnl::memory::format_tag::ABcde8a2b },
};

dnnl::memory::format_tag convert_data_format(cldnn::format fmt) {
    auto ret = std::find_if(format_map.begin(), format_map.end(),
            [fmt](std::pair<cldnn::format, dnnl::memory::format_tag> &e) {
                    return e.first == fmt; });
    if (ret == format_map.end()) {
        GPU_DEBUG_INFO << "[clDNN] Unsupported conversion from "+ fmt.to_string() + " to onednn format_tag. Any tag will be used instead." << std::endl;
        return dnnl::memory::format_tag::any;
    }

    return ret->second;
}

 cldnn::format convert_data_format(dnnl::memory::format_tag fmt) {
    auto ret = std::find_if(format_map.begin(), format_map.end(),
            [fmt](std::pair<cldnn::format, dnnl::memory::format_tag> &e) {
                    return e.second == fmt; });
    if (ret == format_map.end())
        throw std::invalid_argument("[clDNN] Unsupported onednn layout");

    return ret->first;
}

void combine_bf_with_first_spatial_dim(cldnn::layout& l) {
    auto pshape = l.get_partial_shape();
    ov::Shape new_shape{1, 1};
    for (size_t i = 0; i < pshape.size(); ++i) {
        if (i < 2) {
            new_shape[0] *= pshape[i].get_length();
        } else {
            new_shape[1] *= pshape[i].get_length();
        }
    }
    l.set_partial_shape(new_shape);
}

int64_t get_offset(const cldnn::layout& l, dnnl::memory::desc&& desc) {
    int64_t offset = 0;
    auto b_padding = l.data_padding._lower_size[0];
    auto f_padding = l.data_padding._lower_size[1];
    if (b_padding != 0) {
        auto input_pitches = l.get_pitches();
        offset = static_cast<int64_t>(b_padding) * input_pitches[0];
    } else if (f_padding != 0) {
        offset = f_padding;
        for (size_t i = 0; i < l.get_spatial_rank(); ++i) {
            offset *= l.spatial(i);
        }
    }

    switch (desc.get_data_type()) {
        case dnnl::memory::data_type::s4:
        case dnnl::memory::data_type::u4:
            return offset / 2;
        case dnnl::memory::data_type::s8:
        case dnnl::memory::data_type::u8:
            return offset;
        case dnnl::memory::data_type::f16:
        case dnnl::memory::data_type::bf16:
            return (offset * 2);
        case dnnl::memory::data_type::f32:
        case dnnl::memory::data_type::s32:
            return (offset * 4);
        default:
            throw std::runtime_error(std::string("Unsupported offset for dnnl_data_type_t ")
                    + dnnl_dt2str(static_cast<dnnl_data_type_t>(desc.get_data_type())));
    }
}

std::tuple<dnnl::memory::desc, dnnl::memory::desc, dnnl::memory::desc>
get_conv_memory_descs(cldnn::layout input_layout, cldnn::layout weights_layout, cldnn::layout output_layout, dnnl::memory::format_tag target_fmt) {
    mem_flags flag = (input_layout.format.is_blocked() || output_layout.format.is_blocked()) ? mem_flags::need_blocked : mem_flags::None;
    flag = format::is_grouped(weights_layout.format) ? mem_flags::grouped : flag;
    dnnl::memory::desc input_desc   = layout_to_memory_desc(input_layout, target_fmt, flag);
    dnnl::memory::desc weights_desc = layout_to_memory_desc(weights_layout, dnnl::memory::format_tag::any, flag);
    dnnl::memory::desc output_desc  = layout_to_memory_desc(output_layout, target_fmt, flag);
    return {input_desc, weights_desc, output_desc};
}

inline mem_flags operator|(mem_flags lhs, mem_flags rhs) {
    using T = std::underlying_type_t<mem_flags>;
    return static_cast<mem_flags>(static_cast<T>(lhs) | static_cast<T>(rhs));
}

inline mem_flags& operator|=(mem_flags& lhs, mem_flags rhs) {
    lhs = lhs | rhs;
    return lhs;
}

inline mem_flags operator&(mem_flags lhs, mem_flags rhs) {
    using T = std::underlying_type_t<mem_flags>;
    return static_cast<mem_flags>(static_cast<T>(lhs) & static_cast<T>(rhs));
}

inline bool has_mem_flag(mem_flags flags, mem_flags flag_to_check) {
    return (flags & flag_to_check) != mem_flags::None;
}

dnnl::memory::desc layout_to_memory_desc(cldnn::layout l, dnnl::memory::format_tag target_fmt, mem_flags flags) {
    dnnl::memory::dims dims;
    bool flatten = has_mem_flag(flags, mem_flags::flatten);
    bool use_strides = has_mem_flag(flags, mem_flags::use_strides);
    bool need_blocked = has_mem_flag(flags, mem_flags::need_blocked);
    bool is_grouped = has_mem_flag(flags, mem_flags::grouped);
    if (target_fmt == dnnl::memory::format_tag::ab && flatten) {
        dims = flatten_tensor(l.get_tensor());
        dims.insert(dims.begin(), 1);
    } else if (target_fmt == dnnl::memory::format_tag::ab) {
        dims.push_back(l.batch());
        dims.push_back(l.get_tensor().count() / l.batch());
    } else if (target_fmt == dnnl::memory::format_tag::abc) {
        dims.push_back(l.batch());
        dims.push_back(l.feature());
        dims.push_back(l.spatial(1));
    } else if (target_fmt == dnnl::memory::format_tag::acb) {
        dims.push_back(l.batch());
        dims.push_back(l.spatial(1));
        dims.push_back(l.feature());
    } else if (target_fmt == dnnl::memory::format_tag::abdc) {
        dims.push_back(l.batch());
        dims.push_back(l.feature());
        dims.push_back(l.spatial(0));
        dims.push_back(l.spatial(1));
    } else if (target_fmt == dnnl::memory::format_tag::abced) {
        dims.push_back(l.batch());
        dims.push_back(l.feature());
        dims.push_back(l.spatial(2));
        dims.push_back(l.spatial(0));
        dims.push_back(l.spatial(1));
    } else if (target_fmt == dnnl::memory::format_tag::abcdfe) {
        dims.push_back(l.batch());
        dims.push_back(l.feature());
        dims.push_back(l.spatial(3));
        dims.push_back(l.spatial(2));
        dims.push_back(l.spatial(0));
        dims.push_back(l.spatial(1));
    } else if (target_fmt == dnnl::memory::format_tag::ba) {
        dims.push_back(l.feature());
        dims.push_back(l.get_tensor().count() / l.feature());
    } else if (flatten) {
        dims = flatten_tensor(l.get_tensor());
    } else {
        // clDNN expresses 3d tensor with 4d format. This code is to use 3d format on oneDNN for such case.
        // However, if the memory::desc to be converted is related to another blocked format, it should be expanded to a 4d tensor.
        auto shape_rank = l.is_dynamic() ?
            static_cast<size_t>(l.get_partial_shape().rank().get_length()) : l.get_shape().size();
        if (shape_rank == 3 && !need_blocked && !is_grouped) {
            dims.push_back(l.batch());
            dims.push_back(l.feature());
            // In cldnn::layer, when it is a 3D shape, the values ​​of the XY axes can sometimes be flipped,
            // so the larger value of the two is used.
            dims.push_back(std::max(l.spatial(0), l.spatial(1)));
            target_fmt = dnnl::memory::format_tag::abc;
        } else {
            auto rank = cldnn::format::dimension(l.format);
            dims = convert_tensor(l.get_tensor(), rank, cldnn::format::is_grouped(l.format));
        }
    }

    dnnl::memory::data_type dt = convert_data_type(l.data_type);
    if (use_strides) {
        dnnl::memory::dims strides;
        OPENVINO_ASSERT(flatten == false, "The padded layout cannot be flattened.");
        auto padded_dims = l.get_padded_dims();
        if (target_fmt == dnnl::memory::format_tag::ab) {
            strides.push_back(1);
            strides.push_back(padded_dims[0]);
        } else if (target_fmt == dnnl::memory::format_tag::abc) {
            strides.push_back(1);
            strides.push_back(padded_dims[0]);
            strides.push_back(padded_dims[0] * padded_dims[1]);
        } else if (target_fmt == dnnl::memory::format_tag::acb) {
            strides.push_back(1);
            strides.push_back(padded_dims[0]);
            strides.push_back(padded_dims[0] * padded_dims[2]);
        } else if (target_fmt == dnnl::memory::format_tag::abdc) {
            strides.push_back(1);
            strides.push_back(padded_dims[0]);
            strides.push_back(padded_dims[0] * padded_dims[1]);
            strides.push_back(padded_dims[0] * padded_dims[1] * padded_dims[3]);
        } else if (target_fmt == dnnl::memory::format_tag::abced) {
            strides.push_back(1);
            strides.push_back(padded_dims[0]);
            strides.push_back(padded_dims[0] * padded_dims[1]);
            strides.push_back(padded_dims[0] * padded_dims[1] * padded_dims[2]);
            strides.push_back(padded_dims[0] * padded_dims[1] * padded_dims[2] * padded_dims[4]);
        } else if (target_fmt == dnnl::memory::format_tag::abcdfe) {
            strides.push_back(1);
            strides.push_back(padded_dims[0]);
            strides.push_back(padded_dims[0] * padded_dims[1]);
            strides.push_back(padded_dims[0] * padded_dims[1] * padded_dims[2]);
            strides.push_back(padded_dims[0] * padded_dims[1] * padded_dims[2] * padded_dims[3]);
            strides.push_back(padded_dims[0] * padded_dims[1] * padded_dims[2] * padded_dims[3] * padded_dims[5]);
        } else if (target_fmt == dnnl::memory::format_tag::ba) {
            strides.push_back(1);
            strides.push_back(padded_dims[1]);
        } else {
            auto pitches = l.get_pitches();
            strides.assign(pitches.begin(), pitches.end());
        }
        dnnl::memory::desc res(dims, dt, strides);
        return res;
    } else {
        dnnl::memory::format_tag fmt = target_fmt == dnnl::memory::format_tag::undef ? convert_data_format(l.format) : target_fmt;
        dnnl::memory::desc res(dims, dt, fmt);
        return res;
    }
}
static void get_identical_order(std::vector<std::vector<size_t>>& orders, std::vector<size_t> order,
                            size_t first, size_t depth) {
    if (depth == 0)
        return;

    for (size_t idx = first; idx <= first + depth ; idx++) {
        std::swap(order[first], order[idx]);
        if (first != idx)
            orders.push_back(order);

        get_identical_order(orders, order, first+1, depth-1);
        std::swap(order[first], order[idx]);
    }
}
// Get candidate orders calculated by stride value of dnnl::memory::descriptor could be multiple
std::vector<std::vector<size_t>> get_candidate_orders(dnnl::memory::desc desc) {
    std::vector<std::vector<size_t>> orders;
    auto strides = desc.get_strides();
    std::vector<size_t> order(desc.get_ndims());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
                [&strides] (size_t ind_l, size_t ind_r) {
                    return (strides[ind_l] > strides[ind_r]);
                });

    orders.push_back(order);

    // Orders of those axes which have a same stride in memory::desc can be changed.
    // If y and x axes have same, then it can be bfyx or bfxy.
    for (size_t idx = 0 ; idx+1 < order.size() ; idx++) {
        size_t depth = 0;
        for (size_t next = idx+1 ; next < order.size() ; next++) {
            if (strides[order[idx]] == strides[order[next]]) {
                depth++;
            } else {
                break;
            }
        }

        // mutiple axes can have a same stride value of mem descriptor
        get_identical_order(orders, order, idx, depth);
        idx += depth;
    }

    return orders;
}

static bool compare_orders(std::vector<std::vector<size_t>> a, std::vector<size_t> b) {
    for (size_t idx = 0 ; idx < a.size() ; idx++) {
        if (std::equal(a[idx].begin(), a[idx].end(), b.begin()))
            return true;
    }

    return false;
}

cldnn::format find_data_format(dnnl::memory::desc desc) {
    auto order = get_candidate_orders(desc);

    for (int32_t fmt_idx = format::bfyx ; fmt_idx < format::oiyx ; fmt_idx++) {
        auto candidate_trait = format::traits(static_cast<format::type>(fmt_idx));
        if (desc.get_ndims() == static_cast<int>(candidate_trait._order.size())
            && desc.get_inner_nblks() == static_cast<int>(candidate_trait.block_sizes.size())
            && compare_orders(order, candidate_trait._order)) {
            bool is_match = true;
            for (size_t idx = 0 ; idx < candidate_trait.block_sizes.size() ; idx++) {
                if (desc.get_inner_blks()[idx] != static_cast<int>(candidate_trait.block_sizes[idx].second)
                    || desc.get_inner_idxs()[idx] != static_cast<int>(candidate_trait.block_sizes[idx].first)) {
                    is_match = false;
                    break;
                }
            }
            if (is_match)
                return static_cast<format::type>(fmt_idx);
        }
    }

    if (desc.get_ndims() == 3 && desc.get_inner_nblks() == 0) {
        // Special case for 3D tensors without blocking
        if (compare_orders(order, { {0, 1, 2} })) {
            return static_cast<format::type>(format::bfyx);
        } else if (compare_orders(order, { {0, 2, 1} })) {
            return static_cast<format::type>(format::byxf);
        } else if (compare_orders(order, { {1, 0, 2} })) {
            return static_cast<format::type>(format::fbyx);
        } else if (compare_orders(order, { {1, 2, 0} })) {
            return static_cast<format::type>(format::fybx);
        }
    }

    std::stringstream msg;
    msg << "Unsupported onednn dnnl::memory::desc find_data_format. "
        << "ndims: " << desc.get_ndims()
        << ", inner_nblks: " << desc.get_inner_nblks()
        << ", inner_blks: ";
    for (int i = 0; i < desc.get_inner_nblks(); i++)
        msg << "(blk " << desc.get_inner_blks()[i] << ", idx " << desc.get_inner_idxs()[i] << ") ";

    throw std::runtime_error(msg.str());
}

cldnn::format find_format(dnnl::memory::desc desc, bool is_grouped) {
    auto orders = get_candidate_orders(desc);

    format start_format = format::oiyx;
    if (is_grouped)
        start_format = format::goiyx;

    for (int32_t fmt_idx = start_format ; fmt_idx < format::format_num ; fmt_idx++) {
        auto candidate_trait = format::traits(static_cast<format::type>(fmt_idx));
        if (static_cast<size_t>(desc.get_ndims()) == candidate_trait._order.size()
            && static_cast<size_t>(desc.get_inner_nblks()) == candidate_trait.block_sizes.size()
            && compare_orders(orders, candidate_trait._order)) {
            // Compare all pairs of dimension number and block size to format_traits_map of all formats
            bool is_match = true;
            for (size_t idx = 0 ; idx < candidate_trait.block_sizes.size() ; idx++) {
                auto block_idx = static_cast<dnnl_dim_t>(candidate_trait.block_sizes[idx].first);
                auto block_size = static_cast<dnnl_dim_t>(candidate_trait.block_sizes[idx].second);
                if (is_grouped && candidate_trait.is_group_char(candidate_trait.internal_order[block_idx])) {
                    // inner_idx gets the index of group dimension in mem::desc when blocked axis is group
                    auto inner_idx = candidate_trait.order.find_first_of(candidate_trait.internal_order[block_idx]);
                    if (desc.get_inner_blks()[idx] != block_size ||
                        desc.get_inner_idxs()[idx] != static_cast<dnnl_dim_t>(inner_idx)) {
                        is_match = false;
                        break;
                    }
                } else if (is_grouped) {
                    // g,o,i from cldnn formats are matching to a,b,c of dnnl. But g is at the end of internal order.
                    if (desc.get_inner_blks()[idx] != block_size ||
                        (desc.get_inner_idxs()[idx] - static_cast<dnnl_dim_t>(candidate_trait.group_num)) != block_idx) {
                        is_match = false;
                        break;
                    }
                } else {
                    if (desc.get_inner_blks()[idx] != block_size ||
                        desc.get_inner_idxs()[idx] != block_idx) {
                        is_match = false;
                        break;
                    }
                }
            }

            if (is_match)
                return static_cast<format::type>(fmt_idx);
        }
    }

    std::stringstream msg;
    msg << "Unsupported " << (is_grouped ? "grouped" : "") << "onednn dnnl::memory::desc find_format. "
        << "ndims: " << desc.get_ndims()
        << ", inner_nblks: " << desc.get_inner_nblks()
        << ", inner_blks: ";
    for (int i = 0; i < desc.get_inner_nblks(); i++)
        msg << "(blk " << desc.get_inner_blks()[i] << ", idx " << desc.get_inner_idxs()[i] << ") ";
    for (auto order : orders) {
        msg << ", strides_order : ";

        for (const auto& value : order)
            msg << value << " ";
    }
    msg << ", stride_value : ";
    auto strides = desc.get_strides();
    for (size_t idx = 0; idx < orders[0].size() ; idx++) {
        msg << strides[idx] << " ";
    }

    throw std::runtime_error(msg.str());
}

// Currently, usage of alpha and beta between cldnn::pow and dnnl::eltwise::pow is different : d = pow(src, a) / d = a * pow(src, b)
dnnl::algorithm convert_activation_func(cldnn::activation_func func) {
    switch (func) {
        case cldnn::activation_func::relu: return dnnl::algorithm::eltwise_relu;
        case cldnn::activation_func::relu_negative_slope: return dnnl::algorithm::eltwise_relu;
        case cldnn::activation_func::gelu: return dnnl::algorithm::eltwise_gelu_erf;
        case cldnn::activation_func::gelu_tanh: return dnnl::algorithm::eltwise_gelu_tanh;
        case cldnn::activation_func::elu: return dnnl::algorithm::eltwise_elu;
        case cldnn::activation_func::mish: return dnnl::algorithm::eltwise_mish;
        case cldnn::activation_func::swish: return dnnl::algorithm::eltwise_swish;
        case cldnn::activation_func::hswish: return dnnl::algorithm::eltwise_hardswish;
        case cldnn::activation_func::abs: return dnnl::algorithm::eltwise_abs;
        case cldnn::activation_func::exp: return dnnl::algorithm::eltwise_exp;
        case cldnn::activation_func::logistic: return dnnl::algorithm::eltwise_logistic;
        case cldnn::activation_func::clamp: return dnnl::algorithm::eltwise_clip;
        case cldnn::activation_func::hyperbolic_tan: return dnnl::algorithm::eltwise_tanh;
        case cldnn::activation_func::pow: return dnnl::algorithm::eltwise_pow;
        case cldnn::activation_func::sqrt: return dnnl::algorithm::eltwise_sqrt;
        case cldnn::activation_func::square: return dnnl::algorithm::eltwise_square;
        case cldnn::activation_func::hard_sigmoid: return dnnl::algorithm::eltwise_hardsigmoid;
        // Activations that are undef algorithms must be converted to other activations before pushing to post-op.
        case cldnn::activation_func::hsigmoid: return dnnl::algorithm::undef;
        case cldnn::activation_func::negative: return dnnl::algorithm::undef;
        default: throw std::runtime_error("Unsupported activation func for onednn primitive " + std::to_string(static_cast<int>(func)));
    }
}

template <typename T>
bool is_per_tensor(cldnn::data_node& node, int32_t& zp_val) {
    auto ptr = node.get_attached_memory_ptr();
    auto engine = ptr->get_engine();
    auto& stream = engine->get_service_stream();
    auto num_elems = node.get_output_layout().count();
    mem_lock<T, mem_lock_type::read> old_data {ptr, stream};
    auto val = old_data[0];
    for (size_t i = 1; i < num_elems; i++) {
        if (val != old_data[i]) {
            zp_val = DNNL_RUNTIME_S32_VAL;
            return false;
        }
    }

    zp_val = val;
    return true;
}

template bool is_per_tensor<int8_t>(cldnn::data_node& node, int32_t& zp_val);
template bool is_per_tensor<uint8_t>(cldnn::data_node& node, int32_t& zp_val);
template bool is_per_tensor<int32_t>(cldnn::data_node& node, int32_t& zp_val);


static std::string get_external_order(const std::vector<size_t>& order, bool is_weights, bool is_grouped) {
    cldnn::format default_fmt = format::get_default_format(order.size(), is_weights, is_grouped);
    const auto& default_order = default_fmt.order();

    std::string external_order(order.size(), '?');

    for (size_t i = 0; i < order.size(); i++) {
        external_order[i] = default_order[order[i]];
    }

    return external_order;
}

cldnn::format_traits convert_memory_desc_to_traits(const dnnl::memory::desc& desc, bool is_weights, bool is_grouped) {
    OPENVINO_ASSERT(desc.get_format_kind() == dnnl::memory::format_kind::blocked, "[GPU] Only blocked memory desc type is supported");
    auto ndims = desc.get_ndims();
    auto inner_nblks = desc.get_inner_nblks();
    auto inner_blks = desc.get_inner_blks();
    auto inner_idxs = desc.get_inner_idxs();
    auto strides = desc.get_strides();

    std::vector<std::pair<int64_t, size_t>> stride_order;
    for (size_t i = 0; i < strides.size(); i++) {
        stride_order.emplace_back(strides[i], i);
    }

    // sort by strides in descending order
    std::sort(stride_order.begin(), stride_order.end(), [](const std::pair<int64_t, size_t>& first, const std::pair<int64_t, size_t>& second) {
        return first.first > second.first;
    });

    std::vector<size_t> order;
    for (const auto& p : stride_order) {
        order.push_back(p.second);
    }

    std::vector<std::pair<size_t, int>> block_sizes(inner_nblks);
    for (int i = 0; i < inner_nblks; i++) {
        block_sizes[i] = std::make_pair(inner_idxs[i] + (is_grouped && inner_idxs[i] == 0 ? 9 : 0) + (is_grouped ? -1 : 0), inner_blks[i]);
    }

    // all fmts has at least batch and feature dim for now
    const int batch_num = 1;
    const int feature_num = 1;
    const int group_num = is_grouped ? 1 : 0;
    const int spatial_size = std::max<int>(ndims - batch_num - feature_num - group_num, 0);

    std::string internal_order = is_weights ?
                                    (is_grouped ? "oixyz???g" : "oixyz") :
                                    "bfxyzwuv";

    const size_t max_spatial = 2 + (is_weights ? 3 : 6);
    const size_t last_spatial_offset = 2 + spatial_size;
    for (size_t i = last_spatial_offset; i < max_spatial; i++) {
        internal_order[i] = '?';
    }
    std::string outer_order = get_external_order(order, is_weights, is_grouped);

    std::vector<std::pair<size_t, int>> logic_block_sizes(inner_nblks);
    for (int i = 0; i < inner_nblks; i++) {
        auto c = internal_order[block_sizes[i].first];
        auto pos = outer_order.find(c);
        OPENVINO_ASSERT(pos != std::string::npos, "[GPU] Unknown coord type: ", c);

        logic_block_sizes[i] = std::make_pair(order[pos], inner_blks[i]);
    }

    format_traits traits;
    traits.batch_num = batch_num;
    traits.feature_num = feature_num;
    traits.spatial_num = spatial_size;
    traits.group_num = group_num;
    traits._order = order;
    traits.order = outer_order;
    traits.internal_order = internal_order;
    traits.block_sizes = block_sizes;
    traits.logic_block_sizes = logic_block_sizes;
    traits.desc_size = desc.get_size();
    traits.str = "custom";

    return traits;
}

/*
 * This function checks the consistency between the input and output shapes of the onednn reorder.
 * If the shape is expanded from 4D to 5D, the format of the input layout is also updated accordingly.
 */
bool keep_weights_reorder_shape_consistent(cldnn::layout& layout, const dnnl::memory::desc& desc) {
    if (layout.is_dynamic())
        return false;

    auto shape = layout.get_shape();
    auto dims = desc.get_dims();
    std::vector<ov::Dimension::value_type> target_dims;
    std::vector<ov::Dimension::value_type> filtered_target_dims;
    std::transform(shape.begin(), shape.end(), std::back_inserter(target_dims),
                   [](size_t v) { return static_cast<ov::Dimension::value_type>(v); });
    std::copy_if(target_dims.begin(), target_dims.end(), std::back_inserter(filtered_target_dims),
                 [](ov::Dimension::value_type i) { return i != 1; });

    std::vector<ov::Dimension::value_type> desc_dims;
    std::vector<ov::Dimension::value_type> filtered_desc_dims;
    std::transform(dims.cbegin(), dims.cend(), std::back_inserter(desc_dims),
                   [](dnnl::memory::dim v) { return static_cast<ov::Dimension::value_type>(v); });
    std::copy_if(desc_dims.begin(), desc_dims.end(), std::back_inserter(filtered_desc_dims),
                 [](ov::Dimension::value_type i) { return i != 1; });

    if (filtered_target_dims != filtered_desc_dims)
        return false; // We cannot keep the shapes consistent

    layout.set_partial_shape(desc_dims);
    if (layout.get_rank() == desc_dims.size()) {
        return true;
    } else if (layout.get_rank() == 4 && desc_dims.size() == 3) {
        // In the case of a 3D shape, cldnn::layout::get_rank() returns 4.
        return true;
    } else if (layout.get_rank() == 4 && desc_dims.size() == 5) {
        // Since onednn does not support 1D group convolution, a z-axis is added, and format change is required in this case.
        auto is_weights = cldnn::format::is_weights_format(layout.format);
        auto is_grouped = cldnn::format::is_grouped(layout.format);
        auto expected_default_format = cldnn::format::get_default_format(layout.get_rank(), is_weights, is_grouped);
        // Dimension expansion is only allowed when the input layout is in the default format.
        if (layout.format == expected_default_format) {
            layout.format = cldnn::format::get_default_format(desc_dims.size(), is_weights, is_grouped);
            return true;
        } else {
            OPENVINO_ASSERT(false, "Need default format for axis expansion.");
        }
    } else {
        return false;
    }
}

size_t get_post_ops_count(const program_node& node) {
    size_t onednn_post_ops_count = 0;
    for (auto& fo : node.get_fused_primitives()) {
       onednn_post_ops_count += fo.f_param->ops_count();
    }

    return onednn_post_ops_count;
}

bool is_supported_post_ops(const program_node& node) {
    if (get_post_ops_count(node) > 32) {
        return false;
    }

    for (auto& fo : node.get_fused_primitives()) {
        if (fo.is_type<activation>()) {
            // Some activations aren't implemented in oneDNN
            auto activation_prim = fo.typed_desc<activation>();
            if (activation_prim->activation_function == activation_func::negative ||
                activation_prim->activation_function == activation_func::negation ||
                activation_prim->activation_function == activation_func::sign)
                return false;
        }
    }

    return true;
}

bool is_supported_pad(const layout& layout) {
    if (!layout.data_padding)
        return true;

    const auto& pad = layout.data_padding;
    // Check spatial padding
    bool no_spatial_padding = true;
    auto spatial_rank = layout.get_spatial_rank();
    for (size_t i = 0; i < spatial_rank; ++i) {
        no_spatial_padding &= (pad._lower_size[2 + i] == 0);
        no_spatial_padding &= (pad._upper_size[2 + i] == 0);
    }

    // Onednn supports outer padding of batch axis (first element offset) if its format is 'bxxx'
    bool no_batch_padding = true;
    auto fmt = layout.format;
    if (format::is_multi_blocked(fmt) || fmt.dims_order()[0] != 0 || fmt.dims_order()[0] != 0) {
        no_batch_padding &= (pad._lower_size[0] == 0);
        no_batch_padding &= (pad._upper_size[0] == 0);
    }

    return (no_spatial_padding && no_batch_padding);
}

int onednn_post_ops_fusing_helpers::get_prelu_mask_from_layouts(const std::function<layout()>& get_output_layout,
                                                                const std::function<layout(int32_t)>& get_input_layout,
                                                                int32_t slope_input_idx) {
    auto data_layout = get_output_layout();
    auto slope_layout = get_input_layout(slope_input_idx);
    auto input_layout = get_input_layout(0);
    auto data_shape = data_layout.get_shape();
    auto slope_shape = slope_layout.get_shape();
    auto input_shape = input_layout.get_shape();
    int ndims = slope_shape.size() == 1 ? static_cast<int>(input_shape.size()) : static_cast<int>(slope_shape.size());

    bool is_scalar = true;
    bool is_per_tensor = true;
    for (size_t i = 0; i < slope_shape.size(); i++) {
        if (slope_shape[i] != 1)
            is_scalar = false;
        if (slope_shape[i] != data_shape[i])
            is_per_tensor = false;
    }

    if (is_scalar)
        return get_default_mask(post_op_dnnl_policy_type::COMMON, ndims);
    else if (slope_shape.size() != 1 && is_per_tensor)
        return get_default_mask(post_op_dnnl_policy_type::PER_TENSOR, ndims);
    else
        return get_default_mask(post_op_dnnl_policy_type::PER_OC, ndims);
}

int onednn_post_ops_fusing_helpers::get_default_mask(post_op_dnnl_policy_type policy, int ndims) {
    switch (policy) {
        case post_op_dnnl_policy_type::PER_DIM_0: return (1 << 0);
        case post_op_dnnl_policy_type::PER_OC:
        case post_op_dnnl_policy_type::PER_DIM_1: return (1 << 1);
        case post_op_dnnl_policy_type::PER_OCIC:
        case post_op_dnnl_policy_type::PER_DIM_01: return (1 << 0) + (1 << 1);
        case post_op_dnnl_policy_type::PER_DIM_2: return (1 << 2);
        case post_op_dnnl_policy_type::PER_DIM_3: return (1 << 3);
        case post_op_dnnl_policy_type::PER_TENSOR:
            OPENVINO_ASSERT(ndims > 0 && ndims <= DNNL_MAX_NDIMS, "ndims is bigger than DNNL_MAX_NDIMS.");
            return (1 << ndims) - 1;
        case post_op_dnnl_policy_type::COMMON: return 0;
        default: OPENVINO_THROW("Incorrect post_op_dnnl_policy_type");
    }
}

}  // namespace onednn
}  // namespace cldnn
