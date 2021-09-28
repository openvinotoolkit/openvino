// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"
#include <oneapi/dnnl/dnnl_debug.h>
#include <oneapi/dnnl/dnnl_ocl.hpp>

#include "to_string_utils.h"

namespace cldnn {
namespace onednn {

template <typename T>
cldnn::memory::ptr convert_zp_data_to_s32(const memory::ptr zp_memory) {
    auto engine = zp_memory->get_engine();
    auto& stream = engine->get_program_stream();

    auto zp_s32_layout = zp_memory->get_layout();
    zp_s32_layout.data_type = data_types::i32;
    auto zp_s32_memory = engine->allocate_memory(zp_s32_layout, false);

    mem_lock<T> zp_data(zp_memory, stream);
    mem_lock<int32_t> zp_s32_data(zp_s32_memory, stream);
    for (size_t i = 0; i < zp_data.size(); i++) {
        zp_s32_data.data()[i] = static_cast<int32_t>(zp_data.data()[i]);
    }

    return zp_s32_memory;
}

template cldnn::memory::ptr convert_zp_data_to_s32<int8_t>(const memory::ptr zp_memory);
template cldnn::memory::ptr convert_zp_data_to_s32<uint8_t>(const memory::ptr zp_memory);

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

void pad_dims(dnnl::memory::dims& padded_dims, cldnn::format format) {
    auto block_sizes = format.block_sizes();
    for (auto& block : block_sizes) {
        auto rounded_dim = round_up_to(padded_dims[block.first], block.second);
        padded_dims[block.first] = rounded_dim;
    }
}

dnnl::memory::data_type convert_data_type(cldnn::data_types dt) {
    switch (dt) {
        case cldnn::data_types::f32: return dnnl::memory::data_type::f32;
        case cldnn::data_types::f16: return dnnl::memory::data_type::f16;
        case cldnn::data_types::i8: return dnnl::memory::data_type::s8;
        case cldnn::data_types::u8: return dnnl::memory::data_type::u8;
        case cldnn::data_types::i32: return dnnl::memory::data_type::s32;
        default: throw std::invalid_argument("[clDNN] Unsupported conversion from cldnn to ondnn type");
    }
}

dnnl::memory::format_tag convert_data_format(cldnn::format fmt) {
    switch (fmt) {
        case cldnn::format::bfyx: return dnnl::memory::format_tag::nchw;
        case cldnn::format::bfzyx: return dnnl::memory::format_tag::ncdhw;
        case cldnn::format::byxf: return dnnl::memory::format_tag::nhwc;
        case cldnn::format::b_fs_yx_fsv16: return dnnl::memory::format_tag::nChw16c;
        case cldnn::format::b_fs_yx_fsv32: return dnnl::memory::format_tag::aBcd32b;
        case cldnn::format::b_fs_zyx_fsv16: return dnnl::memory::format_tag::nCdhw16c;
        case cldnn::format::b_fs_zyx_fsv32: return dnnl::memory::format_tag::aBcde32b;
        case cldnn::format::bs_fs_yx_bsv16_fsv16: return dnnl::memory::format_tag::NChw16n16c;
        case cldnn::format::bs_fs_yx_bsv32_fsv32: return dnnl::memory::format_tag::NChw32n32c;
        case cldnn::format::bs_fs_yx_bsv4_fsv4: return dnnl::memory::format_tag::ABcd4a4b;
        case cldnn::format::bs_fs_yx_bsv4_fsv2: return dnnl::memory::format_tag::ABcd4a2b;
        case cldnn::format::bs_fs_yx_bsv32_fsv16: return dnnl::memory::format_tag::NChw32n16c;
        case cldnn::format::bs_fs_zyx_bsv16_fsv16: return dnnl::memory::format_tag::NCdhw16n16c;
        default: throw std::invalid_argument("[clDNN] Unsupported conversion from cldnn to ondnn layout " + fmt_to_str(fmt));
    }
}

dnnl::memory::desc layout_to_memory_desc(cldnn::layout l, dnnl::memory::format_tag target_fmt, bool flatten) {
    size_t rank = cldnn::format::dimension(l.format);
    dnnl::memory::dims dims;
    dnnl::memory::dims padded_dims;
    dnnl::memory::dims padded_offset;
    if (target_fmt == dnnl::memory::format_tag::ab && flatten) {
        dims = flatten_tensor(l.size);
        dims.insert(dims.begin(), 1);
        padded_dims = dims;
    } else if (target_fmt == dnnl::memory::format_tag::ab) {
        dims.push_back(l.size.batch[0]);
        dims.push_back(l.size.count() / l.size.batch[0]);
        padded_dims = dims;
    } else if (flatten) {
        dims = flatten_tensor(l.size);
        padded_dims = dims;
    } else {
        auto padded_size = l.size + l.data_padding.lower_size() + l.data_padding.upper_size();
        auto offset = l.data_padding.lower_size();
        dims = convert_tensor(l.size, rank, cldnn::format::is_grouped(l.format));
        padded_dims = convert_tensor(padded_size, rank);
        padded_offset = convert_tensor(offset, rank);
    }

    pad_dims(padded_dims, l.format);

    dnnl::memory::data_type dt = convert_data_type(l.data_type);
    dnnl::memory::format_tag fmt = target_fmt == dnnl::memory::format_tag::undef ? convert_data_format(l.format) : target_fmt;

    dnnl::memory::desc res(dims, dt, fmt);

    std::copy(padded_dims.begin(), padded_dims.end(), res.data.padded_dims);
    std::copy(padded_offset.begin(), padded_offset.end(), res.data.padded_offsets);

    return res;
}

static const std::map<int, std::vector<dnnl::memory::format_tag>> form_tags_by_ndims {
    {0, {
        dnnl::memory::format_tag::a   // TODO :: really 1d layout for scalar??
     }}, {1, {
        dnnl::memory::format_tag::a
     }}, {2, {
        dnnl::memory::format_tag::ab,
        dnnl::memory::format_tag::ba
     }}, {3, {
        dnnl::memory::format_tag::abc,
        dnnl::memory::format_tag::acb,
        dnnl::memory::format_tag::bac,
        dnnl::memory::format_tag::bca,
        dnnl::memory::format_tag::cba,

        dnnl::memory::format_tag::Abc16a,
        dnnl::memory::format_tag::ABc16a16b,
        dnnl::memory::format_tag::ABc4a4b,
        dnnl::memory::format_tag::aBc16b,
        dnnl::memory::format_tag::aBc32b,
        dnnl::memory::format_tag::ABc16b16a,
        dnnl::memory::format_tag::Abc4a,
        dnnl::memory::format_tag::aBc4b,
        dnnl::memory::format_tag::ABc4b16a4b,
        dnnl::memory::format_tag::ABc2b8a4b,
        dnnl::memory::format_tag::ABc16b16a4b,
        dnnl::memory::format_tag::ABc16b16a2b,
        dnnl::memory::format_tag::ABc4b4a,
        dnnl::memory::format_tag::ABc8a16b2a,
        dnnl::memory::format_tag::ABc8a8b,
        dnnl::memory::format_tag::ABc8a4b,
        dnnl::memory::format_tag::aBc8b,
        dnnl::memory::format_tag::ABc8b16a2b,
        dnnl::memory::format_tag::ABc8b8a,
        // dnnl::memory::format_tag::ABc8a2b,
        dnnl::memory::format_tag::Acb16a,
        dnnl::memory::format_tag::Acb4a,
        dnnl::memory::format_tag::Acb8a,
        dnnl::memory::format_tag::BAc16a16b,
        dnnl::memory::format_tag::BAc16b16a,
     }}, {4, {                                 // Popular
        dnnl::memory::format_tag::abcd,      // plain
        dnnl::memory::format_tag::acdb,      // tail_c
        dnnl::memory::format_tag::aBcd8b,    // blocked 8c
        dnnl::memory::format_tag::aBcd16b,   // blocked 16c

        dnnl::memory::format_tag::abdc,

        dnnl::memory::format_tag::bacd,
        dnnl::memory::format_tag::bcda,
        dnnl::memory::format_tag::cdba,
        dnnl::memory::format_tag::dcab,

        dnnl::memory::format_tag::Abcd8a,
        dnnl::memory::format_tag::Abcd16a,
        dnnl::memory::format_tag::Abcd32a,
        dnnl::memory::format_tag::ABcd16a16b,
        dnnl::memory::format_tag::aBcd32b,
        dnnl::memory::format_tag::ABcd16b16a,
        dnnl::memory::format_tag::aBCd16b16c,
        dnnl::memory::format_tag::aBCd16c16b,
        dnnl::memory::format_tag::Abcd4a,
        dnnl::memory::format_tag::aBcd4b,
        dnnl::memory::format_tag::ABcd4b16a4b,
        dnnl::memory::format_tag::ABcd2b8a4b,
        dnnl::memory::format_tag::ABcd4b4a,
        // dnnl::memory::format_tag::ABcd8a2b,
        dnnl::memory::format_tag::ABcd4a4b,
        dnnl::memory::format_tag::aBCd4c16b4c,
        dnnl::memory::format_tag::aBCd2c8b4c,
        dnnl::memory::format_tag::ABcd16b16a4b,
        dnnl::memory::format_tag::ABcd16b16a2b,
        dnnl::memory::format_tag::aBCd16c16b4c,
        dnnl::memory::format_tag::aBCd16c16b2c,
        dnnl::memory::format_tag::aBCd4c4b,
        dnnl::memory::format_tag::aBCd4b4c,
        dnnl::memory::format_tag::ABcd8a16b2a,
        dnnl::memory::format_tag::ABcd8a8b,
        dnnl::memory::format_tag::ABcd32a32b,
        dnnl::memory::format_tag::ABcd8a4b,

        dnnl::memory::format_tag::ABcd8b16a2b,
        dnnl::memory::format_tag::aBCd8b16c2b,
        dnnl::memory::format_tag::ABcd8b8a,
        dnnl::memory::format_tag::aBCd8b8c,
        dnnl::memory::format_tag::aBCd8b4c,
        dnnl::memory::format_tag::aBCd8c16b2c,
        dnnl::memory::format_tag::aBCd8c8b,

        dnnl::memory::format_tag::ABcd4a8b8a4b,
        // dnnl::memory::format_tag::ABcd4a8b8a2b,
        dnnl::memory::format_tag::ABcd2a8b8a2b,

        dnnl::memory::format_tag::aBdc16b,
        dnnl::memory::format_tag::aBdc4b,
        dnnl::memory::format_tag::aBdc8b,
        dnnl::memory::format_tag::aCBd16b16c,
        dnnl::memory::format_tag::aCBd16c16b,
        dnnl::memory::format_tag::Acdb16a,
        dnnl::memory::format_tag::Acdb4a,
        dnnl::memory::format_tag::Acdb8a,
        dnnl::memory::format_tag::BAcd16a16b,
        dnnl::memory::format_tag::BAcd16b16a,
        dnnl::memory::format_tag::ABcd32a32b,
        dnnl::memory::format_tag::Acdb32a,
        dnnl::memory::format_tag::aBCd2b4c2b,
        dnnl::memory::format_tag::aBCd2c4b2c,
        dnnl::memory::format_tag::aBCd4b8c2b,
        dnnl::memory::format_tag::aBCd4c8b2c,
    }}, {5, {                                   // Popular
        dnnl::memory::format_tag::abcde,      // plain
        dnnl::memory::format_tag::acdeb,      // tail_c
        dnnl::memory::format_tag::aBcde8b,    // blocked 8c
        dnnl::memory::format_tag::aBcde16b,   // blocked 16c

        dnnl::memory::format_tag::abdec,
        dnnl::memory::format_tag::acbde,
        dnnl::memory::format_tag::bacde,
        dnnl::memory::format_tag::bcdea,
        dnnl::memory::format_tag::cdeba,
        dnnl::memory::format_tag::decab,

        dnnl::memory::format_tag::Abcde16a,
        dnnl::memory::format_tag::Abcde32a,
        dnnl::memory::format_tag::ABcde16a16b,
        dnnl::memory::format_tag::aBcde32b,
        dnnl::memory::format_tag::ABcde16b16a,
        dnnl::memory::format_tag::aBCde16b16c,
        dnnl::memory::format_tag::aBCde16c16b,
        dnnl::memory::format_tag::aBCde2c8b4c,
        dnnl::memory::format_tag::Abcde4a,
        dnnl::memory::format_tag::aBcde4b,
        dnnl::memory::format_tag::ABcde4b4a,
        dnnl::memory::format_tag::ABcde4a4b,
        dnnl::memory::format_tag::aBCde4b4c,
        dnnl::memory::format_tag::aBCde4c16b4c,
        dnnl::memory::format_tag::aBCde16c16b4c,
        dnnl::memory::format_tag::aBCde16c16b2c,
        dnnl::memory::format_tag::aBCde4c4b,
        dnnl::memory::format_tag::Abcde8a,
        dnnl::memory::format_tag::ABcde8a8b,
        dnnl::memory::format_tag::ABcde8a4b,
        dnnl::memory::format_tag::ABcde8b16a2b,
        dnnl::memory::format_tag::ABcde4b16a4b,
        dnnl::memory::format_tag::ABcde2b8a4b,
        dnnl::memory::format_tag::aBCde8b16c2b,
        dnnl::memory::format_tag::ABcde8b8a,
        dnnl::memory::format_tag::aBCde8b8c,
        dnnl::memory::format_tag::aBCde8b4c,
        dnnl::memory::format_tag::aBCde4b8c8b4c,
        // dnnl::memory::format_tag::aBCde4b8c8b2c,
        dnnl::memory::format_tag::aBCde2b8c8b2c,
        dnnl::memory::format_tag::aBCde8c16b2c,
        dnnl::memory::format_tag::aBCde8c8b,
        dnnl::memory::format_tag::aBdec16b,
        dnnl::memory::format_tag::aBdec4b,
        dnnl::memory::format_tag::aBdec8b,
        dnnl::memory::format_tag::aCBde16b16c,
        dnnl::memory::format_tag::aCBde16c16b,
        dnnl::memory::format_tag::Acdeb16a,
        dnnl::memory::format_tag::Acdeb4a,
        dnnl::memory::format_tag::Acdeb8a,
        dnnl::memory::format_tag::BAcde16b16a,
        dnnl::memory::format_tag::BAcde16a16b,
        dnnl::memory::format_tag::aBdec32b,
        dnnl::memory::format_tag::aBCde2b4c2b,
        dnnl::memory::format_tag::aBCde2c4b2c,
        dnnl::memory::format_tag::aBCde4b8c2b,
        dnnl::memory::format_tag::aBCde4c8b2c,
    }}, {6, {                                    // Popular
        dnnl::memory::format_tag::abcdef,      // plain
        dnnl::memory::format_tag::acbdef,      // permuted
        dnnl::memory::format_tag::defcab,      // permuted
        dnnl::memory::format_tag::aBcdef16b,   // blocked 16c

        dnnl::memory::format_tag::aBCdef16b16c,
        dnnl::memory::format_tag::aBCdef16c16b,
        dnnl::memory::format_tag::aBcdef4b,
        dnnl::memory::format_tag::aBCdef2c8b4c,
        dnnl::memory::format_tag::aBCdef4c4b,
        dnnl::memory::format_tag::aBCdef4b4c,
        dnnl::memory::format_tag::aBCdef8b8c,
        dnnl::memory::format_tag::aBCdef8b4c,
        dnnl::memory::format_tag::aBCdef8c16b2c,
        dnnl::memory::format_tag::aBCdef4c16b4c,
        dnnl::memory::format_tag::aBCdef8c8b,

        dnnl::memory::format_tag::aBdefc16b,
        dnnl::memory::format_tag::aCBdef16c16b,
        dnnl::memory::format_tag::aCBdef16b16c,
        dnnl::memory::format_tag::aBdefc4b,
        dnnl::memory::format_tag::aBdefc8b,

        dnnl::memory::format_tag::Abcdef16a,
        dnnl::memory::format_tag::Abcdef32a,
        dnnl::memory::format_tag::aBCdef2b4c2b,
        dnnl::memory::format_tag::aBCdef2c4b2c,
        dnnl::memory::format_tag::aBCdef4b8c2b,
        dnnl::memory::format_tag::aBCdef4c8b2c,
        }}
};


static bool isSame(dnnl::memory::desc desc, dnnl::memory::format_tag fmt) {
    dnnl::memory::desc refDesc(desc.dims(), desc.data_type(), fmt);

    if (desc.data.ndims != refDesc.data.ndims)
        return false;

    if (desc.data.format_kind != dnnl_blocked || refDesc.data.format_kind != dnnl_blocked)
        throw std::runtime_error("dnnlMemoryDesc::isSame is not implemented for non blocked memory format");

    auto actualBlkDesc = desc.data.format_desc.blocking;
    auto refBlkDesc = refDesc.data.format_desc.blocking;
    if (actualBlkDesc.inner_nblks != refBlkDesc.inner_nblks)
        return false;

    for (int i = 0; i < actualBlkDesc.inner_nblks; ++i)
        if (actualBlkDesc.inner_blks[i] != refBlkDesc.inner_blks[i])
            return false;

    for (int i = 0; i < actualBlkDesc.inner_nblks; ++i)
        if (actualBlkDesc.inner_idxs[i] != refBlkDesc.inner_idxs[i])
            return false;

    auto actualStrides = desc.data.format_desc.blocking.strides;
    auto refStrides = refDesc.data.format_desc.blocking.strides;

    std::vector<size_t> actualOrder(desc.data.ndims);
    std::iota(actualOrder.begin(), actualOrder.end(), 0);
    std::sort(actualOrder.begin(), actualOrder.end(),
              [&actualStrides] (size_t ind_l, size_t ind_r) {
                  return actualStrides[ind_l] > actualStrides[ind_r];
              });

    std::vector<size_t> refOrder(refDesc.data.ndims);
    std::iota(refOrder.begin(), refOrder.end(), 0);
    std::sort(refOrder.begin(), refOrder.end(),
              [&refStrides] (size_t ind_l, size_t ind_r) {
                  return refStrides[ind_l] > refStrides[ind_r];
              });

    if (actualOrder != refOrder) {
        return false;
    }

    return true;
}

dnnl::memory::format_tag get_format_by_desc(dnnl::memory::desc desc) {
    // TODO [OneDNN]: Previously it was a field of tdesc, but now the brute
    //                force search here. Please avoid of using this method.
    const auto ndims = desc.dims().size();

    // There are no suitable format_tag for this
    if (ndims == 0 || ndims > 6)
        return dnnl::memory::format_tag::undef;

    for (const auto fmt : form_tags_by_ndims.at(static_cast<int>(ndims))) {
        if (isSame(desc, fmt))
            return fmt;
    }

    return dnnl::memory::format_tag::undef;
}

dnnl::algorithm convert_activation_func(cldnn::activation_func func) {
    switch (func) {
        case cldnn::activation_func::relu: return dnnl::algorithm::eltwise_relu;
        case cldnn::activation_func::elu: return dnnl::algorithm::eltwise_elu;
        case cldnn::activation_func::logistic: return dnnl::algorithm::eltwise_logistic;
        case cldnn::activation_func::clamp: return dnnl::algorithm::eltwise_clip;
        case cldnn::activation_func::relu_negative_slope: return dnnl::algorithm::eltwise_relu;
        case cldnn::activation_func::hyperbolic_tan: return dnnl::algorithm::eltwise_tanh;
        case cldnn::activation_func::swish: return dnnl::algorithm::eltwise_swish;
        case cldnn::activation_func::abs: return dnnl::algorithm::eltwise_abs;
        default: throw std::runtime_error("Unsupported activation func for onednn primitive " + std::to_string(static_cast<int>(func)));
    }
}

cldnn::format convert_format(dnnl::memory::format_tag fmt, bool is_grouped) {
    if (is_grouped) {
        switch (fmt) {
        case dnnl::memory::format_tag::abcde: return cldnn::format::goiyx;
        case dnnl::memory::format_tag::Abcde16a: return cldnn::format::gs_oiyx_gsv16;
        case dnnl::memory::format_tag::Abcde32a: return cldnn::format::gs_oiyx_gsv32;
        case dnnl::memory::format_tag::aCBde16c16b: return cldnn::format::g_is_os_yx_isv16_osv16;
        default: throw std::runtime_error(std::string("Unsupported grouped onednn fmt ") + dnnl_fmt_tag2str((dnnl_format_tag_t)fmt));
        }
    } else {
        switch (fmt) {
        case dnnl::memory::format_tag::abcd: return cldnn::format::oiyx;
        case dnnl::memory::format_tag::BAcd16b16a: return cldnn::format::is_os_yx_isv16_osv16;
        case dnnl::memory::format_tag::ABcd16b16a: return cldnn::format::os_is_yx_isv16_osv16;
        case dnnl::memory::format_tag::abcde: return cldnn::format::oizyx;
        default: throw std::runtime_error(std::string("Unsupported onednn fmt ") + dnnl_fmt_tag2str((dnnl_format_tag_t)fmt));
        }
    }
}

}  // namespace onednn
}  // namespace cldnn
