/*******************************************************************************
* Copyright 2018-2021 Intel Corporation
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

#include <atomic>
#include <iostream>
#include <sstream>
#include <type_traits>

#include <stdlib.h>
#ifndef _WIN32
#include <sys/time.h>
#else
#include <windows.h>
#endif

#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl_debug.h"
#include "oneapi/dnnl/dnnl_version.h"

#include "c_types_map.hpp"
#include "verbose.hpp"

#include "batch_normalization_pd.hpp"
#include "binary_pd.hpp"
#include "concat_pd.hpp"
#include "convolution_pd.hpp"
#include "deconvolution_pd.hpp"
#include "eltwise_pd.hpp"
#include "inner_product_pd.hpp"
#include "layer_normalization_pd.hpp"
#include "lrn_pd.hpp"
#include "matmul_pd.hpp"
#include "pooling_pd.hpp"
#include "prelu_pd.hpp"
#include "reduction_pd.hpp"
#include "reorder_pd.hpp"
#include "resampling_pd.hpp"
#include "rnn_pd.hpp"
#include "shuffle_pd.hpp"
#include "softmax_pd.hpp"
#include "sum_pd.hpp"

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
#include "cpu/platform.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "gpu/ocl/verbose.hpp"
#endif

#ifdef DNNL_WITH_SYCL
#include "sycl/verbose.hpp"
#endif

namespace dnnl {
namespace impl {

static setting_t<int> verbose {0};
int get_verbose() {
#if !defined(DISABLE_VERBOSE)
    if (!verbose.initialized()) {
        // Assumes that all threads see the same environment
        const int len = 2;
        char val[len] = {0};
        if (getenv("MKLDNN_VERBOSE", val, len) == 1) verbose.set(atoi(val));
        if (getenv("DNNL_VERBOSE", val, len) == 1) verbose.set(atoi(val));
        if (!verbose.initialized()) verbose.set(0);
    }
    static std::atomic_flag version_printed = ATOMIC_FLAG_INIT;
    if (verbose.get() > 0 && !version_printed.test_and_set()) {
        printf("dnnl_verbose,info,oneDNN v%d.%d.%d (commit %s)\n",
                dnnl_version()->major, dnnl_version()->minor,
                dnnl_version()->patch, dnnl_version()->hash);
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
        printf("dnnl_verbose,info,cpu,runtime:%s\n",
                dnnl_runtime2str(dnnl_version()->cpu_runtime));
        printf("dnnl_verbose,info,cpu,isa:%s\n", cpu::platform::get_isa_info());
#endif
        printf("dnnl_verbose,info,gpu,runtime:%s\n",
                dnnl_runtime2str(dnnl_version()->gpu_runtime));
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
        gpu::ocl::print_verbose_header();
#endif
#ifdef DNNL_WITH_SYCL
        sycl::print_verbose_header();
#endif
        printf("dnnl_verbose,info,prim_template:");
        printf("%soperation,engine,primitive,implementation,prop_"
               "kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_"
               "time\n",
                get_verbose_timestamp() ? "timestamp," : "");
    }
#endif
    return verbose.get();
}

static setting_t<bool> verbose_timestamp {false};
bool get_verbose_timestamp() {
#if !defined(DISABLE_VERBOSE)
    if (!verbose_timestamp.initialized()) {
        // Assumes that all threads see the same environment
        const int len = 2;
        char val[len] = {0};
        if (getenv("DNNL_VERBOSE_TIMESTAMP", val, len) == 1)
            verbose_timestamp.set(atoi(val));
        if (!verbose_timestamp.initialized()) verbose_timestamp.set(false);
    }
#endif
    // No effect if verbose is not set.
    return verbose.get() && verbose_timestamp.get();
}

double get_msec() {
#ifdef _WIN32
    static LARGE_INTEGER frequency;
    if (frequency.QuadPart == 0) QueryPerformanceFrequency(&frequency);
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return 1e+3 * now.QuadPart / frequency.QuadPart;
#else
    struct timeval time;
    gettimeofday(&time, nullptr);
    return 1e+3 * time.tv_sec + 1e-3 * time.tv_usec;
#endif
}

#if defined(DISABLE_VERBOSE)
void pd_info_t::init(
        dnnl::impl::engine_t *, const dnnl::impl::primitive_desc_t *) {}

#else

std::ostream &operator<<(std::ostream &ss, engine_kind_t eng_kind) {
    ss << dnnl_engine_kind2str(eng_kind);
    return ss;
}

std::ostream &operator<<(std::ostream &ss, const engine_t *engine) {
    ss << dnnl_engine_kind2str(engine->kind());
    if (dnnl_engine_get_count(engine->kind()) > 1)
        ss << ":" + std::to_string(engine->index());
    return ss;
}

const char *prim_kind2str(primitive_kind_t prim_kind) {
    switch ((int)prim_kind) {
        case primitive_kind::zero_pad: return "zero_pad";
        default: return dnnl_prim_kind2str(prim_kind);
    }
}

std::ostream &operator<<(std::ostream &ss, primitive_kind_t prim_kind) {
    ss << prim_kind2str(prim_kind);
    return ss;
}

std::ostream &operator<<(std::ostream &ss, prop_kind_t prop_kind) {
    ss << dnnl_prop_kind2str(prop_kind);
    return ss;
}

std::ostream &operator<<(std::ostream &ss, data_type_t data_type) {
    ss << dnnl_dt2str(data_type);
    return ss;
}

std::ostream &operator<<(std::ostream &ss, alg_kind_t alg) {
    ss << dnnl_alg_kind2str(alg);
    return ss;
}

std::ostream &operator<<(std::ostream &ss, format_kind_t format_kind) {
    ss << dnnl_fmt_kind2str(format_kind);
    return ss;
}

std::string flags2str(unsigned flags) {
    std::string s;
    if (flags & dnnl_use_global_stats) s += "G";
    if (flags & dnnl_use_scaleshift) s += "S";
    if (flags & dnnl_use_scale) s += "C";
    if (flags & dnnl_use_shift) s += "H";
    if (flags & dnnl_fuse_norm_relu) s += "R";
    return s;
}

std::ostream &operator<<(std::ostream &ss, const memory_extra_desc_t &extra) {
    using namespace memory_extra_flags;

    ss << ":f" << extra.flags;
    if (extra.flags & compensation_conv_s8s8)
        ss << ":s8m" << extra.compensation_mask;
    if (extra.flags & compensation_conv_asymmetric_src)
        ss << ":zpm" << extra.asymm_compensation_mask;
    if (extra.flags & scale_adjust && extra.scale_adjust != 1.f)
        ss << ":sa" << extra.scale_adjust;
    return ss;
}

std::string md2fmt_tag_str(const dnnl_memory_desc_t *md) {
    memory_desc_wrapper mdw(md);

    const auto &blk = mdw.blocking_desc();

    dims_t blocks = {0};
    mdw.compute_blocks(blocks);

    char dim_chars[DNNL_MAX_NDIMS + 1];

    dims_t ou_blocks = {0};
    utils::array_copy(ou_blocks, mdw.padded_dims(), mdw.ndims());

    bool plain = true;
    for (int d = 0; d < mdw.ndims(); ++d) {
        dim_chars[d] = (blocks[d] == 1 ? 'a' : 'A') + (char)d;
        if (blocks[d] != 1) plain = false;
        ou_blocks[d] /= blocks[d];
    }

    // Can't report meaningful tag for runtime dimensions.
    if (mdw.has_runtime_strides()) return "*";

    dims_t strides;
    utils::array_copy(strides, blk.strides, mdw.ndims());

    utils::simultaneous_sort(strides, ou_blocks, dim_chars, mdw.ndims(),
            [](dim_t a, dim_t b) { return b - a; });

    dim_chars[mdw.ndims()] = '\0';

    std::string s(dim_chars);

    if (!plain) {
        for (int iblk = 0; iblk < blk.inner_nblks; ++iblk) {
            char c = ('a' + (char)blk.inner_idxs[iblk]);
            s += (std::to_string(blk.inner_blks[iblk]) + c);
        }
    }
    return s;
}

// Forms a format string for a given memory descriptor.
//
// The format is defined as: 'dt:[p|o|0]:fmt_kind:fmt:extra'.
// Here:
//  - dt       -- data type
//  - p        -- indicates there is non-trivial padding
//  - o        -- indicates there is non-trivial padding offset
//  - 0        -- indicates there is non-trivial offset0
//  - fmt_kind -- format kind (blocked, wino, etc...)
//  - fmt      -- extended format string (format_kind specific)
//  - extra    -- shows extra fields (underspecified)
std::string md2fmt_str(const dnnl_memory_desc_t *md) {
    std::stringstream ss;
    if (!md) {
        ss << data_type::undef << "::" << format_kind::undef << "::";
        return ss.str();
    }

    memory_desc_wrapper mdw(md);
    ss << mdw.data_type() << ":";

    bool padded_dims = false, padded_offsets = false;
    for (int d = 0; d < mdw.ndims(); ++d) {
        if (mdw.dims()[d] != mdw.padded_dims()[d]) padded_dims = true;
        if (mdw.padded_offsets()[d] != 0) padded_offsets = true;
    }
    bool offset0 = mdw.offset0();
    ss << (padded_dims ? "p" : "") << (padded_offsets ? "o" : "");
    ss << (offset0 ? "0" : "") << ":" << mdw.format_kind() << ":";

    if (mdw.is_blocking_desc()) ss << md2fmt_tag_str(md);

    ss << mdw.extra();

    return ss.str();
}

// Puts memory_desc information into stream without dimensions
std::ostream &operator<<(std::ostream &ss, const memory_desc_t *md) {
    ss << md2fmt_str(md);
    return ss;
}

template <typename T>
static std::string get_val_str(T val) {
    static_assert(
            std::is_arithmetic<T>::value, "T must be an arithmetic type.");
    if (is_runtime_value(val)) return std::string("*");
    return std::to_string(val);
}

// Returns string with dimensions from a given memory descriptor.
// The format is defined as: dim0xdim1x...xdimN, with RT values signed as `*`.
std::string md2dim_str(const dnnl_memory_desc_t *md) {
    if (md == nullptr || md->ndims == 0) return "";

    memory_desc_wrapper mdw(md);
    std::string s;

    s += get_val_str(mdw.dims()[0]);
    for (int d = 1; d < mdw.ndims(); ++d)
        s += ("x" + get_val_str(mdw.dims()[d]));

    return s;
}

// Returns string with descriptor style from memory_desc since there's an
// operator<< for memory_desc.
std::string md2desc_str(const memory_desc_t *md) {
    const auto dims = md->dims;
    std::string s;
    if (md->ndims >= 6) return md2dim_str(md);

    if (md->ndims == 1) {
        s += "x" + std::to_string(dims[0]);
        return s;
    }

    s += "mb" + std::to_string(dims[0]) + "ic" + std::to_string(dims[1]);
    if (md->ndims >= 5) s += "id" + std::to_string(dims[md->ndims - 3]);
    if (md->ndims >= 4) s += "ih" + std::to_string(dims[md->ndims - 2]);
    if (md->ndims >= 3) s += "iw" + std::to_string(dims[md->ndims - 1]);
    return s;
}

std::ostream &operator<<(std::ostream &ss, const scales_t &oscale) {
    ss << oscale.mask_;
    if (oscale.mask_ == 0) ss << ":" << get_val_str(oscale.scales_[0]);
    return ss;
}

std::ostream &operator<<(std::ostream &ss, const primitive_attr_t *attr) {
    // scratchpad and fpmath mode are not a part of
    // has_default_values(). Check them first.
    const scratchpad_mode_t &spm = attr->scratchpad_mode_;
    if (spm != scratchpad_mode_t::dnnl_scratchpad_mode_library) {
        ss << "attr-scratchpad:" << dnnl_scratchpad_mode2str(spm) << " ";
    }
    const fpmath_mode_t &fpm = attr->fpmath_mode_;
    if (fpm != fpmath_mode_t::dnnl_fpmath_mode_strict) {
        ss << "attr-fpmath_mode:" << dnnl_fpmath_mode2str(fpm) << " ";
    }

    if (attr->has_default_values()) return ss;

    const scales_t &os = attr->output_scales_;
    if (!os.has_default_values()) { ss << "attr-oscale:" << os << " "; }

    std::string empty_delim, attr_delim = "+";

    const arg_scales_t &as = attr->scales_;
    if (!as.has_default_values()) {
        std::string delim = empty_delim;
        ss << "attr-scales:";
        for (const auto &map_entry : as.scales_) {
            const auto &val = map_entry.second;
            if (val.has_default_values()) continue;

            int idx = as.get_index_val(map_entry.first);
            ss << delim << "src" << idx << ":" << val;
            delim = attr_delim;
        }
        ss << " ";
    }

    const zero_points_t &zp = attr->zero_points_;
    if (!zp.has_default_values()) {
        std::string delim = empty_delim;
        ss << "attr-zero-points:";
        for (const auto &arg : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
            if (zp.has_default_values(arg)) continue;

            int mask = 0;
            const int *zpp = nullptr;
            zp.get(arg, nullptr, &mask, &zpp);

            ss << delim
               << (arg == DNNL_ARG_SRC ? "src"
                                       : arg == DNNL_ARG_DST ? "dst" : "wei")
               << ":" << mask;
            if (mask == 0) ss << ":" << get_val_str(*zpp);
            delim = attr_delim;
        }
        ss << " ";
    }

    const post_ops_t &po = attr->post_ops_;
    if (!po.has_default_values()) {
        std::string delim = empty_delim;
        ss << "attr-post-ops:";
        for (int i = 0; i < po.len(); ++i) {
            const post_ops_t::entry_t &e = po.entry_[i];
            switch (e.kind) {
                case primitive_kind::sum: {
                    const auto &s = e.sum;
                    ss << delim << "sum";
                    if (s.scale != 1.f || s.zero_point != 0
                            || s.dt != data_type::undef)
                        ss << ":" << s.scale;
                    if (s.zero_point != 0 || s.dt != data_type::undef)
                        ss << ":" << s.zero_point;
                    if (s.dt != data_type::undef) ss << ":" << s.dt;
                } break;
//                case primitive_kind::convolution: {
//                    using namespace data_type;
//                    const auto &c = e.depthwise_conv;
//                    ss << delim << "dw_k3s" << c.stride << "p1";
//                    if (c.wei_dt == s8 || c.dst_dt != f32)
//                        ss << ":" << c.dst_dt;
//                    if (c.wei_dt == s8) {
//                        ss << ":" << c.mask;
//                        if (c.mask == 0) ss << ":" << c.scales[0];
//                    }
//                } break;
                case primitive_kind::convolution: {
                    const char *alg_str = "depthwise_conv_old";
                    ss << delim << alg_str;
                } break;
                case primitive_kind::eltwise: {
                    const post_ops_t::entry_t::eltwise_t &ew = e.eltwise;
                    ss << delim << ew.alg;
                    if (ew.alpha != 0.f || ew.beta != 0.f || ew.scale != 1.f)
                        ss << ":" << ew.alpha;
                    if (ew.beta != 0.f || ew.scale != 1.f) ss << ":" << ew.beta;
                    if (ew.scale != 1.f) ss << ":" << ew.scale;
                } break;
                case primitive_kind::binary: {
                    const post_ops_t::entry_t::binary_t &eb = e.binary;
                    const auto &md = eb.src1_desc;
                    int mask = 0;
                    for (int d = 0; d < md.ndims; ++d)
                        mask += md.dims[d] != 1 ? (1 << d) : 0;
                    ss << delim << eb.alg << ":" << md.data_type << ":" << mask;
                    if (!memory_desc_wrapper(md).count_non_unit_dims(1))
                        ss << ":" << md2fmt_tag_str(&md);
                } break;
                case primitive_kind::prelu: {
                    const auto &ep = e.prelu;
                    ss << delim << "prelu"
                       << ":" << ep.mask;
                } break;
                case primitive_kind::depthwise: {
                    const post_ops_t::entry_t::depthwise_t &dw = e.depthwise;
                    ss << delim << dw.alg;
                } break;
                case primitive_kind::quantization: {
                    const post_ops_t::entry_t::quantization_t &qt = e.quantization;
                    ss << delim << qt.alg;
                } break;
                default: assert(!"unsupported post op primitive kind!"); break;
            }
            delim = attr_delim;
        }
        ss << " ";
    }

    const rnn_data_qparams_t &rnn_qp = attr->rnn_data_qparams_;
    if (!rnn_qp.has_default_values()) {
        ss << "rnn_data_qparams:" << rnn_qp.scale_ << ":" << rnn_qp.shift_
           << ";";
    }

    return ss;
}

/* init_info section */
namespace {

template <typename pd_t>
static std::string init_info_batch_normalization(
        const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->src_md();
    auto diff_src_md = pd->diff_src_md();
    ss << "data_" << src_md;
    if (diff_src_md) ss << " diff_" << diff_src_md;
    ss << ",";

    ss << pd->attr() << ",";
    ss << "flags:" << flags2str(pd->desc()->flags) << ",";
    ss << md2desc_str(src_md);

    return ss.str();
}

template <typename pd_t>
static std::string init_info_binary(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << "," << prop_kind::undef
       << ",";

    auto src0_md = pd->src_md(0);
    auto src1_md = pd->src_md(1);
    auto dst_md = pd->dst_md();
    ss << "src_" << src0_md << " src_" << src1_md << " dst_" << dst_md << ",";

    ss << pd->attr() << ",";
    ss << "alg:" << pd->desc()->alg_kind << ",";
    ss << md2dim_str(src0_md) << ":" << md2dim_str(src1_md);

    return ss.str();
}

template <typename pd_t>
static std::string init_info_concat(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << "," << prop_kind::undef
       << ",";

    for (int i = 0; i < pd->n_inputs(); ++i) {
        auto src_i_md = pd->src_md(i);
        ss << "src_" << src_i_md << " ";
    }
    auto dst_md = pd->dst_md();
    ss << "dst_" << dst_md << ",";

    ss << pd->attr() << ",";
    ss << "axis:" << pd->desc()->concat_dimension << ",";

    for (int i = 0; i < pd->n_inputs(); ++i) {
        auto src_i_md = pd->src_md(i);
        ss << md2dim_str(src_i_md);
        if (i < pd->n_inputs() - 1) ss << ":";
    }

    return ss.str();
}

template <typename pd_t>
static std::string init_info_convolution(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->desc()->prop_kind == prop_kind::backward_data
            ? pd->diff_src_md()
            : pd->src_md();
    auto wei_md = pd->desc()->prop_kind == prop_kind::backward_weights
            ? pd->diff_weights_md(0)
            : pd->weights_md(0);
    auto bia_md = pd->desc()->prop_kind == prop_kind::backward_weights
            ? pd->diff_weights_md(1)
            : pd->weights_md(1);
    auto dst_md = !pd->is_fwd() ? pd->diff_dst_md() : pd->dst_md();

    ss << "src_" << src_md << " wei_" << wei_md;
    if (bia_md) ss << " bia_" << bia_md;
    ss << " dst_" << dst_md << ",";

    ss << pd->attr() << ",";
    ss << "alg:" << pd->desc()->alg_kind << ",";

    if (pd->with_groups()) ss << "g" << pd->G();
    ss << "mb" << pd->MB() << "_"
       << "ic" << pd->IC() << "oc" << pd->OC() << "_";
    if (pd->ndims() >= 5)
        ss << "id" << pd->ID() << "od" << pd->OD() << "kd" << pd->KD() << "sd"
           << pd->KSD() << "dd" << pd->KDD() << "pd" << pd->padFront() << "_";
    if (pd->ndims() >= 4)
        ss << "ih" << pd->IH() << "oh" << pd->OH() << "kh" << pd->KH() << "sh"
           << pd->KSH() << "dh" << pd->KDH() << "ph" << pd->padT() << "_";
    ss << "iw" << pd->IW() << "ow" << pd->OW() << "kw" << pd->KW() << "sw"
       << pd->KSW() << "dw" << pd->KDW() << "pw" << pd->padL();

    return ss.str();
}

template <typename pd_t>
static std::string init_info_deconvolution(const engine_t *e, const pd_t *pd) {
    return init_info_convolution(e, pd);
}

template <typename pd_t>
static std::string init_info_eltwise(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto data_md = pd->use_dst() ? pd->dst_md() : pd->src_md();
    auto diff_src_md = pd->diff_src_md();
    ss << "data_" << data_md;
    if (diff_src_md) ss << " diff_" << diff_src_md;
    ss << ",";

    ss << pd->attr() << ",";
    ss << "alg:" << pd->desc()->alg_kind << " alpha:" << pd->desc()->alpha
       << " beta:" << pd->desc()->beta << ",";
    ss << md2dim_str(data_md);

    return ss.str();
}

template <typename pd_t>
static std::string init_info_inner_product(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->desc()->prop_kind == prop_kind::backward_data
            ? pd->diff_src_md()
            : pd->src_md();
    auto wei_md = pd->desc()->prop_kind == prop_kind::backward_weights
            ? pd->diff_weights_md(0)
            : pd->weights_md(0);
    auto bia_md = pd->desc()->prop_kind == prop_kind::backward_weights
            ? pd->diff_weights_md(1)
            : pd->weights_md(1);
    auto dst_md = !pd->is_fwd() ? pd->diff_dst_md() : pd->dst_md();

    ss << "src_" << src_md << " wei_" << wei_md;
    if (bia_md) ss << " bia_" << bia_md;
    ss << " dst_" << dst_md << ",";

    ss << pd->attr() << ",,";

    ss << md2desc_str(src_md);
    ss << "oc" << pd->OC();

    return ss.str();
}

template <typename pd_t>
static std::string init_info_layer_normalization(
        const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->src_md(0);
    auto stats_md = pd->is_fwd() && !pd->stats_are_src() ? pd->dst_md(1)
                                                         : pd->src_md(1);
    auto diff_src_md = pd->diff_src_md();
    ss << "data_" << src_md;
    if (stats_md) ss << " stats_" << stats_md;
    if (diff_src_md) ss << " diff_" << diff_src_md;
    ss << ",";

    ss << pd->attr() << ",";
    ss << "flags:" << flags2str(pd->desc()->flags) << ",";
    ss << md2dim_str(src_md);

    return ss.str();
}

template <typename pd_t>
static std::string init_info_lrn(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto data_md = pd->src_md();
    auto diff_src_md = pd->diff_src_md();
    ss << "data_" << data_md;
    if (diff_src_md) ss << " diff_" << diff_src_md;
    ss << ",";

    ss << pd->attr() << ",";
    ss << "alg:" << pd->desc()->alg_kind << ",";
    ss << md2desc_str(data_md);
    ss << "ls" << pd->desc()->local_size << "beta" << pd->desc()->lrn_beta;

    return ss.str();
}

template <typename pd_t>
static std::string init_info_matmul(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << "," << prop_kind::undef
       << ",";

    auto src_md = pd->src_md();
    auto wei_md = pd->weights_md(0);
    auto bia_md = pd->weights_md(1);
    auto dst_md = pd->dst_md();

    auto get_bia_mask = [&bia_md]() {
        auto bia_ndims = bia_md->ndims;
        auto bia_dims = bia_md->dims;
        int mask = 0;
        for (int d = bia_ndims - 1; d >= 0; --d) {
            mask += bia_dims[d] != 1 ? 1 << d : 0;
        }
        return mask;
    };

    ss << "src_" << src_md << " wei_" << wei_md;
    if (pd->with_bias()) ss << " bia_" << bia_md << "_mask" << get_bia_mask();
    ss << " dst_" << dst_md << ",";

    ss << pd->attr() << ",,";

    ss << md2dim_str(src_md) << ":" << md2dim_str(wei_md) << ":"
       << md2dim_str(dst_md);

    return ss.str();
}

template <typename pd_t>
static std::string init_info_pooling(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->is_fwd() ? pd->src_md() : pd->diff_src_md();
    auto dst_md = pd->is_fwd() ? pd->dst_md() : pd->diff_dst_md();
    auto ws_md = pd->workspace_md();

    ss << "src_" << src_md << " dst_" << dst_md;
    if (ws_md) ss << " ws_" << ws_md;
    ss << ",";

    ss << pd->attr() << ",";
    ss << "alg:" << pd->desc()->alg_kind << ",";

    ss << "mb" << pd->MB() << "ic" << pd->IC() << "_";
    if (pd->ndims() >= 5)
        ss << "id" << pd->ID() << "od" << pd->OD() << "kd" << pd->KD() << "sd"
           << pd->KSD() << "dd" << pd->KDD() << "pd" << pd->padFront() << "_";
    if (pd->ndims() >= 4)
        ss << "ih" << pd->IH() << "oh" << pd->OH() << "kh" << pd->KH() << "sh"
           << pd->KSH() << "dh" << pd->KDH() << "ph" << pd->padT() << "_";
    ss << "iw" << pd->IW() << "ow" << pd->OW() << "kw" << pd->KW() << "sw"
       << pd->KSW() << "dw" << pd->KDW() << "pw" << pd->padL();

    return ss.str();
}

template <typename pd_t>
static std::string init_info_prelu(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto data_md = pd->src_md(0);
    auto wei_md = pd->weights_md(0);
    auto diff_data_md = pd->diff_src_md(0);
    auto diff_wei_md = pd->diff_weights_md(0);

    ss << "data_" << data_md << " wei_" << wei_md;
    if (diff_data_md) ss << " diff_" << diff_data_md;
    if (diff_wei_md) ss << " diff_wei_" << diff_wei_md;
    ss << ",";

    ss << pd->attr() << ",,";
    ss << md2dim_str(data_md) << ":" << md2dim_str(wei_md);

    return ss.str();
}

template <typename pd_t>
static std::string init_info_reduction(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << "," << prop_kind::undef
       << ",";

    auto src_md = pd->src_md();
    auto dst_md = pd->dst_md();
    ss << "src_" << src_md << " dst_" << dst_md << ",";

    ss << pd->attr() << ",";
    ss << "alg:" << pd->desc()->alg_kind << " p:" << pd->desc()->p
       << " eps:" << pd->desc()->eps << ",";
    ss << md2dim_str(src_md) << ":" << md2dim_str(dst_md);

    return ss.str();
}

template <typename pd_t>
static std::string init_info_reorder(const engine_t *e, pd_t *pd) {
    std::stringstream ss;

    const auto src_ek = pd->desc()->src_engine_kind;
    const auto dst_ek = pd->desc()->dst_engine_kind;

    if (src_ek != dst_ek)
        ss << src_ek << "2" << dst_ek;
    else
        ss << e;

    ss << "," << pd->kind() << "," << pd->name() << "," << prop_kind::undef
       << ",";

    auto src_md = pd->src_md();
    auto dst_md = pd->dst_md();
    ss << "src_" << src_md << " dst_" << dst_md << ",";

    ss << pd->attr() << ",,";
    ss << md2dim_str(dst_md);

    return ss.str();
}

template <typename pd_t>
static std::string init_info_resampling(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_md = pd->is_fwd() ? pd->src_md() : pd->diff_src_md();
    auto dst_md = pd->is_fwd() ? pd->dst_md() : pd->diff_dst_md();

    ss << "src_" << src_md << " dst_" << dst_md << ",";

    ss << pd->attr() << ",";
    ss << "alg:" << pd->desc()->alg_kind << ",";

    ss << "mb" << pd->MB() << "ic" << pd->C() << "_";
    if (pd->ndims() >= 5) ss << "id" << pd->ID() << "od" << pd->OD() << "_";
    if (pd->ndims() >= 4) ss << "ih" << pd->IH() << "oh" << pd->OH() << "_";
    ss << "iw" << pd->IW() << "ow" << pd->OW();

    return ss.str();
}

template <typename pd_t>
static std::string init_info_rnn(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto src_layer_md = pd->is_fwd() ? pd->src_md(0) : pd->diff_src_md(0);
    ss << "src_layer_" << src_layer_md;
    if (pd->with_src_iter()) {
        auto src_iter_md = pd->is_fwd() ? pd->src_md(1) : pd->diff_src_md(1);
        ss << " src_iter_" << src_iter_md;
    }
    auto wei_layer_md
            = pd->is_fwd() ? pd->weights_md(0) : pd->diff_weights_md(0);
    ss << " wei_layer_" << wei_layer_md;
    auto wei_iter_md
            = pd->is_fwd() ? pd->weights_md(1) : pd->diff_weights_md(1);
    ss << " wei_iter_" << wei_iter_md;
    if (pd->is_lstm_peephole()) {
        auto wei_peephole_md
                = pd->arg_md(pd->is_fwd() ? DNNL_ARG_WEIGHTS_PEEPHOLE
                                          : DNNL_ARG_DIFF_WEIGHTS_PEEPHOLE);
        ss << " wei_peephole_" << wei_peephole_md;
    }
    if (pd->is_lstm_projection()) {
        auto wei_projection_md
                = pd->arg_md(pd->is_fwd() ? DNNL_ARG_WEIGHTS_PROJECTION
                                          : DNNL_ARG_DIFF_WEIGHTS_PROJECTION);
        ss << " wei_proj_" << wei_projection_md;
    }
    if (pd->with_bias()) {
        auto bia_md
                = pd->arg_md(pd->is_fwd() ? DNNL_ARG_BIAS : DNNL_ARG_DIFF_BIAS);
        ss << " bias_" << bia_md;
    }
    auto dst_layer_md = pd->is_fwd() ? pd->dst_md(0) : pd->diff_dst_md(0);
    ss << " dst_layer_" << dst_layer_md;
    if (pd->with_dst_iter()) {
        auto dst_iter_md = pd->is_fwd() ? pd->dst_md(1) : pd->diff_dst_md(1);
        ss << " dst_iter_" << dst_iter_md;
    }
    ss << ",";

    ss << pd->attr() << ",";
    ss << "alg:" << pd->cell_kind()
       << " direction:" << dnnl_rnn_direction2str(pd->direction())
       << " activation:" << pd->activation_kind() << ",";

    ss << "l" << pd->L() << "t" << pd->T() << "mb" << pd->MB() << "sic"
       << pd->SIC() << "slc" << pd->SLC() << "dhc" << pd->DHC() << "dic"
       << pd->DIC();

    return ss.str();
}

template <typename pd_t>
static std::string init_info_shuffle(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto data_md = pd->is_fwd() ? pd->src_md() : pd->diff_src_md();
    ss << "data_" << data_md << ",";

    ss << pd->attr() << ",";
    ss << "axis:" << pd->axis() << " group:" << pd->group_size() << ",";
    ss << md2dim_str(data_md);

    return ss.str();
}

template <typename pd_t>
static std::string init_info_softmax(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << ","
       << pd->desc()->prop_kind << ",";

    auto data_md = pd->dst_md();
    auto diff_data_md = pd->diff_src_md();
    ss << "data_" << data_md << " diff_" << diff_data_md << ",";

    ss << pd->attr() << ",";
    ss << "alg:" << (pd->is_softmax() ? "softmax" : "logsoftmax")
       << " axis:" << pd->axis() << ",";
    ss << md2dim_str(data_md);

    return ss.str();
}

template <typename pd_t>
static std::string init_info_logsoftmax(const engine_t *e, const pd_t *pd) {
    return init_info_softmax(e, pd);
}

template <typename pd_t>
static std::string init_info_sum(const engine_t *e, const pd_t *pd) {
    std::stringstream ss;
    ss << e << "," << pd->kind() << "," << pd->name() << "," << prop_kind::undef
       << ",";

    for (int i = 0; i < pd->n_inputs(); ++i) {
        auto src_i_md = pd->src_md(i);
        ss << "src_" << src_i_md << " ";
    }
    auto dst_md = pd->dst_md();
    ss << "dst_" << dst_md << ",";

    ss << pd->attr() << ",,";
    ss << md2dim_str(dst_md);

    return ss.str();
}

} // namespace

void pd_info_t::init(engine_t *engine, const primitive_desc_t *pd) {
    if (is_initialized_) return;

    std::call_once(initialization_flag_, [&] {
        using logsoftmax_pd_t = softmax_pd_t;
// clang-format off
#define CASE(kind) \
    case primitive_kind::kind: \
        str_ = init_info_##kind(engine, (const kind##_pd_t *)pd); \
        break

        switch ((int)pd->kind()) {
            CASE(batch_normalization);
            CASE(binary);
            CASE(concat);
            CASE(convolution);
            CASE(deconvolution);
            CASE(eltwise);
            CASE(inner_product);
            CASE(layer_normalization);
            CASE(lrn);
            CASE(logsoftmax);
            CASE(matmul);
            case primitive_kind::pooling_v2:
            CASE(pooling);
            CASE(prelu);
            CASE(reduction);
            CASE(reorder);
            CASE(resampling);
            CASE(rnn);
            CASE(shuffle);
            CASE(softmax);
            CASE(sum);
            case primitive_kind::zero_pad: break;
            default: assert(!"unknown primitive kind");
        }
#undef CASE
        // clang-format on

        is_initialized_ = true;
    });
}
#endif

} // namespace impl
} // namespace dnnl

dnnl_status_t dnnl_set_verbose(int level) {
    using namespace dnnl::impl::status;
    if (level < 0 || level > 2) return invalid_arguments;
    dnnl::impl::verbose.set(level);
    return success;
}

const dnnl_version_t *dnnl_version(void) {
    static const dnnl_version_t ver
            = {DNNL_VERSION_MAJOR, DNNL_VERSION_MINOR, DNNL_VERSION_PATCH,
                    DNNL_VERSION_HASH, DNNL_CPU_RUNTIME, DNNL_GPU_RUNTIME};
    return &ver;
}
