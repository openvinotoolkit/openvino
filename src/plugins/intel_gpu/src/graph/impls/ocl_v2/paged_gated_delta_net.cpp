// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_gated_delta_net.hpp"

#include <string_view>

#include "intel_gpu/primitives/paged_gated_delta_net.hpp"
#include "paged_gated_delta_net_jit_utils.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

size_t get_v_block_size(size_t v_head_dims) {
    // v_block_size is determined by GRF size and performance data
    return 4;
}

size_t get_vec_size(const RuntimeParams& params) {
    const auto& q_layout = params.get_input_layout(paged_gated_delta_net::QUERY);
    const auto& q_shape = q_layout.get_partial_shape();
    const auto& v_shape = params.get_input_layout(paged_gated_delta_net::VALUE).get_partial_shape();

    const size_t k_head_dims = q_shape[2].get_length();
    const size_t v_head_dims = v_shape[2].get_length();
    const size_t subgroup_size = paged_gated_delta_net_jit::get_subgroup_size(params.get_device_info().arch);

    if ((k_head_dims % 16) != 0 || (v_head_dims % 16) != 0) {
        return 1;
    }

    size_t vec_size = 1;
    switch (q_layout.data_type) {
    case ov::element::f16:
    case ov::element::bf16:
        vec_size = 8;
        break;
    case ov::element::f32:
        vec_size = 8;
        break;
    default:
        vec_size = 1;
        break;
    }

    while (vec_size > 1 && ((k_head_dims % (subgroup_size * vec_size)) != 0)) {
        vec_size /= 2;
    }

    return vec_size;
}

// A call with few tokens per subsequence, decode included: up to 4 tokens for
// one subsequence; for more, at most 8 tokens per subsequence on average for an
// f32 query and at most 16 otherwise. The host sees only the totals, so the
// average stands for the lengths.
//
// NOTE: The thresholds are deliberately powers of 2. They were chosen on Panther
// Lake at (k_head_dims, v_head_dims) = (128, 128) with 16 key and 32 value
// heads, in f16 and in f32 alike from the crossovers measured with one and
// with two subsequences, and serve the other head sizes as well.
// TODO: Consider retuning these thresholds for other devices, other (K, V) head
// sizes and other numbers of subsequences.
bool is_short_call(const RuntimeParams& params) {
    const auto& q_layout = params.get_input_layout(paged_gated_delta_net::QUERY);
    const auto& q_shape = q_layout.get_partial_shape();
    const auto& seq_shape = params.get_input_layout(paged_gated_delta_net::SUBSEQUENCE_BEGINS).get_partial_shape();
    if (q_shape[0].is_dynamic() || seq_shape[0].is_dynamic() || seq_shape[0].get_length() < 2) {
        return false;
    }
    const auto tokens = q_shape[0].get_length();
    const auto sequences = seq_shape[0].get_length() - 1;
    const int64_t per_sequence = q_layout.data_type == ov::element::f32 ? 8 : 16;
    return sequences == 1 ? tokens <= 4 : tokens <= per_sequence * sequences;
}

// Xe cores: the slices times the sub-slices per slice, or 0 where the driver
// does not report them.
size_t get_xe_core_count(const cldnn::device_info& info) {
    return static_cast<size_t>(info.num_slices) * info.num_sub_slices_per_slice;
}

// A call runs in the short stage where is_short_call() holds, and every call does
// on devices with more than 20 Xe cores.
//
// NOTE: On the B70 (BMG-G31), the general stage's kernel run on its own, at
// (k_head_dims, v_head_dims) = (128, 128) with 16 key and 32 value heads, in f16,
// for one subsequence, was on par with the short kernel at 256 tokens and faster
// on longer subsequences up to 16Ki tokens, but the gain did not appear inside
// Qwen3.5-9B-int4-ov. The cause is not known, nor whether it lies in the general
// stage's kernel.
// TODO: Find the cause on the B70 and run the general stage on the devices with
// more than 20 Xe cores again.
bool uses_short_stage(const RuntimeParams& params) {
    return get_xe_core_count(params.get_device_info()) > 20 || is_short_call(params);
}

class PagedGatedDeltaNetBaseGenerator : public KernelGenerator {
public:
    PagedGatedDeltaNetBaseGenerator(std::string_view kernel_name, bool force_ref_path) : KernelGenerator(kernel_name), m_force_ref_path(force_ref_path) {}

protected:
    bool m_force_ref_path = false;

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<paged_gated_delta_net>();

        const auto& q_shape = params.get_input_layout(paged_gated_delta_net::QUERY).get_partial_shape();
        const auto& v_shape = params.get_input_layout(paged_gated_delta_net::VALUE).get_partial_shape();

        const size_t k_head_nums = q_shape[1].get_length();
        const size_t k_head_dims = q_shape[2].get_length();
        const size_t v_head_nums = v_shape[1].get_length();
        const size_t v_head_dims = v_shape[2].get_length();
        const float scale_factor = 1.0f / std::sqrt(static_cast<float>(k_head_dims));

        jit.make("K_HEAD_NUM", k_head_nums);
        jit.make("V_HEAD_NUM", v_head_nums);
        jit.make("K_HEAD_DIM", k_head_dims);
        jit.make("V_HEAD_DIM", v_head_dims);
        jit.make("V_BLOCK_SIZE", get_v_block_size(v_head_dims));
        jit.make("SUBGROUP_SIZE", paged_gated_delta_net_jit::get_subgroup_size(params.get_device_info().arch));
        jit.make("K_VEC_SIZE", m_force_ref_path ? 1 : get_vec_size(params));
        jit.make("FUSE_QK_L2NORM", desc->use_qk_l2norm ? 1 : 0);
        jit.make("Q_L2_NORM_EPS", desc->q_l2_norm_eps);
        jit.make("K_L2_NORM_EPS", desc->k_l2_norm_eps);
        jit.make("SCALE_FACTOR", scale_factor);

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        for (uint32_t i = 0; i < params.input_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }

        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        for (size_t i = 0; i < 10; i++) {
            args.push_back({ArgumentDescriptor::Types::SCALAR, static_cast<uint32_t>(i)});
        }

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;
            const auto& v_shape = params.get_input_layout(paged_gated_delta_net::VALUE).get_partial_shape();
            const auto& seq_shape = params.get_input_layout(paged_gated_delta_net::SUBSEQUENCE_BEGINS).get_partial_shape();

            const size_t sequences = seq_shape[0].get_length() > 0 ? seq_shape[0].get_length() - 1 : 0;
            const size_t head_nums = v_shape[1].get_length();
            const size_t v_head_dims = v_shape[2].get_length();
            const size_t current_v_block_size = get_v_block_size(v_head_dims);
            const size_t v_blocks = (v_head_dims + current_v_block_size - 1) / current_v_block_size;
            const size_t subgroup_size = paged_gated_delta_net_jit::get_subgroup_size(params.get_device_info().arch);

            auto get_head_offset = [](const cldnn::layout& layout, size_t head_dim_idx) {
                const auto& lower_pads = layout.data_padding._lower_size;
                return lower_pads.size() > head_dim_idx ? lower_pads[head_dim_idx] : 0;
            };

            const auto& q_layout = params.input_layouts[paged_gated_delta_net::QUERY];
            const auto& k_layout = params.input_layouts[paged_gated_delta_net::KEY];
            const auto& v_layout = params.input_layouts[paged_gated_delta_net::VALUE];
            auto read_pitch = [](const cldnn::layout& layout, size_t idx) -> int32_t {
                const auto& pitches = layout.get_pitches();
                if (idx < pitches.size()) {
                    return static_cast<int32_t>(pitches[idx]);
                }
                return 1;
            };

            const int32_t q_token_stride = read_pitch(q_layout, 0);
            const int32_t q_head_stride = read_pitch(q_layout, 1);
            const int32_t k_token_stride = read_pitch(k_layout, 0);
            const int32_t k_head_stride = read_pitch(k_layout, 1);
            const int32_t v_token_stride = read_pitch(v_layout, 0);
            const int32_t v_head_stride = read_pitch(v_layout, 1);

            const int32_t query_head_offset = static_cast<int32_t>(get_head_offset(q_layout, 1));
            const int32_t key_head_offset = static_cast<int32_t>(get_head_offset(k_layout, 1));
            const int32_t value_head_offset = static_cast<int32_t>(get_head_offset(v_layout, 1));

            wgs.global = {sequences, head_nums, v_blocks * subgroup_size};
            wgs.local = {1, 1, subgroup_size};

            kd.params.scalars.clear();
            std::vector<int32_t> scalars{
                static_cast<int32_t>(sequences),
                query_head_offset,
                key_head_offset,
                value_head_offset,
                q_token_stride,
                q_head_stride,
                k_token_stride,
                k_head_stride,
                v_token_stride,
                v_head_stride,
            };

            for (auto v : scalars) {
                scalar_desc desc;
                desc.t = scalar_desc::Types::INT32;
                desc.v.s32 = v;
                kd.params.scalars.push_back(desc);
            }
        }};
    }
};

class PagedGatedDeltaNetRefGenerator : public PagedGatedDeltaNetBaseGenerator {
public:
    PagedGatedDeltaNetRefGenerator() : PagedGatedDeltaNetBaseGenerator("paged_gated_delta_net_ref", true) {}
};

class PagedGatedDeltaNetShortGenerator : public PagedGatedDeltaNetBaseGenerator {
public:
    PagedGatedDeltaNetShortGenerator() : PagedGatedDeltaNetBaseGenerator("paged_gated_delta_net_short", false) {}
};

class PagedGatedDeltaNetOptGenerator : public PagedGatedDeltaNetBaseGenerator {
public:
    PagedGatedDeltaNetOptGenerator() : PagedGatedDeltaNetBaseGenerator("paged_gated_delta_net_opt", false) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = PagedGatedDeltaNetBaseGenerator::get_jit_constants(params);

        const auto& q_shape = params.get_input_layout(paged_gated_delta_net::QUERY).get_partial_shape();
        const size_t k_head_dims = q_shape[2].get_length();
        const auto arch = params.get_device_info().arch;
        const bool d_on_lanes = paged_gated_delta_net_jit::uses_d_on_lanes(k_head_dims, arch);
        // NOTE: For k_head_dims > 128, where the tiling is always d-on-lanes, every
        // call runs the plain token loop with one accumulator chain and no
        // prefetch, instead of the staged loop with 4 chains and an 8-token
        // prefetch. Tested on Panther Lake at (256, 256), f16 and f32, where the
        // staged loop is faster on some very long subsequences and on some calls
        // with many subsequences.
        const bool plain_loop = k_head_dims > 128;

        jit.make("D_LANES", d_on_lanes ? 1 : 0);
        jit.remove("V_BLOCK_SIZE");
        jit.make("V_BLOCK_SIZE", paged_gated_delta_net_jit::get_v_rows_per_thread(k_head_dims, arch));
        jit.make("PLAIN_TOKEN_LOOP", plain_loop ? 1 : 0);

        // Independent accumulator chains in the two dot products over the k axis.
        jit.make("ACC_WAYS", plain_loop ? 1 : 4);

        // Prefetch distance in tokens, and the request size in bytes.
        jit.make("PF_AHEAD", plain_loop ? 0 : 8);
        jit.make("PF_BYTES", 8);

        jit.make("OCL_REQD_WG", d_on_lanes ? 1 : 0);

        return jit;
    }

    // The state lives in registers, so the kernel is built for the large
    // register file where the target has one.
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        auto options = PagedGatedDeltaNetBaseGenerator::get_build_options(params);
        if (paged_gated_delta_net_jit::supports_large_grf(params.get_device_info().arch)) {
            options += " -cl-intel-256-GRF-per-thread";
        }
        return options;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;
            const auto& q_shape = params.get_input_layout(paged_gated_delta_net::QUERY).get_partial_shape();
            const auto& v_shape = params.get_input_layout(paged_gated_delta_net::VALUE).get_partial_shape();
            const auto& seq_shape = params.get_input_layout(paged_gated_delta_net::SUBSEQUENCE_BEGINS).get_partial_shape();

            const size_t sequences = seq_shape[0].get_length() > 0 ? seq_shape[0].get_length() - 1 : 0;
            const size_t head_nums = v_shape[1].get_length();
            const size_t k_head_dims = q_shape[2].get_length();
            const size_t v_head_dims = v_shape[2].get_length();
            const auto arch = params.get_device_info().arch;
            const size_t subgroup_size = paged_gated_delta_net_jit::get_subgroup_size(arch);
            const size_t v_rows = paged_gated_delta_net_jit::get_v_rows_per_thread(k_head_dims, arch);
            const size_t v_tiles = v_head_dims / v_rows;

            // The tile axis goes in dimension 0, so that the tiles of one head,
            // which read the same q and k rows, have consecutive group ids.
            wgs.global = {v_tiles * subgroup_size, head_nums, sequences};
            wgs.local = {subgroup_size, 1, 1};

            auto get_head_offset = [](const cldnn::layout& layout, size_t head_dim_idx) {
                const auto& lower_pads = layout.data_padding._lower_size;
                return lower_pads.size() > head_dim_idx ? lower_pads[head_dim_idx] : 0;
            };

            const auto& q_layout = params.input_layouts[paged_gated_delta_net::QUERY];
            const auto& k_layout = params.input_layouts[paged_gated_delta_net::KEY];
            const auto& v_layout = params.input_layouts[paged_gated_delta_net::VALUE];
            auto read_pitch = [](const cldnn::layout& layout, size_t idx) -> int32_t {
                const auto& pitches = layout.get_pitches();
                if (idx < pitches.size()) {
                    return static_cast<int32_t>(pitches[idx]);
                }
                return 1;
            };

            kd.params.scalars.clear();
            std::vector<int32_t> scalars{
                static_cast<int32_t>(sequences),
                static_cast<int32_t>(get_head_offset(q_layout, 1)),
                static_cast<int32_t>(get_head_offset(k_layout, 1)),
                static_cast<int32_t>(get_head_offset(v_layout, 1)),
                read_pitch(q_layout, 0),
                read_pitch(q_layout, 1),
                read_pitch(k_layout, 0),
                read_pitch(k_layout, 1),
                read_pitch(v_layout, 0),
                read_pitch(v_layout, 1),
            };

            for (auto v : scalars) {
                scalar_desc desc;
                desc.t = scalar_desc::Types::INT32;
                desc.v.s32 = v;
                kd.params.scalars.push_back(desc);
            }
        }};
    }
};

class PagedGatedDeltaNetRefImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::PagedGatedDeltaNetRefImpl)

    Stage::Ptr paged_gated_delta_net = make_stage<PagedGatedDeltaNetRefGenerator>();

    PagedGatedDeltaNetRefImpl() : PrimitiveImplOCL(PagedGatedDeltaNetRef::get_type_info_static()) {}
    PagedGatedDeltaNetRefImpl(const program_node& node, const RuntimeParams& params) : PagedGatedDeltaNetRefImpl() {
        add_stage(paged_gated_delta_net, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<PagedGatedDeltaNetRefImpl>(this);
    }
};

class PagedGatedDeltaNetOptImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::PagedGatedDeltaNetOptImpl)

    Stage::Ptr paged_gated_delta_net = make_stage<PagedGatedDeltaNetOptGenerator>();
    Stage::Ptr paged_gated_delta_net_short = make_stage<PagedGatedDeltaNetShortGenerator>();

    PagedGatedDeltaNetOptImpl() : PrimitiveImplOCL(PagedGatedDeltaNetOpt::get_type_info_static()) {}
    PagedGatedDeltaNetOptImpl(const program_node& node, const RuntimeParams& params) : PagedGatedDeltaNetOptImpl() {
        // The general stage is always added, and added first: set_kernels() gives
        // the i-th compiled kernel to the i-th stage declared with make_stage().
        add_stage(paged_gated_delta_net, params);
        if (params.is_dynamic() || uses_short_stage(params)) {
            add_stage(paged_gated_delta_net_short, params);
        }
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<PagedGatedDeltaNetOptImpl>(this);
    }

    // Indices in the order of make_stage().
    [[nodiscard]] std::vector<size_t> get_stages_execution_order(const RuntimeParams& params) const override {
        return {has_stage(paged_gated_delta_net_short) && uses_short_stage(params) ? size_t{1} : size_t{0}};
    }
};

}  // namespace

std::unique_ptr<primitive_impl> PagedGatedDeltaNetRef::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<paged_gated_delta_net>());
    return std::make_unique<PagedGatedDeltaNetRefImpl>(node, params);
}

std::unique_ptr<primitive_impl> PagedGatedDeltaNetOpt::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<paged_gated_delta_net>());
    return std::make_unique<PagedGatedDeltaNetOptImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::paged_gated_delta_net)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::PagedGatedDeltaNetRefImpl)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::PagedGatedDeltaNetOptImpl)
