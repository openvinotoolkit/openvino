#include "fc_compressed_generate_opt.hpp"

#include "../common_utils/dispatch_utils.hpp"
#include "fully_connected_inst.h"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "../primitive_ocl_base.hpp"
#include "../utils/jitter.hpp"
#include "../utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

// Vectorisation / dispatch constants — must match gemm_generate_opt.cl definitions.
static constexpr int SG_SIZE  = 16;   // Intel GPU sub-group width for f16
static constexpr int VEC_SIZE = 8;    // half8 / u4-nibble-vec per iteration
// N-parallel high-occupancy dispatch: each work-item owns 1 output channel.
// WG_SIZE controls how many N-channels share a work-group (for EU occupancy).
static constexpr int WG_SIZE  = 256;
// Legacy constants still emitted for JIT but unused in N-parallel W4A16 path.
// W4A8 path still references TILE_N and FAKE_GROUP_SIZE for compat.
// TILE_N is always 2 (matches reference MoE kernel design).
static constexpr int TILE_N_LARGE = 2;
static constexpr int TILE_N_SMALL = 2;
static constexpr int TILE_N_THRESHOLD = 2048;  // unused when LARGE==SMALL
static constexpr int FAKE_GROUP_SIZE = 128;

// -----------------------------------------------------------------------
// Helper: derive the packed-layout input index ordering from kernel_impl_params.
//
// At kernel_impl_params time (after update_impl_params / fused-op folding),
// the input_layouts are ordered as:
//   [0] activation  (f16 for W4A16, i8 for W4A8)
//   [1] weight      (u4/i4)
//   [2] weight scale (f16)                          ← always present
//   [3] weight ZP   (u4/i4, optional)
//   [3 or 4] activation scale (f16, W4A8 only)
//
// Bias is folded into fused ops and does NOT appear in input_layouts here.
// -----------------------------------------------------------------------
static bool detect_has_zp(const RuntimeParams& params) {
    if (params.input_layouts.size() <= 3)
        return false;
    const auto dt = params.input_layouts[3].data_type;
    return dt == data_types::u4 || dt == data_types::i4 || dt == data_types::u8;
}

static bool detect_zp_is_u8(const RuntimeParams& params) {
    return detect_has_zp(params) &&
           params.input_layouts[3].data_type == data_types::u8;
}

// Returns the input_layouts index of the activation scale, or SIZE_MAX if not present.
// The activation scale is f16 dtype; weight ZP is u4/i4/u8.  We validate dtype to guard
// against layout reordering edge-cases (e.g. activation_precomputed_reduction at [4]).
// NOTE: do NOT check lay.count()==0 here — the layout may have a stale compile-time
// shape {0,1,1} (dynamic dimension), while the actual runtime memory is non-empty.
// Rely on dtype to identify the tensor type.
static size_t act_scale_index(const RuntimeParams& params) {
    const bool act_is_i8 = (params.input_layouts[0].data_type == data_types::i8);
    if (!act_is_i8)
        return SIZE_MAX;
    const size_t candidate = detect_has_zp(params) ? 4 : 3;
    // Validate dtype: the candidate layout must exist and be f16 (act_scale) not u4/u8 (ZP).
    if (candidate >= params.input_layouts.size())
        return SIZE_MAX;
    const auto& lay = params.input_layouts[candidate];
    if (lay.data_type != data_types::f16)
        return SIZE_MAX;
    return candidate;
}

// -----------------------------------------------------------------------
// Kernel generator
// Reuses the "gemm_generate_opt" .cl template, but emits `IS_WEIGHT_INT4=1`
// together with WOQ-specific constants so the `#if IS_WEIGHT_INT4` branch
// inside the template gets compiled.
// When the activation is i8 (W4A8), also emits `IS_ACT_INT8=1` to select
// the inner branch that reads char activation and multiplies by act_scale.
// -----------------------------------------------------------------------
class FCCompressedOptGenerator : public KernelGenerator {
public:
    FCCompressedOptGenerator() : KernelGenerator("gemm_generate_opt") {}

protected:
    // Content-based entry point: FC layers with identical GEMV kernel configs
    // (K, N, B, group_size, data types) produce the same entry point, enabling
    // cross-layer kernel compilation cache hits in s_gemv_compiled_cache.
    // The default base-class entry point uses params.hash() which includes
    // per-instance info (primitive ID, layout addresses), defeating the cache
    // for every new FC layer even when the kernel logic is identical.
    [[nodiscard]] std::string get_entry_point(const RuntimeParams& params) const override {
        const auto& in0    = params.input_layouts[0];
        const auto& in1    = params.input_layouts[1];
        const auto& in_sc  = params.input_layouts[2];
        const auto& shape_a  = in0.get_shape();
        const auto& shape_w  = in1.get_shape();
        const auto& shape_sc = in_sc.get_shape();

        const size_t rank = shape_a.size();
        const size_t K = shape_a[rank - 1];
        const size_t N = shape_w[0];
        size_t B = 1;
        for (size_t i = 0; i + 1 < rank; ++i)
            B *= shape_a[i];
        const size_t num_groups = shape_sc[shape_sc.size() - 1];
        const size_t group_size = (num_groups > 0) ? (K / num_groups) : K;

        const int tile_n = (N >= TILE_N_THRESHOLD) ? TILE_N_LARGE : TILE_N_SMALL;

        return get_kernel_name()
               + "_K" + std::to_string(K)
               + "_N" + std::to_string(N)
               + "_B" + std::to_string(B)
               + "_G" + std::to_string(group_size)
               + "_T" + std::to_string(tile_n)
               + (in1.data_type == data_types::i4 ? "_i4" : "_u4")
               + (detect_has_zp(params) ? "_zp" : "")
               + (detect_zp_is_u8(params) ? "_u8zp" : "")
               + (in0.data_type == data_types::i8 ? "_i8act" : "_f16act");
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);

        // After update_impl_params, input_layouts are reordered to:
        //   [0] activation (f16 or i8)  [1] weight (u4/i4)  [2] scale (f16)  [3] ZP(opt, u4)
        //   [3 or 4] act_scale (f16, W4A8 only)
        const auto& in0   = params.input_layouts[0];  // activation
        const auto& in1   = params.input_layouts[1];  // weight
        const auto& in_sc = params.input_layouts[2];  // weight scale [N, NUM_GROUPS]

        const auto& shape_a  = in0.get_shape();
        const auto& shape_w  = in1.get_shape();   // [N, K] after reshape
        const auto& shape_sc = in_sc.get_shape(); // [N, K/group_size] = [N, NUM_GROUPS]

        // Derive tensor dimensions from (updated) activation and weight layouts.
        const size_t rank = shape_a.size();
        const size_t K    = shape_a[rank - 1];    // reduction dimension
        const size_t N    = shape_w[0];           // weight rows = output features
        // Batch: product of all leading dims of activation above K.
        size_t B = 1;
        for (size_t i = 0; i + 1 < rank; ++i)
            B *= shape_a[i];

        // -----------------------------------------------------------------------
        // Derive group size from the weight-scale tensor shape.
        //
        // Scale shape is [N, NUM_GROUPS] where:
        //   shape_sc[0]      = N        (output channels)
        //   shape_sc[last]   = NUM_GROUPS = K / group_size
        //
        // This matches OneDNN's convention:
        //   ngroups    = scale_layout.get_dim(weight_rank - 1)  // last dim
        //   group_size = K / ngroups
        //
        // Do NOT hardcode 128 — group sizes of 64, 32, … are also valid.
        // -----------------------------------------------------------------------
        const size_t actual_num_groups = shape_sc[shape_sc.size() - 1];
        const size_t actual_group_size = (actual_num_groups > 0) ? (K / actual_num_groups) : K;

        // Determine u4 vs i4.
        const bool weight_is_signed = (in1.data_type == data_types::i4);

        // Detect optional weight ZP — it is present when input[3] is u4/i4.
        const bool has_zp = detect_has_zp(params);

        // Detect W4A8 mode.
        const bool act_is_i8 = (in0.data_type == data_types::i8);

        // Dynamic TILE_N: 4 for large N (better memory-level parallelism), 2 for small N.
        const int tile_n = (N >= TILE_N_THRESHOLD) ? TILE_N_LARGE : TILE_N_SMALL;

        // Dispatch/size constants for K-parallel sub-group approach.
        jit.add({
            make_jit_constant("K_SIZE",           static_cast<int>(K)),
            make_jit_constant("N_SIZE",           static_cast<int>(N)),
            make_jit_constant("B_SIZE",           static_cast<int>(B)),
            make_jit_constant("SG_SIZE",          SG_SIZE),
            make_jit_constant("VEC_SIZE",         VEC_SIZE),
            make_jit_constant("WG_SIZE",          WG_SIZE),
            make_jit_constant("TILE_N",           tile_n),
            make_jit_constant("FAKE_GROUP_SIZE",  FAKE_GROUP_SIZE),
        });

        // WOQ-specific constants.
        const bool zp_is_u8 = detect_zp_is_u8(params);
        jit.add({
            make_jit_constant("IS_WEIGHT_INT4",   1),
            make_jit_constant("IS_ACT_INT8",      act_is_i8 ? 1 : 0),
            make_jit_constant("WEIGHT_IS_SIGNED", weight_is_signed ? 1 : 0),
            make_jit_constant("HAS_ZP",           has_zp ? 1 : 0),
            make_jit_constant("ZP_IS_U8",         zp_is_u8 ? 1 : 0),
            make_jit_constant("GROUP_SIZE",       static_cast<int>(actual_group_size)),
            make_jit_constant("NUM_GROUPS",       static_cast<int>(actual_num_groups)),
        });

        // Float32 accumulator for numerical stability.
        jit.add(make_type_jit_constants("ACCUMULATOR", data_types::f32));

        // DEBUG_FCCmpOpt levels:
        // 0 = off (production)
        // 1 = per-group scale/ZP + final result for n=0..7; also prints [DIAG] Scale[0]/A[0]
        // 2 = also dump raw tensor bytes (verbose, causes ~6x slowdown via GPU printf)
        jit.add(make_jit_constant("DEBUG_FCCmpOpt", 0));

        return jit;
    }

    // Override argument descriptor to explicitly select only the kernel's inputs,
    // skipping any trailing inputs not used by this kernel (e.g. activation_precomputed_reduction).
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        // arg[0]: activation (f16 or i8)
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        // arg[1]: weight (u4/i4 packed)
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});
        // arg[2]: weight scale (f16)
        args.push_back({ArgumentDescriptor::Types::INPUT, 2});

        // arg[3]: weight ZP (u4, optional)
        const bool has_zp = detect_has_zp(params);
        if (has_zp) {
            args.push_back({ArgumentDescriptor::Types::INPUT, 3});
        }

        // For W4A8: arg[3 or 4]: per-token activation scale (f16)
        const size_t as_idx = act_scale_index(params);
        if (as_idx != SIZE_MAX) {
            args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(as_idx)});
        }

        // output
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            assert(!params.is_dynamic());

            const auto& in0   = params.input_layouts[0];
            const auto& in1   = params.input_layouts[1];
            const auto& shape_a = in0.get_shape();

            const size_t rank  = shape_a.size();
            const size_t N     = in1.get_shape()[0];  // weight [N, K]
            size_t B = 1;
            for (size_t i = 0; i + 1 < rank; ++i)
                B *= shape_a[i];

            // SLM-based sub-group cooperative dispatch:
            //   Each work-group has WG_SIZE work-items = num_subgroups subgroups.
            //   Each subgroup handles tile_n output channels (N-block tiling).
            //   N_BLOCK_WG = num_subgroups * tile_n
            //   dim0 = num_work_groups * WG_SIZE, dim1 = B
            const size_t num_subgroups = WG_SIZE / SG_SIZE;
            const int tile_n = (N >= TILE_N_THRESHOLD) ? TILE_N_LARGE : TILE_N_SMALL;
            const size_t n_block_wg = num_subgroups * tile_n;
            const size_t num_wg = (N + n_block_wg - 1) / n_block_wg;
            auto& wgs = kd.params.workGroups;
            wgs.global = {num_wg * WG_SIZE, B, 1};
            wgs.local  = {static_cast<size_t>(WG_SIZE), 1, 1};
        }};
    }
};

// -----------------------------------------------------------------------
class FCCompressedOptImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::FCCompressedOptImpl)

    Stage::Ptr gemv_stage = make_stage<FCCompressedOptGenerator>();

    FCCompressedOptImpl() : PrimitiveImplOCL(FCCompressedGenerateOpt::get_type_info_static()) {}
    FCCompressedOptImpl(const program_node& node, const RuntimeParams& params) : FCCompressedOptImpl() {
        add_stage(gemv_stage, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<FCCompressedOptImpl>(this);
    }

    // -----------------------------------------------------------------------
    // For fully_connected, primitive::input only contains the activation tensor,
    // so inputs_memory_count() == 1.  That means the default get_arguments() in
    // PrimitiveImplOCL only places dep[0] (activation) in data.inputs, and any
    // kernel arg index > 0 causes an assertion failure in set_arguments_impl.
    //
    // We override get_arguments() to populate data.inputs from the raw dependency
    // memories corresponding to each entry in _impl_params->input_layouts.  After
    // OCL update_impl_params the input_layouts order is:
    //   [0] activation  [1] weight  [2] weight_scale  [3] weight_ZP (opt)
    // plus any trailing entries (activation_scale, ...) that were not stripped.
    // dep_memory_ptr(i) for i in [0, input_layouts.size()) correctly returns the
    // memory for input_layouts[i] in the no-bias case (the only path FCCompressedOptImpl
    // supports — validate_impl rejects nodes with bias).
    // -----------------------------------------------------------------------
    [[nodiscard]] cldnn::kernel_arguments_data get_arguments(const cldnn::primitive_inst& instance) const override {
        cldnn::kernel_arguments_data data;
        const auto& params = *instance.get_impl_params();
        const size_t n = params.input_layouts.size();
        const auto* fc_inst = dynamic_cast<const cldnn::fully_connected_inst*>(&instance);
        for (size_t i = 0; i < n; ++i) {
            if (i == 1 && fc_inst) {
                // Use weights_memory() for index 1 (weights): handles dynamic-mode reordering.
                // For static mode it falls back to dep_memory_ptr(1) automatically.
                data.inputs.push_back(fc_inst->weights_memory());
            } else {
                data.inputs.push_back(instance.dep_memory_ptr(i));
            }
        }
        for (size_t i = 0; i < instance.outputs_memory_count(); ++i)
            data.outputs.push_back(instance.output_memory_ptr(i));
        if (instance.has_fused_primitives()) {
            for (size_t i = 0; i < instance.get_fused_mem_count(); ++i)
                data.fused_op_inputs.push_back(instance.fused_memory(i));
        }
        data.shape_info = instance.shape_info_memory_ptr();

        return data;
    }
};

}  // namespace

std::unique_ptr<primitive_impl> FCCompressedGenerateOpt::create_impl(const program_node& node,
                                                                      const RuntimeParams& params) const {
    assert(node.is_type<fully_connected>());
    return std::make_unique<FCCompressedOptImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::FCCompressedOptImpl)
