// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../primitive_ocl_base.hpp"
#include "../utils/jitter.hpp"
#include "../utils/kernel_generator.hpp"
#include "fully_connected_gguf_impl.hpp"
#include "fully_connected_gguf_inst.h"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/runtime/utils.hpp"

#ifdef ENABLE_ONEDNN_FOR_GPU
#    include <oneapi/dnnl/dnnl.hpp>
#    include <unordered_map>

#    include "intel_gpu/runtime/lru_cache.hpp"
#endif

// FullyConnectedGGUF reads the GGUF blocks in its kernels. By activation rows M:
//   M < 8 (decode)  : quantize the activations to int8 per 32 values, then an int8 matrix-vector kernel.
//   M >= 8, Xe2+    : matrix-engine kernel that decodes each block into local memory once for many rows.
//   M >= 8, others  : decode the weights into an f16 scratch buffer (memory pool), then a oneDNN matmul.

namespace ov::intel_gpu::ocl {
namespace {

using ov::op::internal::GGUFDequantize;

enum InternalBufferIdx {
    XPACK = 0,        // quantized activation rows
    WEIGHTS_F16 = 1,  // decoded weights for the oneDNN path, f16 [N, K]
};

constexpr size_t min_prefill_rows = 8;

// Work-group tile of the matrix-engine kernel: sgm x sgn sub-groups, each with tm x tn tiles of 8 x 16, own
// sgm * tm * 8 activation rows and sgn * tn * 16 weight rows.
struct PrefillTile {
    size_t sgm;
    size_t sgn;
    size_t tm;
    size_t tn;
    bool grf256;
    size_t rows_m() const {
        return sgm * tm * 8;
    }
    size_t rows_n() const {
        return sgn * tn * 16;
    }
    size_t local_bytes() const {
        return 2 * rows_n() * 256 * sizeof(ov::float16);  // double-buffered decoded block
    }
};

// Tiles by activation rows. Each decoded block serves the tile's activation rows, so larger M takes larger tiles; small
// M takes small tiles to keep enough work-groups.
constexpr PrefillTile prefill_tiles[] = {
    {8, 1, 1, 4, false},   // M <= 64:  64 x 64
    {16, 1, 1, 4, false},  // M <= 128: 128 x 64
    {16, 1, 2, 4, false},  // M <= 256: 256 x 64
    {16, 2, 4, 4, true},   // otherwise: 512 x 128, 128 KB of local memory
};
constexpr size_t num_prefill_tiles = std::size(prefill_tiles);

size_t prefill_tile_idx(size_t m) {
    for (size_t i = 0; i + 1 < num_prefill_tiles; i++) {
        if (m <= prefill_tiles[i].rows_m()) {
            return i;
        }
    }
    return num_prefill_tiles - 1;
}

// The tile with sgn halved until it fits in the device's local memory.
PrefillTile prefill_tile(const RuntimeParams& params, size_t idx) {
    PrefillTile t = prefill_tiles[idx];
    while (t.sgn > 1 && t.local_bytes() > params.get_device_info().max_local_mem_size) {
        t.sgn /= 2;
    }
    return t;
}

struct GGUFShape {
    size_t m;  // activation rows
    size_t n;  // weight rows
    size_t k;
};

GGUFShape get_gguf_shape(const RuntimeParams& params) {
    const auto format = params.typed_desc<fully_connected_gguf>()->format;
    const auto& w_shape = params.get_input_layout(1).get_shape();
    GGUFShape s{0, w_shape[0], w_shape[1] / GGUFDequantize::block_bytes(format) * GGUFDequantize::block_size};
    const auto& a_shape = params.get_input_layout(0).get_partial_shape();
    if (a_shape.is_static()) {
        s.m = ov::shape_size(a_shape.to_shape()) / s.k;
    }
    return s;
}

bool is_q5_k(const RuntimeParams& params) {
    return params.typed_desc<fully_connected_gguf>()->format == GGUFDequantize::Format::Q5_K;
}

bool has_fused_add(const RuntimeParams& params) {
    return params.has_fused_primitives();
}

// Quantized activation row: int8[K] | float scale[K / 32], and for Q5_K float sum[K / 16].
size_t xpack_bytes(size_t k, bool q5_k) {
    return k + k / 32 * sizeof(float) + (q5_k ? k / 16 * sizeof(float) : 0);
}

enum class Path { gemv, scratch, matrix_engine };

Path get_path(const RuntimeParams& params, size_t m) {
    const auto& config = params.get_program().get_config();
    const auto& forced = config.get_gguf_prefill_path();
    const bool f16 = params.get_input_layout(0).data_type == data_types::f16 && params.get_output_layout(0).data_type == data_types::f16;
    if (m < min_prefill_rows || !f16 || !params.get_device_info().supports_immad || forced == "gemv") {
        return Path::gemv;
    }
    const auto& info = params.get_device_info();
    const bool matrix_engine = info.arch >= gpu_arch::xe2 && info.max_local_mem_size >= prefill_tile(params, 0).local_bytes();
    if (matrix_engine && forced != "scratch") {
        return Path::matrix_engine;
    }
#ifdef ENABLE_ONEDNN_FOR_GPU
    if (config.get_use_onednn()) {
        return Path::scratch;
    }
#endif
    return Path::gemv;
}

class GGUFGeneratorBase : public KernelGenerator {
public:
    explicit GGUFGeneratorBase(std::string_view name, std::string_view suffix = "") : KernelGenerator(name, suffix) {}

protected:
    // N and K only, so layers with the same weight shape share one kernel.
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto s = get_gguf_shape(params);
        jit.make("GGUF_N", s.n);
        jit.make("GGUF_K", s.k);
        jit.make("GGUF_XPACK", xpack_bytes(s.k, is_q5_k(params)));
        jit.make(is_q5_k(params) ? "GGUF_Q5_K" : "GGUF_Q6_K", 1);
        if (has_fused_add(params)) {
            jit.make("GGUF_FUSED_ADD", 1);
        }
        return jit;
    }

    static void add_shape_info(Arguments& args, const RuntimeParams& params) {
        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }
    }

    static void add_fused_add(Arguments& args, const RuntimeParams& params) {
        if (has_fused_add(params)) {
            args.push_back({ArgumentDescriptor::Types::INPUT_OF_FUSED_PRIMITIVE, 0});
        }
    }
};

class GGUFQuantizeGenerator : public GGUFGeneratorBase {
public:
    GGUFQuantizeGenerator() : GGUFGeneratorBase("fully_connected_gguf_quantize") {}

protected:
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        add_shape_info(args, params);
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, XPACK});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            const auto s = get_gguf_shape(params);
            kd.params.workGroups.global = {s.k / 32, std::max<size_t>(s.m, 1), 1};
            kd.params.workGroups.local = {8, 1, 1};
        }};
    }
};

class GGUFGemvGenerator : public GGUFGeneratorBase {
public:
    GGUFGemvGenerator() : GGUFGeneratorBase("fully_connected_gguf_gemv") {}

protected:
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        add_shape_info(args, params);
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, XPACK});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        add_fused_add(args, params);
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            const auto s = get_gguf_shape(params);
            kd.params.workGroups.global = {ceil_div(s.n, 4) * 64, std::max<size_t>(s.m, 1), 1};
            kd.params.workGroups.local = {64, 1, 1};
        }};
    }
};

class GGUFDequantizeGenerator : public GGUFGeneratorBase {
public:
    GGUFDequantizeGenerator() : GGUFGeneratorBase("fully_connected_gguf_dequantize") {}

protected:
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        add_shape_info(args, params);
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, WEIGHTS_F16});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            const auto s = get_gguf_shape(params);
            kd.params.workGroups.global = {align_to(s.n * s.k / 16, 64), 1, 1};
            kd.params.workGroups.local = {64, 1, 1};
        }};
    }
};

class GGUFPrefillGenerator : public GGUFGeneratorBase {
public:
    explicit GGUFPrefillGenerator(size_t tile_idx)
        : GGUFGeneratorBase("fully_connected_gguf_prefill", "m" + std::to_string(prefill_tiles[tile_idx].rows_m())),
          m_tile_idx(tile_idx) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = GGUFGeneratorBase::get_jit_constants(params);
        const auto t = prefill_tile(params, m_tile_idx);
        jit.make("SGM", t.sgm);
        jit.make("SGN", t.sgn);
        jit.make("TM", t.tm);
        jit.make("TN", t.tn);
        return jit;
    }

    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        auto options = GGUFGeneratorBase::get_build_options(params);
        if (prefill_tiles[m_tile_idx].grf256) {
            options += " -cl-intel-256-GRF-per-thread";
        }
        return options;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        add_shape_info(args, params);
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        add_fused_add(args, params);
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[tile_idx = m_tile_idx](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            const auto s = get_gguf_shape(params);
            const auto t = prefill_tile(params, tile_idx);
            kd.params.workGroups.global = {ceil_div(s.n, t.rows_n()) * 16, ceil_div(std::max<size_t>(s.m, 1), t.rows_m()) * t.sgm * t.sgn, 1};
            kd.params.workGroups.local = {16, t.sgm * t.sgn, 1};
            kd.params.scalars.clear();
            scalar_desc m;
            m.t = scalar_desc::Types::INT32;
            m.v.s32 = static_cast<int32_t>(s.m);
            kd.params.scalars.push_back(m);
        }};
    }

private:
    size_t m_tile_idx;
};

class FullyConnectedGGUFOCLImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::FullyConnectedGGUFOCLImpl)

    // Stage order: quantize, gemv, dequantize, prefill tiles (see get_stages_execution_order).
    enum StageIdx : size_t { QUANTIZE = 0, GEMV = 1, DEQUANTIZE = 2, PREFILL = 3 };

    Stage::Ptr quantize = make_stage<GGUFQuantizeGenerator>();
    Stage::Ptr gemv = make_stage<GGUFGemvGenerator>();
    Stage::Ptr dequantize = make_stage<GGUFDequantizeGenerator>();
    std::vector<Stage::Ptr> prefill;

    FullyConnectedGGUFOCLImpl() : PrimitiveImplOCL(FullyConnectedGGUFImpl::get_type_info_static()) {
        for (size_t i = 0; i < num_prefill_tiles; i++) {
            prefill.push_back(make_stage<GGUFPrefillGenerator>(i));
        }
    }
    explicit FullyConnectedGGUFOCLImpl(const RuntimeParams& params) : FullyConnectedGGUFOCLImpl() {
        add_stage(quantize, params);
        add_stage(gemv, params);
        add_stage(dequantize, params);
        for (auto& stage : prefill) {
            add_stage(stage, params);
        }
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<FullyConnectedGGUFOCLImpl>(this);
    }

    std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        const auto s = get_gguf_shape(params);
        const auto path = get_path(params, s.m);
        const size_t xpack = path == Path::gemv ? std::max<size_t>(s.m, 1) * xpack_bytes(s.k, is_q5_k(params)) : 1;
        const size_t weights_f16 = path == Path::scratch ? s.n * s.k : 1;
        return {BufferDescriptor{xpack, ov::element::i8}, BufferDescriptor{weights_f16, ov::element::f16}};
    }

    void update_rt_params(const primitive_inst& instance) override {
        // Only some stages run per execution: keep the update flags on the others until they run.
        const auto flags = instance.get_impl_params()->flags.to_ulong();
        constexpr size_t mask_args = (1 << cldnn::ExecutionFlags::ARG_UPDATE_REQUIRED) | (1 << cldnn::ExecutionFlags::IMPL_CHANGED);
        constexpr size_t mask_dispatch = (1 << cldnn::ExecutionFlags::SHAPE_CHANGED);
        for (auto& stage : _stages) {
            stage->kd.need_args_update |= (flags & mask_args) != 0;
            stage->kd.need_dispatch_data_update |= (flags & mask_dispatch) != 0;
        }
    }

    [[nodiscard]] event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        auto& stream = instance.get_network().get_stream();
        if (instance.can_be_optimized()) {
            return stream.aggregate_events(events, events.size() > 1, instance.is_output());
        }
        update_rt_params(instance);
        const auto& params = *instance.get_impl_params();
        const auto s = get_gguf_shape(params);

        switch (get_path(params, s.m)) {
        case Path::matrix_engine:
            return execute_stage(events, instance, prefill[prefill_tile_idx(s.m)]);
#ifdef ENABLE_ONEDNN_FOR_GPU
        case Path::scratch:
            return execute_matmul({execute_stage(events, instance, dequantize)}, instance, s);
#endif
        default:
            return execute_stage({execute_stage(events, instance, quantize)}, instance, gemv);
        }
    }

    std::vector<size_t> get_stages_execution_order(const cldnn::kernel_impl_params& params) const override {
        const auto s = get_gguf_shape(params);
        switch (get_path(params, s.m)) {
        case Path::matrix_engine:
            return {PREFILL + prefill_tile_idx(s.m)};
        case Path::scratch:
            return {DEQUANTIZE};
        default:
            return {QUANTIZE, GEMV};
        }
    }

private:
#ifdef ENABLE_ONEDNN_FOR_GPU
    struct Matmul {
        dnnl::matmul::primitive_desc pd;
        dnnl::matmul prim;
    };
    cldnn::LruCache<size_t, std::shared_ptr<Matmul>> _matmuls{16};  // by M

    Matmul& get_matmul(primitive_inst& instance, const GGUFShape& s) {
        if (!_matmuls.has(s.m)) {
            using dt = dnnl::memory::data_type;
            using tag = dnnl::memory::format_tag;
            const auto M = static_cast<dnnl::memory::dim>(s.m);
            const auto N = static_cast<dnnl::memory::dim>(s.n);
            const auto K = static_cast<dnnl::memory::dim>(s.k);
            dnnl::memory::desc src_md({M, K}, dt::f16, tag::ab);
            dnnl::memory::desc w_md({K, N}, dt::f16, tag::ba);  // the [N, K] decoded weights
            dnnl::memory::desc dst_md({M, N}, dt::f16, tag::ab);
            dnnl::primitive_attr attr;
            if (has_fused_add(*instance.get_impl_params())) {
                dnnl::post_ops ops;
                ops.append_binary(dnnl::algorithm::binary_add, dst_md);
                attr.set_post_ops(ops);
            }
            auto mm = std::make_shared<Matmul>();
            mm->pd = dnnl::matmul::primitive_desc(instance.get_network().get_engine().get_onednn_engine(), src_md, w_md, dst_md, attr);
            mm->prim = dnnl::matmul(mm->pd);
            _matmuls.add(s.m, mm);
        }
        return *_matmuls.get(s.m);
    }

    event::ptr execute_matmul(const std::vector<event::ptr>& events, primitive_inst& instance, const GGUFShape& s) {
        auto& stream = instance.get_network().get_stream();
        auto& mm = get_matmul(instance, s);
        std::unordered_map<int, dnnl::memory> args{
            {DNNL_ARG_SRC, instance.input_memory(0).get_onednn_memory(mm.pd.src_desc())},
            {DNNL_ARG_WEIGHTS, instance.get_intermediates_memories()[WEIGHTS_F16]->get_onednn_memory(mm.pd.weights_desc())},
            {DNNL_ARG_DST, instance.output_memory(0).get_onednn_memory(mm.pd.dst_desc())},
        };
        if (has_fused_add(*instance.get_impl_params())) {
            args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, instance.fused_memory(0)->get_onednn_memory(mm.pd.dst_desc())});
        }
        // The in-order queue runs the matmul after the dequantize kernel.
        mm.prim.execute(stream.get_onednn_stream(), args);
        return instance.needs_completion_event() ? stream.enqueue_marker({}) : nullptr;
    }
#endif
};

}  // namespace

std::unique_ptr<primitive_impl> FullyConnectedGGUFImpl::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<fully_connected_gguf>());
    return std::make_unique<FullyConnectedGGUFOCLImpl>(params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::fully_connected_gguf)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::FullyConnectedGGUFOCLImpl)
