// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "slice_scatter.hpp"

#include <numeric>

#include "common_utils/jitter.hpp"
#include "data_inst.h"
#include "intel_gpu/primitives/slice_scatter.hpp"
#include "ocl_v2/utils/jitter.hpp"
#include "primitive_inst.h"
#include "primitive_ocl_base.hpp"
#include "slice_scatter_inst.h"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {

namespace {

static constexpr size_t MAX_SUPPORTED_DIM = 5;
static constexpr size_t VEC_SIZE = 8;
static constexpr size_t SIMD_SIZE = 16;

// SliceScatter input port indices
static constexpr size_t kData = 0;
static constexpr size_t kUpdates = 1;
static constexpr size_t kStart = 2;
// kStop = 3 (not used by kernel)
static constexpr size_t kStep = 4;
static constexpr size_t kAxes = 5;

enum KernelsTypes {
    UPDATE_REF = 0,
    UPDATE_OPT,
};

// --- Helper to extract compile-time constant integer data ---

template <typename T, class = typename std::enable_if<std::is_integral<T>::value>::type>
std::vector<int64_t> extractTypedIntegerData(const data_node& node, const stream& stream) {
    mem_lock<T, mem_lock_type::read> lock{node.get_attached_memory_ptr(), stream};
    T* data_ptr = lock.data();
    std::vector<int64_t> result;
    result.reserve(node.get_output_layout().count());
    for (size_t i = 0; i < node.get_output_layout().count(); i++) {
        result.emplace_back(static_cast<int64_t>(data_ptr[i]));
    }
    return result;
}

std::vector<int64_t> extractIntegerData(const data_node& node, const stream& stream) {
    auto dt = node.get_output_layout().data_type;
    switch (dt) {
    case data_types::u8:
        return extractTypedIntegerData<uint8_t>(node, stream);
    case data_types::i8:
        return extractTypedIntegerData<int8_t>(node, stream);
    case data_types::i32:
        return extractTypedIntegerData<int32_t>(node, stream);
    case data_types::i64:
        return extractTypedIntegerData<int64_t>(node, stream);
    default:
        OPENVINO_THROW("[GPU] SliceScatter parameters should be of integral type, got ", dt);
    }
    return {};
}

// --- Configuration for compile-time vs runtime parameters ---

struct ParamConfig {
    std::vector<int64_t> compile_time_start;  // empty => dynamic (buffer passed at runtime)
    std::vector<int64_t> compile_time_step;
    std::vector<int64_t> compile_time_axes;
    ov::element::Type start_data_type = ov::element::i64;
    ov::element::Type step_data_type = ov::element::i64;
    ov::element::Type axes_data_type = ov::element::i64;
    size_t input_rank = 4;
};

ParamConfig getParamConfig(const program_node& node, const RuntimeParams& params) {
    ParamConfig config;
    config.input_rank = params.get_input_layout(kData).get_partial_shape().size();
    const auto& deps = node.get_dependencies();
    const auto& strm = node.get_program().get_stream();

    // Start
    if (deps.size() > kStart && deps[kStart].first->is_constant()) {
        config.compile_time_start = extractIntegerData(deps[kStart].first->as<data>(), strm);
    } else if (deps.size() <= kStart) {
        config.compile_time_start.resize(config.input_rank, 0);
    } else {
        config.start_data_type = deps[kStart].first->get_output_layout(0).data_type;
    }

    // Step
    if (deps.size() > kStep && deps[kStep].first->is_constant()) {
        config.compile_time_step = extractIntegerData(deps[kStep].first->as<data>(), strm);
    } else if (deps.size() <= kStep) {
        config.compile_time_step.resize(config.input_rank, 1);
    } else {
        config.step_data_type = deps[kStep].first->get_output_layout(0).data_type;
    }

    // Axes
    if (deps.size() > kAxes && deps[kAxes].first->is_constant()) {
        config.compile_time_axes = extractIntegerData(deps[kAxes].first->as<data>(), strm);
        for (auto& axis : config.compile_time_axes) {
            if (axis < 0)
                axis += static_cast<int64_t>(config.input_rank);
        }
    } else if (deps.size() <= kAxes) {
        config.compile_time_axes.resize(config.input_rank);
        std::iota(config.compile_time_axes.begin(), config.compile_time_axes.end(), 0);
    } else {
        config.axes_data_type = deps[kAxes].first->get_output_layout(0).data_type;
    }

    return config;
}

// --- JIT generation for buffer parameters ---

void addJitConstantsForParam(JitConstants& jit,
                             const std::string& name,
                             const std::vector<int64_t>& compile_time_param,
                             ov::element::Type data_type,
                             bool is_axes) {
    const std::string BUFF_CONST_NAME = name + "_BUFFER";
    const std::string BUFF_PTR_NAME = name + "_buffer_ptr";

    if (compile_time_param.empty()) {
        // Dynamic: generate buffer pointer declaration for kernel signature
        const std::string type_str = to_ocl_type(data_type);
        jit.add(make_jit_constant(BUFF_CONST_NAME, "__global const " + type_str + "* restrict " + BUFF_PTR_NAME + ","));

        for (size_t i = 0; i < MAX_SUPPORTED_DIM; ++i) {
            const std::string i_str = std::to_string(i);
            const std::string jit_name = name + "_VAL" + i_str;
            std::string access_str;
            if (is_axes) {
                access_str = BUFF_PTR_NAME + "[" + i_str + "] < 0 ? INPUT0_DIMS + " + BUFF_PTR_NAME + "[" + i_str + "] : " + BUFF_PTR_NAME + "[" + i_str + "]";
            } else {
                access_str = BUFF_PTR_NAME + "[" + i_str + "]";
            }
            jit.add(make_jit_constant(jit_name, i_str + " < AXES_BUFFER_SIZE ? (" + access_str + ") : -1"));
        }
    } else {
        // Static: embed values directly as JIT constants
        jit.add(make_jit_constant(BUFF_CONST_NAME, ""));
        for (size_t i = 0; i < MAX_SUPPORTED_DIM; ++i) {
            const std::string jit_name = name + "_VAL" + std::to_string(i);
            const int64_t val = i < compile_time_param.size() ? compile_time_param[i] : -1;
            jit.add(make_jit_constant(jit_name, val));
        }
    }
}

// --- Kernel Generators ---

class SliceScatterBase : public KernelGenerator {
public:
    explicit SliceScatterBase(std::string_view name, const ParamConfig* config) : KernelGenerator(name), m_config(config) {}

protected:
    const ParamConfig* m_config;

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);

        // AXES_BUFFER_SIZE
        if (m_config->compile_time_axes.empty()) {
            // Dynamic axes: derive size from the axes tensor (INPUT5)
            // Axes is a 1D tensor, so batch dimension holds the element count in bfyx format
            jit.add(make_jit_constant("AXES_BUFFER_SIZE", "INPUT5_BATCH_NUM"));
        } else {
            jit.add(make_jit_constant("AXES_BUFFER_SIZE", static_cast<int64_t>(m_config->compile_time_axes.size())));
        }

        addJitConstantsForParam(jit, "START", m_config->compile_time_start, m_config->start_data_type, false);
        addJitConstantsForParam(jit, "STEP", m_config->compile_time_step, m_config->step_data_type, false);
        addJitConstantsForParam(jit, "AXES", m_config->compile_time_axes, m_config->axes_data_type, true);

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }

        // data (INPUT0) and updates (INPUT1) are always needed
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});

        // start (INPUT2) - only if dynamic
        if (m_config->compile_time_start.empty()) {
            args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(kStart)});
        }

        // step (INPUT4) - only if dynamic (subclass may override for opt kernel)
        if (include_step_arg() && m_config->compile_time_step.empty()) {
            args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(kStep)});
        }

        // axes (INPUT5) - only if dynamic
        if (m_config->compile_time_axes.empty()) {
            args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(kAxes)});
        }

        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        return args;
    }

    virtual bool include_step_arg() const {
        return true;
    }
};

class SliceScatterRef : public SliceScatterBase {
public:
    explicit SliceScatterRef(const ParamConfig* config) : SliceScatterBase("slice_scatter_ref", config) {}

protected:
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            const auto& updates_l = params.get_input_layout(kUpdates);
            auto b = extract_channel(ChannelName::BATCH, updates_l);
            auto f = extract_channel(ChannelName::FEATURE, updates_l);
            auto z = extract_channel(ChannelName::Z, updates_l);
            auto y = extract_channel(ChannelName::Y, updates_l);
            auto x = extract_channel(ChannelName::X, updates_l);

            auto& wgs = kd.params.workGroups;
            wgs.global = {b, f, z * y * x};
            wgs.local = ov::intel_gpu::get_optimal_lws(wgs.global, params.get_device_info());
        }};
    }
};

class SliceScatterOpt : public SliceScatterBase {
public:
    explicit SliceScatterOpt(const ParamConfig* config) : SliceScatterBase("slice_scatter_opt", config) {}

protected:
    // Opt kernel has step=1 hardcoded, so step buffer is never a kernel argument
    bool include_step_arg() const override {
        return false;
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);

        jit.add(make_jit_constant("SLICE_SCATTER_VEC_SIZE", static_cast<int64_t>(VEC_SIZE)));
        jit.add(make_jit_constant("SIMD_SIZE", static_cast<int64_t>(SIMD_SIZE)));

        // AXES_BUFFER_SIZE
        if (m_config->compile_time_axes.empty()) {
            jit.add(make_jit_constant("AXES_BUFFER_SIZE", "INPUT5_BATCH_NUM"));
        } else {
            jit.add(make_jit_constant("AXES_BUFFER_SIZE", static_cast<int64_t>(m_config->compile_time_axes.size())));
        }

        // START and AXES follow the config
        addJitConstantsForParam(jit, "START", m_config->compile_time_start, m_config->start_data_type, false);
        addJitConstantsForParam(jit, "AXES", m_config->compile_time_axes, m_config->axes_data_type, true);

        // STEP is always compile-time 1 for opt kernel
        jit.add(make_jit_constant("STEP_BUFFER", ""));
        for (size_t i = 0; i < MAX_SUPPORTED_DIM; ++i) {
            jit.add(make_jit_constant("STEP_VAL" + std::to_string(i), 1));
        }

        return jit;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            const auto& updates_l = params.get_input_layout(kUpdates);
            auto b = extract_channel(ChannelName::BATCH, updates_l);
            auto f = extract_channel(ChannelName::FEATURE, updates_l);
            auto z = extract_channel(ChannelName::Z, updates_l);
            auto y = extract_channel(ChannelName::Y, updates_l);
            auto x = extract_channel(ChannelName::X, updates_l);

            const size_t total_spatial = z * y * x;
            const size_t num_blocks = (total_spatial + VEC_SIZE - 1) / VEC_SIZE;

            auto& wgs = kd.params.workGroups;
            wgs.global = {b, f, num_blocks};
            wgs.local = ov::intel_gpu::get_optimal_lws(wgs.global, params.get_device_info());
        }};
    }
};

// --- PrimitiveImplOCL ---

class SliceScatterImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::SliceScatterImpl)

    ParamConfig m_config;
    Stage::Ptr ref_stage = make_stage<SliceScatterRef>(&m_config);
    Stage::Ptr opt_stage = make_stage<SliceScatterOpt>(&m_config);

    SliceScatterImpl() : PrimitiveImplOCL(SliceScatter::get_type_info_static()) {}
    SliceScatterImpl(const program_node& node, const RuntimeParams& params) : SliceScatterImpl() {
        m_config = getParamConfig(node, params);
        add_stage(ref_stage, params);
        if (canCompileOpt()) {
            add_stage(opt_stage, params);
        }
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        auto copy = make_deep_copy<SliceScatterImpl>(this);
        copy->m_config = m_config;
        return copy;
    }

    std::vector<size_t> get_stages_execution_order(const cldnn::primitive_inst& instance) const override {
        if (canCompileOpt()) {
            auto params = instance.get_impl_params();
            auto x = extract_channel(ChannelName::X, params->get_input_layout(kUpdates));
            if (x >= VEC_SIZE) {
                return {KernelsTypes::UPDATE_OPT};
            }
        }
        return {KernelsTypes::UPDATE_REF};
    }

private:
    bool canCompileOpt() const {
        // Opt kernel requires all steps to be compile-time 1
        if (m_config.compile_time_step.empty())
            return false;
        for (auto s : m_config.compile_time_step) {
            if (s != 1)
                return false;
        }
        return true;
    }
};

}  // namespace

std::unique_ptr<primitive_impl> SliceScatter::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<slice_scatter>());
    return std::make_unique<SliceScatterImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::slice_scatter)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::SliceScatterImpl)
