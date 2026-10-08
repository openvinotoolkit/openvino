// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <llvm/ADT/SmallVector.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/convert_common.hpp"
#include "gc/GpuCompiler.h"
#include "gc/Runtime/GpuRuntime.h"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/kernel.hpp"
#include "intel_gpu/runtime/kernel_args.hpp"
#include "intel_gpu/runtime/kernel_builder.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "interface/gpu_runtime.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "primitive_inst.h"

namespace ov::intel_gpu::mlir {

namespace {

using ProgramArgs = std::vector<void*>;
using ProgramStorage = std::vector<::gc::gpu::Program::ArgStorage>;

// GpuRuntime::Event is an incomplete opaque type. The pointer is the address of a
// cldnn::event::ptr slot owned by GcGpuRuntime::events.
::gc::gpu::GpuRuntime::Event* as_event(cldnn::event::ptr& slot) {
    return reinterpret_cast<::gc::gpu::GpuRuntime::Event*>(&slot);
}

cldnn::event::ptr& as_slot(::gc::gpu::GpuRuntime::Event* event) {
    return *reinterpret_cast<cldnn::event::ptr*>(event);
}

cldnn::scalar_desc::Types to_scalar_type(::gc::gpu::GpuRuntime::Kernel::ArgType type) {
    using ArgType = ::gc::gpu::GpuRuntime::Kernel::ArgType;
    switch (type) {
    case ArgType::I8:
        return cldnn::scalar_desc::Types::INT8;
    case ArgType::UI8:
        return cldnn::scalar_desc::Types::UINT8;
    case ArgType::I16:
        return cldnn::scalar_desc::Types::INT16;
    case ArgType::UI16:
        return cldnn::scalar_desc::Types::UINT16;
    case ArgType::I32:
        return cldnn::scalar_desc::Types::INT32;
    case ArgType::UI32:
        return cldnn::scalar_desc::Types::UINT32;
    case ArgType::I64:
        return cldnn::scalar_desc::Types::INT64;
    case ArgType::UI64:
        return cldnn::scalar_desc::Types::UINT64;
    case ArgType::F32:
        return cldnn::scalar_desc::Types::FLOAT32;
    case ArgType::F64:
        return cldnn::scalar_desc::Types::FLOAT64;
    case ArgType::PTR:
        break;
    }
    OPENVINO_THROW("[GPU] Unexpected scalar ArgType for an MLIR kernel");
}

// Created by the compiled program and destroyed by it, thus outliving the runtime.
// Holds no per-stream state: the kernels and their argument descriptors are cached
// by the stream-local runtime, see GcGpuRuntime::kernel_data().
struct GcKernel final : ::gc::gpu::GpuRuntime::Kernel {
    GcKernel(const uint8_t* bin, size_t size, std::string name, uint32_t n_args, const ArgType* arg_types)
        : bin(bin),
          size(size),
          name(std::move(name)),
          argTypes(arg_types, arg_types + n_args) {}

    const uint8_t* const bin;
    const size_t size;
    const std::string name;
    const std::vector<ArgType> argTypes;
};

class GcGpuRuntime final : public ::gc::gpu::GpuRuntime, public MLIRGpuRuntime {
public:
    GcGpuRuntime(cldnn::stream& stream, cldnn::engine& engine) : gpuStream(stream), gpuEngine(engine) {
        const auto& info = engine.get_device_info();
        deviceInfo = ::gc::gpu::DeviceInfo{info.device_id};
        deviceInfo.name = info.dev_name;
        deviceInfo.maxWgSize = static_cast<uint32_t>(info.max_work_group_size);
    }

    [[nodiscard]] cldnn::stream& stream() const {
        return gpuStream;
    }

    [[nodiscard]] const ::gc::gpu::DeviceInfo& getDeviceInfo() const override {
        return deviceInfo;
    }

    Kernel* createKernel(const uint8_t* data, size_t size, const char* name, uint32_t n_args, Kernel::ArgType* arg_types) override {
        return new GcKernel(data, size, name, n_args, arg_types);
    }

    Event* launch(Kernel* kernel,
                  void** args,
                  uint32_t n_args,
                  uint32_t grid_x,
                  uint32_t grid_y,
                  uint32_t grid_z,
                  uint32_t block_x,
                  uint32_t block_y,
                  uint32_t block_z,
                  Event* const* deps,
                  uint32_t n_deps) override {
        OPENVINO_DEBUG_ASSERT(grid_x && grid_y && grid_z && block_x && block_y && block_z, "[GPU] MLIR kernel launch with a zero grid or local size");

        const auto& gc_kernel = *static_cast<GcKernel*>(kernel);
        auto& kd = kernel_data(gc_kernel, n_args);

        auto& global = kd.desc.workGroups.global;
        auto& local = kd.desc.workGroups.local;
        global[0] = static_cast<size_t>(grid_x) * block_x;
        global[1] = static_cast<size_t>(grid_y) * block_y;
        global[2] = static_cast<size_t>(grid_z) * block_z;
        local[0] = block_x;
        local[1] = block_y;
        local[2] = block_z;

        set_arg_values(kd, gc_kernel.argTypes, args);
        gpuStream.set_arguments(*kd.kernel, kd.desc, kd.args);
        auto* ev = store(gpuStream.enqueue_kernel(*kd.kernel, kd.desc, kd.args, collect_events(deps, n_deps), /*is_output_event=*/true));
        eventScratch.clear();
        kd.args.intermediates.clear();
        return ev;
    }

    void* allocDevice(size_t size, size_t alignment) override {
        return alloc(size, alignment, cldnn::allocation_type::usm_device);
    }
    void* allocShared(size_t size, size_t alignment) override {
        return alloc(size, alignment, cldnn::allocation_type::usm_shared);
    }
    void* allocHost(size_t size, size_t alignment) override {
        return alloc(size, alignment, cldnn::allocation_type::usm_host);
    }

    void free(void* ptr) override {
        OPENVINO_ASSERT(ptr != nullptr, "[GPU] An MLIR program attempts to free a null pointer");
        auto* held = reinterpret_cast<cldnn::memory::ptr*>(ptr);
        OPENVINO_ASSERT(*held, "[GPU] An MLIR program attempts to free memory it does not own: ", ptr);
        held->reset();
    }

    Event* memcpy(const void* src, void* dst, size_t size, Event* const* deps, uint32_t n_deps) override {
        const auto* src_mem = reinterpret_cast<const cldnn::memory::ptr*>(src);
        auto* dst_mem = reinterpret_cast<cldnn::memory::ptr*>(dst);
        OPENVINO_ASSERT(src_mem && *src_mem && dst_mem && *dst_mem, "[GPU] MLIR memcpy arguments must be memory::ptr*");

        if (n_deps > 0) {
            // copy_from() takes no dependencies, order the copy after them on the queue.
            gpuStream.enqueue_marker(collect_events(deps, n_deps), /*is_output_event=*/false);
            gpuStream.enqueue_barrier();
            eventScratch.clear();
        }
        return store((*dst_mem)->copy_from(gpuStream, **src_mem, 0, 0, size, /*blocking=*/false));
    }

    void wait(Event* const* events, uint32_t n_events) override {
        gpuStream.wait_for_events(collect_events(events, n_events));
        for (uint32_t i = 0; i < n_events; ++i) {
            as_slot(events[i]).reset();
        }
        eventScratch.clear();
    }

    cldnn::memory::ptr* store_memory(cldnn::memory::ptr mem) {
        memories.push_back(std::move(mem));
        return &memories.back();
    }

    Event* store(cldnn::event::ptr event) {
        events.push_back(std::move(event));
        return as_event(events.back());
    }

    // Copies the events into the scratch vector, reused to keep the capacity.
    const std::vector<cldnn::event::ptr>& collect_events(Event* const* events, uint32_t n) {
        eventScratch.clear();
        for (uint32_t i = 0; i < n; ++i) {
            if (const auto& ev = as_slot(events[i])) {
                eventScratch.push_back(ev);
            }
        }
        return eventScratch;
    }

    static void register_self() {
        MLIRGpuRuntime::create = &create_runtime;
    }

private:
    struct KernelData {
        cldnn::kernel::ptr kernel;
        // The argument and scalar descriptors are built once, only the values change per launch.
        cldnn::kernel_arguments_desc desc;
        cldnn::kernel_arguments_data args;
    };

    // The kernels are stream-local, while GcKernel is shared by all the streams running the program.
    // The program, and thus the GcKernel keys, outlive this runtime.
    KernelData& kernel_data(const GcKernel& kernel, uint32_t n_args) {
        OPENVINO_DEBUG_ASSERT(kernel.argTypes.size() == n_args,
                              "[GPU] An MLIR kernel is launched with ",
                              n_args,
                              " arguments, while ",
                              kernel.argTypes.size(),
                              " are declared");

        if (auto it = kernels.find(&kernel); it != kernels.end()) {
            return it->second;
        }

        KernelData kd;
        std::vector<cldnn::kernel::ptr> built;
        gpuEngine.create_kernel_builder()->build_kernels(kernel.bin, kernel.size, cldnn::KernelFormat::NATIVE_BIN, "", built);
        auto found = std::find_if(built.begin(), built.end(), [&kernel](const cldnn::kernel::ptr& k) {
            return k->get_id() == kernel.name;
        });
        OPENVINO_ASSERT(found != built.end(), "[GPU] Kernel '", kernel.name, "' is not found in the compiled MLIR module");
        kd.kernel = std::move(*found);

        uint32_t mem_idx = 0;
        kd.desc.arguments.reserve(n_args);
        for (auto type : kernel.argTypes) {
            if (type == Kernel::ArgType::PTR) {
                kd.desc.arguments.push_back({cldnn::argument_desc::Types::INTERNAL_BUFFER, mem_idx++});
            } else {
                kd.desc.arguments.push_back({cldnn::argument_desc::Types::SCALAR, static_cast<uint32_t>(kd.desc.scalars.size())});
                kd.desc.scalars.push_back({to_scalar_type(type), {}});
            }
        }
        kd.args.intermediates.reserve(mem_idx);
        auto& cached = kernels.emplace(&kernel, std::move(kd)).first->second;
        cached.args.scalars = &cached.desc.scalars;
        return cached;
    }

    static void set_arg_values(KernelData& kd, const std::vector<Kernel::ArgType>& types, void** args) {
        kd.args.intermediates.clear();
        size_t scalar_idx = 0;
        for (size_t i = 0; i < types.size(); ++i) {
            if (types[i] == Kernel::ArgType::PTR) {
                auto* held = *reinterpret_cast<cldnn::memory::ptr**>(args[i]);
                OPENVINO_ASSERT(held && *held, "[GPU] MLIR kernel pointer argument ", i, " is null");
                kd.args.intermediates.push_back(*held);
            } else {
                std::memcpy(&kd.desc.scalars[scalar_idx++].v, args[i], Kernel::argSize(types[i]));
            }
        }
    }

    void* alloc(size_t size, size_t alignment, cldnn::allocation_type type) {
        cldnn::layout layout({static_cast<int64_t>(std::max<size_t>(size, 1))}, ov::element::u8, cldnn::format::bfyx);
        auto mem = gpuEngine.allocate_memory(layout, type, /*reset=*/false);
        void* ptr = mem->buffer_ptr();
        OPENVINO_ASSERT(ptr != nullptr, "[GPU] Failed to allocate ", size, " bytes of USM memory");
        // USM allocator chooses alignment; only verify the GC request.
        OPENVINO_ASSERT(alignment == 0 || reinterpret_cast<uintptr_t>(ptr) % alignment == 0,
                        "[GPU] USM allocation is not aligned to ",
                        alignment,
                        " bytes as requested by an MLIR program");
        return store_memory(std::move(mem));
    }

    friend class GcGpuProgram;
    cldnn::stream& gpuStream;
    cldnn::engine& gpuEngine;
    ::gc::gpu::DeviceInfo deviceInfo;
    std::unordered_map<const GcKernel*, KernelData> kernels;
    std::deque<cldnn::memory::ptr> memories;
    std::deque<cldnn::event::ptr> events;
    std::vector<cldnn::event::ptr> eventScratch;
    std::vector<::gc::gpu::GpuRuntime::Event*> eventHandles;
    ProgramArgs args;
    ProgramStorage storage;

    static std::unique_ptr<MLIRGpuRuntime> create_runtime(cldnn::stream& stream, cldnn::engine& engine) {
        return std::make_unique<GcGpuRuntime>(stream, engine);
    }
};

class GcGpuProgram final : public MLIRGpuProgram {
public:
    GcGpuProgram(::gc::gpu::GpuCompiler& compiler, ::mlir::OwningOpRef<::mlir::ModuleOp> module, uint32_t device_id) {
        const bool dump = is_debug();
        if (dump) {
            OPENVINO_MLIR_DEBUG_PRINT("-------------- Source MLIR --------------");
            module->dump();
            OPENVINO_MLIR_DEBUG_PRINT("-----------------------------------------");
        }

        ::gc::gpu::GpuCompiler::Options opts(device_id);
        opts.wait = false;
        opts.dump = dump;
        ::mlir::ModuleOp mod = module.get();
        future = compiler.compile(mod, opts);
    }

    void wait_compiled() override {
        if (program) {
            return;
        }
        program = future.get();
        OPENVINO_ASSERT(program != nullptr, "[GPU] Failed to compile an MLIR subgraph with the Graph Compiler");
        future = {};
    }

    cldnn::event::ptr execute(MLIRGpuRuntime& runtime,
                              const ov::Node& op,
                              cldnn::primitive_inst& instance,
                              const std::vector<cldnn::event::ptr>& deps,
                              bool need_event) override {
        OPENVINO_ASSERT(program != nullptr, "[GPU] MLIR program is not compiled");
        OPENVINO_ASSERT(op.get_input_size() == instance.inputs_memory_count(), "[GPU] mlir_primitive '", instance.id(), "': input count mismatch");
        OPENVINO_ASSERT(op.get_output_size() == instance.outputs_memory_count(), "[GPU] mlir_primitive '", instance.id(), "': output count mismatch");
        auto& rt = static_cast<GcGpuRuntime&>(runtime);
        rt.args.clear();
        rt.storage.clear();

        ::gc::gpu::Program::ArgsBuilder<ProgramArgs, ProgramStorage> builder(rt.args, rt.storage);
        const auto store_memref = [&](const cldnn::memory::ptr& mem, size_t idx, bool input, unsigned rank, ov::element::Type type) {
            const auto argument = [&]() -> std::string {
                if (!input) {
                    return "output " + std::to_string(idx);
                }
                const auto& dependency = instance.dependencies().at(idx);
                return "input " + std::to_string(idx) + " from dependency '" + dependency.first->id() + "' (" +
                       dependency.first->get_node().get_primitive()->type_string() + ")";
            };

            const auto& layout = mem->get_layout();
            OPENVINO_ASSERT(cldnn::format::is_default_format(layout.format), "[GPU] Only plain layouts are supported by mlir_primitive '", instance.id(), "'");
            OPENVINO_ASSERT(mem->buffer_ptr() != nullptr, "[GPU] mlir_primitive '", instance.id(), "' requires USM buffers");
            OPENVINO_ASSERT(layout.data_type == type,
                            "[GPU] mlir_primitive '",
                            instance.id(),
                            "', ",
                            argument(),
                            ": element type mismatch, layout ",
                            layout.data_type,
                            " vs ",
                            type);
            // create_subbuffer borrows; this slot keeps the parent alive.
            auto* held = rt.store_memory(mem);

            const auto& ps = layout.get_partial_shape();
            OPENVINO_ASSERT(ps.size() >= rank, "[GPU] mlir_primitive '", instance.id(), "': expected rank ", rank, ", got ", ps.size());
            // A plain layout without inner padding is dense row-major, the outermost padding
            // shifts the offset only, thus the strides are the trailing dimension products.
            llvm::SmallVector<int64_t, 12> strides(ps.size());
            int64_t stride = 1;
            for (size_t i = ps.size(); i--;) {
                OPENVINO_ASSERT(ps[i].is_static(), "[GPU] mlir_primitive '", instance.id(), "' requires static buffer dimensions");
                OPENVINO_ASSERT(i < rank || ps[i].get_length() == 1,
                                "[GPU] mlir_primitive '",
                                instance.id(),
                                "': trailing dimension ",
                                i,
                                " of ",
                                argument(),
                                " is not 1");
                strides[i] = stride;
                stride *= ps[i].get_length();
            }

            const auto& lower = layout.data_padding._lower_size;
            const auto has_lower_padding = std::any_of(lower.begin(), lower.begin() + layout.format.dimension(), [](auto pad) {
                return pad != 0;
            });
            if (const auto offset = has_lower_padding ? layout.get_linear_offset() : 0) {
                // GC launch ignores the memref offset field; apply it as a USM subbuffer.
                auto view_layout = layout;
                view_layout.data_padding = {};
                const auto bytes = offset * ov::element::Type(layout.data_type).size();
                auto* engine = mem->get_engine();
                OPENVINO_ASSERT(engine != nullptr, "[GPU] mlir_primitive '", instance.id(), "' memory has no engine");
                held = rt.store_memory(engine->create_subbuffer(*mem, view_layout, bytes));
            }

            builder.storeMemref(
                held,
                rank,
                [&](unsigned i) {
                    return ps[i].get_length();
                },
                [&](unsigned i) {
                    return strides[i];
                });
        };

        llvm::SmallVector<unsigned, 16> ranks;
        ranks.reserve(op.get_input_size() + op.get_output_size());
        const auto store_all = [&](size_t n, bool input) {
            for (size_t i = 0; i < n; ++i) {
                const auto rank = input ? op.get_input_partial_shape(i).rank() : op.get_output_partial_shape(i).rank();
                OPENVINO_ASSERT(rank.is_static(), "[GPU] MLIR subgraph arguments must have a static rank");
                const auto r = static_cast<unsigned>(rank.get_length());
                store_memref(input ? instance.input_memory_ptr(i) : instance.output_memory_ptr(i),
                             i,
                             input,
                             r,
                             input ? op.get_input_element_type(i) : op.get_output_element_type(i));
                ranks.push_back(r);
            }
        };
        store_all(op.get_input_size(), /*input=*/true);
        store_all(op.get_output_size(), /*input=*/false);

        size_t storage_idx = 0;
        for (auto r : ranks) {
            builder.appendStoredMemref(storage_idx, r);
        }

        rt.eventHandles.clear();
        for (const auto& dep : deps) {
            rt.eventHandles.push_back(rt.store(dep));
        }

        auto* gc_runtime = static_cast<::gc::gpu::GpuRuntime*>(&rt);
        auto** events =
            program->main(rt.args, gc_runtime, rt.eventHandles.empty() ? nullptr : rt.eventHandles.data(), static_cast<uint32_t>(rt.eventHandles.size()));
        const auto& completions = rt.collect_events(events, program->eventNum());
        auto event = completions.empty() ? rt.stream().enqueue_marker(deps, need_event) : rt.stream().aggregate_events(completions, /*group=*/true, need_event);
        rt.events.clear();
        rt.memories.clear();
        rt.eventScratch.clear();
        return event;
    }

private:
    ::gc::gpu::FutureProgram future;
    std::shared_ptr<::gc::gpu::Program> program;
};

}  // namespace

void register_mlir_gpu_runtime() {
    GcGpuRuntime::register_self();
}

std::shared_ptr<MLIRGpuProgram> create_gpu_program(::mlir::OwningOpRef<::mlir::ModuleOp> module, uint32_t device_id) {
    return std::make_shared<GcGpuProgram>(::gc::gpu::GpuCompiler::get(), std::move(module), device_id);
}

}  // namespace ov::intel_gpu::mlir
