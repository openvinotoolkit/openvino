// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// MegaKernel plugin implementation — task-system-scheduled decoder.
// The entire Qwen3 decoder (all layers) runs in ONE kernel launch per token, but
// instead of a persistent grid + grid-wide software barrier, the work is driven
// by the fine-tuned GPU task system (MEGAKERNEL_POC/research/preloading_gemv):
// a pool of persistent worker work-groups pulls topologically-sorted tasks FIFO
// from a shared work queue; each layer stage is a set of per-workgroup tiles, and
// inter-stage ordering (the old grid barrier) is expressed as global atomic
// sync-flag dependencies resolved by the tasks themselves. Decode is one launch;
// prefill loops it per token. Key techniques: intel_sub_group_block_read, fused
// RMSNorm, fused RoPE, workgroup-cooperative flash-decoding attention, split-K GEMV.

#include "megakernel.hpp"

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include "../primitive_ocl_base.hpp"
#include "iOVIntegration.h"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/primitives/megakernel.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "megakernelImpl.h"
#include "megakernel_inst.h"
#include "ocl/ocl_engine.hpp"
#include "ocl/ocl_event.hpp"
#include "ocl/ocl_stream.hpp"

namespace ov::intel_gpu::ocl {

using cldnn::ocl::ocl_engine;
using cldnn::ocl::ocl_event;
using cldnn::ocl::ocl_stream;

namespace {

// ---------------------------------------------------------------------------
// MegaKernelFastImpl
// ---------------------------------------------------------------------------

class MegaKernelFastImpl : public cldnn::primitive_impl {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::MegaKernelFastImpl)

    MegaKernelFastImpl() = default;
    explicit MegaKernelFastImpl(const cldnn::program_node&, const RuntimeParams&) {}
    // Copy constructor: copy the primitive_impl base subobject so metadata such
    // as m_manager and the dynamic flag are preserved (required by the impl
    // caches in ImplementationsFactory). The OpenCL/runtime members below keep
    // their default null/zero initializers so device state is re-created lazily.
    MegaKernelFastImpl(const MegaKernelFastImpl& other) : cldnn::primitive_impl(other) {}

    [[nodiscard]] std::unique_ptr<cldnn::primitive_impl> clone() const override {
        OPENVINO_ASSERT(megakernelRuntime_ == nullptr,
                        "[GPU] MegaKernelFastImpl::clone() should not be called if megakernel runtime is initialized; use create_impl() instead.");
        return std::make_unique<MegaKernelFastImpl>(*this);
    }
    bool is_cpu() const override {
        return false;
    }
    void save(BinaryOutputBuffer&) const override {}
    void load(BinaryInputBuffer&) override {}
    void init_kernels(const cldnn::kernels_cache&, const cldnn::kernel_impl_params&) override {}
    void set_arguments(cldnn::primitive_inst&) override {}
    void set_arguments(cldnn::primitive_inst&, cldnn::kernel_arguments_data&) override {}
    std::vector<cldnn::BufferDescriptor> get_internal_buffer_descs(const cldnn::kernel_impl_params&) const override {
        return {};
    }

    void ensure_ready(cldnn::primitive_inst& instance) {
        std::lock_guard<std::mutex> g(mu_);
        if (megakernelRuntime_ != nullptr) {
            return;
        }

        megakernelRuntime_ = CreateMegaKernelPOCRuntime();

        auto& eng = cldnn::downcast<cldnn::ocl::ocl_engine>(instance.get_network().get_engine());
        cl_context ctx = eng.get_cl_context().get();
        cl_device_id dl_device = eng.get_cl_device().get();
        cl_command_queue queue = cldnn::downcast<cldnn::ocl::ocl_stream>(instance.get_network().get_stream()).get_cl_queue().get();

        auto params = mk::GetMegaKernelConstantParams(&instance);

        // Specific to OpenCL platform, but will have to be generalized to any plaform supported by megakernel runtime.
        mk::PlatformParamsImpl platformParams{};
        platformParams.context = ctx;
        platformParams.deviceId = dl_device;
        platformParams.stream = queue;
        mk::TErrorcode rc = megakernelRuntime_->Init(params.get(), &platformParams);
        OPENVINO_ASSERT(rc == 0, "[MegaKernel] runtime Init failed: ", rc);
    }

    cldnn::event::ptr execute(const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& instance) override {
        ensure_ready(instance);

        auto& strm = instance.get_network().get_stream();
        auto& ocls = downcast<ocl_stream>(strm);
        cl_command_queue q = ocls.get_cl_queue().get();

        for (auto& e : events)
            strm.wait_for_events({e});  // inputs ready before we read them

        auto params = mk::GetMegaKernelRuntimeParams(&instance);
        mk::TErrorcode rc = megakernelRuntime_->Execute(params.get());
        OPENVINO_ASSERT(rc == 0, "[MegaKernel] runtime Execute failed: ", rc);

        cl_event marker;
        clEnqueueMarkerWithWaitList(q, 0, nullptr, &marker);
        return std::make_shared<ocl_event>(cl::Event(marker, false), 0ULL);
    }

    ~MegaKernelFastImpl() override {
        if (megakernelRuntime_)
            DestroyMegaKernelPOCRuntime(megakernelRuntime_);
        megakernelRuntime_ = nullptr;
    }

private:
    std::mutex mu_;
    mk::IMegakernelRuntime* megakernelRuntime_ = nullptr;
};

}  // namespace

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------
std::unique_ptr<cldnn::primitive_impl> MegaKernelImpl::create_impl(const cldnn::program_node& node, const RuntimeParams& params) const {
    OPENVINO_ASSERT(node.is_type<cldnn::megakernel>());
    return std::make_unique<MegaKernelFastImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::megakernel)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::MegaKernelFastImpl)