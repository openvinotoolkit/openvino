// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// PrimitiveImplOCL keeps every stage a primitive may use and `_order`, the activated subset: sources, kernels and the
// model cache all follow `_order`, not the stage index. These tests pin that contract with a fake impl whose stages
// are activated out of order, without compiling any kernel.

#include <array>
#include <numeric>

#include "impls/ocl_v2/primitive_ocl_base.hpp"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/graph/serialization/utils.hpp"
#include "test_utils.h"

using namespace ov::intel_gpu;
using namespace ov::intel_gpu::ocl;

namespace {

constexpr size_t stage_count = 4;
constexpr size_t failing_stage = 1;

std::string entry_point_of(size_t stage) {
    return "fake_stage_" + std::to_string(stage);
}

class FakeKernel : public cldnn::kernel {
public:
    explicit FakeKernel(std::string id) : m_id(std::move(id)) {}
    std::shared_ptr<cldnn::kernel> clone(bool) const override {
        return std::make_shared<FakeKernel>(m_id);
    }
    bool is_same(const cldnn::kernel& other) const override {
        return m_id == other.get_id();
    }
    std::string get_id() const override {
        return m_id;
    }
    std::vector<uint8_t> get_binary() const override {
        return {};
    }
    std::string get_build_log() const override {
        return {};
    }

private:
    std::string m_id;
};

class FakeGenerator : public KernelGeneratorBase {
public:
    explicit FakeGenerator(size_t stage) : m_stage(stage) {}

    [[nodiscard]] KernelData get_kernel_data(const RuntimeParams&) const override {
        // A stage that cannot serve the given params is dropped from `_order`.
        OPENVINO_ASSERT(m_stage != failing_stage, "stage is not applicable");
        KernelData kd;
        kd.code = std::make_shared<KernelString>();
        kd.code->entry_point = entry_point_of(m_stage);
        kd.params.layerID = entry_point_of(m_stage);
        kd.params.workGroups.global = {m_stage + 1, 1, 1};
        kd.params.workGroups.local = {1, 1, 1};
        return kd;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams&, KernelData&, ImplRuntimeParams*) {}};
    }

private:
    size_t m_stage;
};

class FakeImpl : public PrimitiveImplOCL {
public:
    std::array<Stage::Ptr, stage_count> stages;

    FakeImpl() : PrimitiveImplOCL("fake_impl") {
        for (size_t i = 0; i < stage_count; i++) {
            stages[i] = make_stage<FakeGenerator>(i);
        }
    }

    explicit FakeImpl(const std::vector<size_t>& activation) : FakeImpl() {
        RuntimeParams params;
        for (const auto i : activation) {
            add_stage(stages[i], params);
        }
    }

    const std::string& get_type_info() const override {
        static const std::string type = "fake_ocl_v2_impl";
        return type;
    }

    std::unique_ptr<cldnn::primitive_impl> clone() const override {
        return make_deep_copy(this);
    }
};

cldnn::kernels_cache::compiled_kernels compiled(const std::vector<std::pair<std::string, size_t>>& kernels) {
    cldnn::kernels_cache::compiled_kernels result;
    auto& vec = result[cldnn::kernel_impl_params{}];
    for (const auto& [id, idx] : kernels) {
        vec.emplace_back(std::make_shared<FakeKernel>(id), idx);
    }
    return result;
}

// Activations that leave `_order` different from the stage indices; stage `failing_stage` never activates.
const std::vector<std::vector<size_t>> activations = {{3, 1, 0}, {2, 0}, {3, 2, 1, 0}, {2}, {3}};

std::vector<size_t> expected_order(const std::vector<size_t>& activation) {
    std::vector<size_t> order;
    std::copy_if(activation.begin(), activation.end(), std::back_inserter(order), [](size_t i) {
        return i != failing_stage;
    });
    return order;
}

}  // namespace

TEST(ocl_v2_impl_base, add_stage_skips_stage_without_kernel_data) {
    for (const auto& activation : activations) {
        FakeImpl impl(activation);
        EXPECT_EQ(impl._order, expected_order(activation));
    }
}

TEST(ocl_v2_impl_base, kernels_source_follows_activation_order) {
    for (const auto& activation : activations) {
        FakeImpl impl(activation);
        const auto sources = impl.get_kernels_source();
        const auto order = expected_order(activation);
        ASSERT_EQ(sources.size(), order.size());
        for (size_t i = 0; i < order.size(); i++) {
            EXPECT_EQ(sources[i]->entry_point, entry_point_of(order[i]));
        }
    }
}

// set_kernels() gets the position of each kernel within get_kernels_source(), not a stage index.
TEST(ocl_v2_impl_base, set_kernels_binds_kernel_to_the_stage_it_was_built_from) {
    for (const auto& activation : activations) {
        FakeImpl impl(activation);
        const auto order = expected_order(activation);

        std::vector<std::pair<std::string, size_t>> kernels;
        for (size_t i = 0; i < order.size(); i++) {
            kernels.emplace_back(entry_point_of(order[i]), i);
        }
        // The batch build may hand the kernels back in any order.
        std::reverse(kernels.begin(), kernels.end());
        impl.set_kernels(compiled(kernels));

        for (size_t stage = 0; stage < stage_count; stage++) {
            const bool active = std::find(order.begin(), order.end(), stage) != order.end();
            if (active) {
                ASSERT_NE(impl.stages[stage]->kernel, nullptr) << "stage " << stage << " got no kernel";
                EXPECT_EQ(impl.stages[stage]->kernel->get_id(), entry_point_of(stage));
            } else {
                EXPECT_EQ(impl.stages[stage]->kernel, nullptr) << "inactive stage " << stage << " got a kernel";
            }
        }

        const auto bound = impl.get_kernels();
        ASSERT_EQ(bound.size(), order.size());
        for (size_t i = 0; i < order.size(); i++) {
            EXPECT_EQ(bound[i]->get_id(), entry_point_of(order[i]));
        }
    }
}

TEST(ocl_v2_impl_base, save_load_restores_activation_order_and_stage_data) {
    for (const auto& activation : activations) {
        FakeImpl original(activation);
        const auto order = expected_order(activation);

        cldnn::membuf mem_buf;
        {
            std::ostream out_mem(&mem_buf);
            cldnn::BinaryOutputBuffer ob(out_mem);
            original.save(ob);
        }

        FakeImpl restored;
        {
            std::istream in_mem(&mem_buf);
            cldnn::BinaryInputBuffer ib(in_mem, ::tests::get_test_engine());
            restored.load(ib);
        }

        ASSERT_EQ(restored._order, order);
        for (const auto stage : order) {
            const auto& kd = restored.stages[stage]->kd;
            EXPECT_EQ(kd.params.layerID, entry_point_of(stage));
            EXPECT_EQ(kd.params.workGroups.global[0], stage + 1);
            // The dispatch function is not serialized; it has to come back from the generator.
            EXPECT_TRUE(kd.update_dispatch_data_func.m_dispatch_data_func != nullptr);
        }
    }
}

TEST(ocl_v2_impl_base, deep_copy_keeps_activation_order_and_kernels) {
    FakeImpl impl({3, 1, 0});
    impl.set_kernels(compiled({{entry_point_of(3), 0}, {entry_point_of(0), 1}}));

    auto copy = impl.clone();
    const auto kernels = copy->get_kernels();
    ASSERT_EQ(kernels.size(), 2u);
    EXPECT_EQ(kernels[0]->get_id(), entry_point_of(3));
    EXPECT_EQ(kernels[1]->get_id(), entry_point_of(0));
}
