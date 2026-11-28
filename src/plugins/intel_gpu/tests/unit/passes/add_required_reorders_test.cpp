// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/internal_properties.hpp"
#include "random_generator.hpp"
#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "concatenation_inst.h"
#include "crop_inst.h"
#include "data_inst.h"
#include "eltwise_inst.h"
#include "fully_connected_inst.h"
#include "gather_inst.h"
#include "gather_nd_inst.h"
#include "pass_manager.h"
#include "permute_inst.h"
#include "reshape_inst.h"
#include "shape_of_inst.h"
#include "convolution_inst.h"
#include "dft_inst.h"
#include "mvn_inst.h"
#include "to_string_utils.h"
#include "kernel_base.h"
#include "kernel_selector.h"
#include "intel_gpu/primitives/scatter_elements_update.hpp"
#include "intel_gpu/primitives/scatter_update.hpp"
#include "intel_gpu/primitives/rms.hpp"
#include "registry/implementation_manager.hpp"

#include "program_wrapper.h"

#include <memory>
#include <stdexcept>

using namespace cldnn;
using namespace ::tests;

namespace {

void test_bf16_scatter_fallback(bool dynamic, bool caching, bool duplicate_users = false,
                              data_types output_type = data_types::bf16) {
    auto& engine = get_test_engine();
    const auto shape = dynamic ? ov::PartialShape{1, 1, 1, ov::Dimension::dynamic()} : ov::PartialShape{1, 1, 1, 4};
    const layout data_layout{shape, data_types::bf16, format::bfyx};
    const layout indices_layout{shape, data_types::i32, format::bfyx};
    auto scatter = std::make_shared<scatter_elements_update>("scatter", input_info("input"), input_info("indices"),
                                                             input_info("updates"), 3);
    scatter->output_data_types[0] = output_type;
    topology topology;
    topology.add(input_layout("input", data_layout));
    topology.add(input_layout("indices", indices_layout));
    topology.add(input_layout("updates", data_layout));
    topology.add_primitive(scatter);
    if (duplicate_users)
        topology.add(eltwise("product", {input_info("scatter"), input_info("scatter")}, eltwise_mode::prod, data_types::bf16));
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    if (duplicate_users)
        config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"scatter", "product"}));
    auto network = get_network(engine, topology, config, get_test_stream_ptr(), caching);
    for (const size_t count : (dynamic ? std::vector<size_t>{4, 2, 4} : std::vector<size_t>{4})) {
        const ov::PartialShape runtime_shape{1, 1, 1, static_cast<int64_t>(count)};
        auto input = engine.allocate_memory({runtime_shape, data_types::bf16, format::bfyx});
        auto updates = engine.allocate_memory({runtime_shape, data_types::bf16, format::bfyx});
        auto indices = engine.allocate_memory({runtime_shape, data_types::i32, format::bfyx});
        set_values<uint16_t>(input, std::vector<uint16_t>(count, 0x3f80));
        set_values<uint16_t>(updates, std::vector<uint16_t>(count, 0x4000));
        std::vector<int32_t> index_values(count);
        std::iota(index_values.begin(), index_values.end(), 0);
        set_values<int32_t>(indices, index_values);
        network->set_input_data("input", input);
        network->set_input_data("indices", indices);
        network->set_input_data("updates", updates);
        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), duplicate_users ? 2 : 1);
        auto output = outputs.at("scatter").get_memory();
        ASSERT_EQ(output->get_layout().data_type, output_type);
        if (output_type == data_types::bf16) {
            mem_lock<uint16_t, mem_lock_type::read> values(output, get_test_stream());
            for (size_t idx = 0; idx < count; ++idx)
                EXPECT_EQ(values[idx], 0x4000);
        } else {
            mem_lock<float, mem_lock_type::read> values(output, get_test_stream());
            for (size_t idx = 0; idx < count; ++idx)
                EXPECT_EQ(values[idx], 2.f);
        }
        mem_lock<uint16_t, mem_lock_type::read> original_values(input, get_test_stream());
        EXPECT_EQ(original_values[0], 0x3f80);
    }
    EXPECT_EQ(scatter->id, "scatter");
    EXPECT_EQ(scatter->output_data_types[0].value(), output_type);
    const auto& output_node = network->get_program()->get_node("scatter");
    if (output_type == data_types::bf16) {
        ASSERT_TRUE(output_node.is_type<reorder>());
    }
    const auto& compute_node = output_type == data_types::bf16 ? output_node.get_dependency(0) : output_node;
    ASSERT_TRUE(compute_node.is_type<scatter_elements_update>());
    EXPECT_EQ(compute_node.get_input_layout(0).data_type, data_types::f32);
    EXPECT_EQ(compute_node.get_input_layout(1).data_type, data_types::i32);
    EXPECT_EQ(compute_node.get_input_layout(2).data_type, data_types::f32);
    EXPECT_EQ(compute_node.get_output_layout().data_type, data_types::f32);
    EXPECT_EQ(network->get_program()->get_node("input").get_output_layout().data_type, data_types::bf16);
    if (duplicate_users) {
        const auto& product = network->get_program()->get_node("product");
        ASSERT_EQ(product.get_dependencies().size(), 2);
        EXPECT_EQ(product.get_dependency_with_port(0).first, &output_node);
        EXPECT_EQ(product.get_dependency_with_port(1).first, &output_node);
        EXPECT_EQ(product.get_dependency_with_port(0).second, 0);
        EXPECT_EQ(product.get_dependency_with_port(1).second, 0);
        EXPECT_TRUE(output_node.is_output());
    }
}

void test_bf16_scatter_update_fallback(bool dynamic, bool caching) {
    auto& engine = get_test_engine();
    const auto count_dim = dynamic ? ov::Dimension::dynamic() : ov::Dimension(4);
    const layout data_layout{ov::PartialShape{1, 1, 1, count_dim}, data_types::bf16, format::bfyx};
    const layout indices_layout{ov::PartialShape{1, 1, 1, 2}, data_types::i32, format::bfyx};
    const layout updates_layout{ov::PartialShape{1, 1, 1, 2}, data_types::bf16, format::bfyx};
    auto scatter = std::make_shared<scatter_update>("scatter_update", input_info("input"), input_info("indices"),
                                                   input_info("updates"), 3);
    scatter->output_data_types[0] = data_types::bf16;
    topology topology;
    topology.add(input_layout("input", data_layout));
    topology.add(input_layout("indices", indices_layout));
    topology.add(input_layout("updates", updates_layout));
    topology.add_primitive(scatter);
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto network = get_network(engine, topology, config, get_test_stream_ptr(), caching);
    for (const size_t count : (dynamic ? std::vector<size_t>{4, 3, 4} : std::vector<size_t>{4})) {
        auto input = engine.allocate_memory({ov::PartialShape{1, 1, 1, static_cast<int64_t>(count)}, data_types::bf16, format::bfyx});
        auto indices = engine.allocate_memory({ov::PartialShape{1, 1, 1, 2}, data_types::i32, format::bfyx});
        auto updates = engine.allocate_memory({ov::PartialShape{1, 1, 1, 2}, data_types::bf16, format::bfyx});
        set_values<uint16_t>(input, std::vector<uint16_t>(count, 0x3f80));
        set_values<int32_t>(indices, std::vector<int32_t>{1, static_cast<int32_t>(count - 1)});
        set_values<uint16_t>(updates, std::vector<uint16_t>{0x4000, 0x4040});
        network->set_input_data("input", input);
        network->set_input_data("indices", indices);
        network->set_input_data("updates", updates);
        auto outputs = network->execute();
        auto output = outputs.at("scatter_update").get_memory();
        ASSERT_EQ(output->get_layout().data_type, data_types::bf16);
        mem_lock<uint16_t, mem_lock_type::read> values(output, get_test_stream());
        EXPECT_EQ(values[1], 0x4000);
        EXPECT_EQ(values[count - 1], 0x4040);
        for (size_t index = 0; index < count; ++index) {
            if (index != 1 && index != count - 1) {
                EXPECT_EQ(values[index], 0x3f80);
            }
        }
    }
    const auto& output_node = network->get_program()->get_node("scatter_update");
    ASSERT_TRUE(output_node.is_type<reorder>());
    const auto& compute_node = output_node.get_dependency(0);
    ASSERT_TRUE(compute_node.is_type<scatter_update>());
    EXPECT_EQ(compute_node.get_input_layout(0).data_type, data_types::f32);
    EXPECT_EQ(compute_node.get_input_layout(1).data_type, data_types::i32);
    EXPECT_EQ(compute_node.get_input_layout(2).data_type, data_types::f32);
    EXPECT_EQ(compute_node.get_output_layout().data_type, data_types::f32);
    const auto params = compute_node.get_kernel_impl_params();
    const auto shape_type = ImplementationManager::get_shape_type(*params);
    bool checked_ocl_manager = false;
    for (const auto& manager : compute_node.type()->get_all_implementations()) {
        if (manager->get_impl_type() != impl_types::ocl || (manager->get_shape_type() & shape_type) != shape_type) {
            continue;
        }
        EXPECT_EQ(manager->query_data_types(*params), implementation_data_type_support::supported);
        checked_ocl_manager = true;
    }
    EXPECT_TRUE(checked_ocl_manager);
}

void test_bf16_gather_nd_fallback(bool dynamic, bool caching, data_types output_type = data_types::bf16) {
    auto& engine = get_test_engine();
    const auto count_dim = dynamic ? ov::Dimension::dynamic() : ov::Dimension(4);
    const layout data_layout{ov::PartialShape{1, 1, 1, count_dim}, data_types::bf16, format::bfyx};
    const layout indices_layout{ov::PartialShape{count_dim, 4}, data_types::i32, format::bfyx};
    auto gather = std::make_shared<gather_nd>("gather_nd", input_info("input"), input_info("indices"), 4, 2, 0, false);
    gather->output_data_types[0] = data_types::bf16;
    topology topology;
    topology.add(input_layout("input", data_layout));
    topology.add(input_layout("indices", indices_layout));
    topology.add_primitive(gather);
    const auto output_id = output_type == data_types::bf16 ? "gather_nd" : "gather_output";
    if (output_type != data_types::bf16) {
        topology.add(reorder(output_id, input_info("gather_nd"),
                             layout{ov::PartialShape{count_dim}, output_type, format::bfyx}));
    }
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto network = get_network(engine, topology, config, get_test_stream_ptr(), caching);
    const std::vector<uint16_t> input_values{0x3f80, 0x4000, 0x4040, 0x4080};
    for (const size_t count : (dynamic ? std::vector<size_t>{4, 2, 4} : std::vector<size_t>{4})) {
        auto input = engine.allocate_memory({ov::PartialShape{1, 1, 1, static_cast<int64_t>(count)}, data_types::bf16, format::bfyx});
        auto indices = engine.allocate_memory({ov::PartialShape{static_cast<int64_t>(count), 4}, data_types::i32, format::bfyx});
        set_values<uint16_t>(input, std::vector<uint16_t>(input_values.begin(), input_values.begin() + count));
        std::vector<int32_t> index_values(count * 4, 0);
        for (size_t index = 0; index < count; ++index) {
            index_values[index * 4 + 3] = static_cast<int32_t>(count - index - 1);
        }
        set_values<int32_t>(indices, index_values);
        network->set_input_data("input", input);
        network->set_input_data("indices", indices);
        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), 1);
        auto output = outputs.at(output_id).get_memory();
        ASSERT_EQ(output->get_layout().data_type, output_type);
        EXPECT_EQ(output->get_layout().get_partial_shape(), ov::PartialShape{static_cast<int64_t>(count)});
        if (output_type == data_types::bf16) {
            mem_lock<uint16_t, mem_lock_type::read> values(output, get_test_stream());
            for (size_t index = 0; index < count; ++index) {
                EXPECT_EQ(values[index], input_values[count - index - 1]);
            }
        } else {
            mem_lock<float, mem_lock_type::read> values(output, get_test_stream());
            for (size_t index = 0; index < count; ++index) {
                EXPECT_EQ(values[index], static_cast<float>(count - index));
            }
        }
        mem_lock<int32_t, mem_lock_type::read> original_indices(indices, get_test_stream());
        for (size_t index = 0; index < index_values.size(); ++index) {
            EXPECT_EQ(original_indices[index], index_values[index]);
        }
    }
    const auto& processing_order = network->get_program()->get_processing_order();
    const auto compute = std::find_if(processing_order.begin(), processing_order.end(), [](const auto* node) {
        return node->template is_type<gather_nd>();
    });
    ASSERT_NE(compute, processing_order.end());
    const auto& compute_node = **compute;
    EXPECT_EQ(compute_node.get_input_layout(0).data_type, data_types::f32);
    EXPECT_EQ(compute_node.get_input_layout(1).data_type, data_types::i32);
    EXPECT_EQ(compute_node.get_output_layout().data_type, data_types::f32);
    EXPECT_EQ(network->get_program()->get_node("input").get_output_layout().data_type, data_types::bf16);
    EXPECT_EQ(network->get_program()->get_node("indices").get_output_layout().data_type, data_types::i32);
    EXPECT_EQ(gather->id, "gather_nd");
    EXPECT_EQ(gather->output_data_types[0].value(), data_types::bf16);
    const auto& managers = compute_node.type()->get_all_implementations();
    ASSERT_FALSE(managers.empty());
    auto bf16_params = *compute_node.get_kernel_impl_params();
    bf16_params.input_layouts[0].data_type = data_types::bf16;
    bf16_params.output_layouts[0].data_type = output_type;
    for (const auto& manager : managers) {
        EXPECT_EQ(manager->query_data_types(bf16_params), implementation_data_type_support::unsupported);
        EXPECT_EQ(manager->query_data_types(*compute_node.get_kernel_impl_params()), implementation_data_type_support::supported);
    }
}

}  // namespace

TEST(BF16Fallback, ScatterStatic) { test_bf16_scatter_fallback(false, false); }
TEST(BF16Fallback, ScatterDynamicShapeChanges) { test_bf16_scatter_fallback(true, false); }
TEST(BF16Fallback, ScatterStaticCache) { test_bf16_scatter_fallback(false, true); }
TEST(BF16Fallback, ScatterDynamicCache) { test_bf16_scatter_fallback(true, true); }
TEST(BF16Fallback, ScatterPublicOutputAndDuplicateUsers) { test_bf16_scatter_fallback(false, false, true); }
TEST(BF16Fallback, ScatterPublicFP32Output) { test_bf16_scatter_fallback(false, false, false, data_types::f32); }
TEST(BF16Fallback, ScatterUpdateStatic) { test_bf16_scatter_update_fallback(false, false); }
TEST(BF16Fallback, ScatterUpdateDynamicShapeChanges) { test_bf16_scatter_update_fallback(true, false); }
TEST(BF16Fallback, ScatterUpdateStaticCache) { test_bf16_scatter_update_fallback(false, true); }
TEST(BF16Fallback, ScatterUpdateDynamicCache) { test_bf16_scatter_update_fallback(true, true); }

TEST(BF16Fallback, GatherNDStatic) { test_bf16_gather_nd_fallback(false, false); }
TEST(BF16Fallback, GatherNDDynamicShapeChanges) { test_bf16_gather_nd_fallback(true, false); }
TEST(BF16Fallback, GatherNDStaticCache) { test_bf16_gather_nd_fallback(false, true); }
TEST(BF16Fallback, GatherNDDynamicCache) { test_bf16_gather_nd_fallback(true, true); }
TEST(BF16Fallback, GatherNDPublicFP32Output) { test_bf16_gather_nd_fallback(false, false, data_types::f32); }

TEST(BF16Fallback, NativeBF16EltwisePreserved) {
    auto& engine = get_test_engine();
    const layout input_layout_value{ov::PartialShape{1, 1, 1, 4}, data_types::bf16, format::bfyx};
    auto input = engine.allocate_memory(input_layout_value);
    set_values<uint16_t>(input, {0x3f80, 0x4000, 0x4040, 0x4080});
    topology topology;
    topology.add(input_layout("input", input_layout_value));
    topology.add(eltwise("product", {input_info("input"), input_info("input")}, eltwise_mode::prod, data_types::bf16));
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();
    const auto& node = network.get_program()->get_node("product");
    ASSERT_TRUE(node.is_type<eltwise>());
    EXPECT_EQ(node.get_input_layout(0).data_type, data_types::bf16);
    EXPECT_EQ(node.get_output_layout().data_type, data_types::bf16);
    bool native_support_found = false;
    for (const auto& manager : node.type()->get_all_implementations()) {
        native_support_found |= manager->query_data_types(*node.get_kernel_impl_params()) == implementation_data_type_support::supported;
    }
    EXPECT_TRUE(native_support_found);
    auto output = outputs.at("product").get_memory();
    mem_lock<uint16_t, mem_lock_type::read> values(output, get_test_stream());
    EXPECT_EQ(values[0], 0x3f80);
    EXPECT_EQ(values[1], 0x4080);
}

TEST(BF16Fallback, NativeBF16RMSPreserved) {
    auto& engine = get_test_engine();
    const layout input_layout_value{ov::PartialShape{1, 1, 1, 4}, data_types::bf16, format::bfyx};
    auto input = engine.allocate_memory(input_layout_value);
    auto gamma = engine.allocate_memory(input_layout_value);
    set_values<uint16_t>(input, {0x3f80, 0x3f80, 0x3f80, 0x3f80});
    set_values<uint16_t>(gamma, {0x3f80, 0x3f80, 0x3f80, 0x3f80});
    topology topology;
    topology.add(input_layout("input", input_layout_value));
    topology.add(input_layout("gamma", input_layout_value));
    topology.add(rms("rms", input_info("input"), input_info("gamma"), 1e-5f));
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    network.set_input_data("gamma", gamma);
    auto outputs = network.execute();
    const auto& node = network.get_program()->get_node("rms");
    ASSERT_TRUE(node.is_type<rms>());
    EXPECT_EQ(node.get_input_layout(0).data_type, data_types::bf16);
    EXPECT_EQ(node.get_output_layout().data_type, data_types::bf16);
    auto output = outputs.at("rms").get_memory();
    mem_lock<uint16_t, mem_lock_type::read> values(output, get_test_stream());
    EXPECT_EQ(values[0], 0x3f80);
}

TEST(BF16Fallback, MissingForcedKernelRemainsAnError) {
    auto& engine = get_test_engine();
    const layout data_layout{ov::PartialShape{1, 1, 1, 4}, data_types::bf16, format::bfyx};
    const layout indices_layout{ov::PartialShape{1, 1, 1, 4}, data_types::i32, format::bfyx};
    topology topology;
    topology.add(input_layout("input", data_layout));
    topology.add(input_layout("indices", indices_layout));
    topology.add(input_layout("updates", data_layout));
    topology.add(scatter_elements_update("scatter", input_info("input"), input_info("indices"), input_info("updates"), 3));
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
        {"scatter", {format::bfyx, "missing_bf16_kernel", impl_types::ocl}}
    }));
    EXPECT_ANY_THROW(network(engine, topology, config));
}

TEST(BF16Fallback, WrongForcedBackendRemainsAnError) {
    auto& engine = get_test_engine();
    const layout data_layout{ov::PartialShape{1, 1, 1, 4}, data_types::bf16, format::bfyx};
    const layout indices_layout{ov::PartialShape{1, 1, 1, 4}, data_types::i32, format::bfyx};
    topology topology;
    topology.add(input_layout("input", data_layout));
    topology.add(input_layout("indices", indices_layout));
    topology.add(input_layout("updates", data_layout));
    topology.add(scatter_elements_update("scatter", input_info("input"), input_info("indices"), input_info("updates"), 3));
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
        {"scatter", {format::bfyx, "", impl_types::onednn}}
    }));
    EXPECT_ANY_THROW(network(engine, topology, config));
}

TEST(BF16Fallback, BF16ConstantInputExcluded) {
    auto& engine = get_test_engine();
    const layout data_layout{ov::PartialShape{1, 1, 1, 4}, data_types::bf16, format::bfyx};
    const layout indices_layout{ov::PartialShape{1, 1, 1, 4}, data_types::i32, format::bfyx};
    auto updates = engine.allocate_memory(data_layout);
    set_values<uint16_t>(updates, {0x4000, 0x4000, 0x4000, 0x4000});
    topology topology;
    topology.add(input_layout("input", data_layout));
    topology.add(input_layout("indices", indices_layout));
    topology.add(data("updates", updates));
    topology.add(scatter_elements_update("scatter", input_info("input"), input_info("indices"), input_info("updates"), 3));
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    EXPECT_ANY_THROW(network(engine, topology, config));
    EXPECT_EQ(updates->get_layout().data_type, data_types::bf16);
}

namespace {

struct PrecisionQueryParams : kernel_selector::Params {
    PrecisionQueryParams() : Params(kernel_selector::KernelType::ELTWISE, "precision_query") {}
    kernel_selector::ParamsKey GetParamsKey() const override { return required_key; }
    kernel_selector::ParamsKey required_key;
};

class FP32PrecisionQueryKernel : public kernel_selector::KernelBase {
public:
    FP32PrecisionQueryKernel() : KernelBase("fp32_precision_query") {}
    kernel_selector::ParamsKey GetSupportedKey() const override {
        kernel_selector::ParamsKey supported;
        supported.EnableInputDataType(kernel_selector::Datatype::F32);
        supported.EnableOutputDataType(kernel_selector::Datatype::F32);
        supported.EnableInputLayout(kernel_selector::DataLayout::bfyx);
        supported.EnableOutputLayout(kernel_selector::DataLayout::bfyx);
        return supported;
    }
    kernel_selector::KernelsData GetKernelsData(const kernel_selector::Params&) const override {
        OPENVINO_THROW("Precision queries must not generate kernels");
    }
};

class BF16PrecisionQueryKernel : public FP32PrecisionQueryKernel {
public:
    std::string GetName() const override { return "bf16_precision_query"; }
    kernel_selector::ParamsKey GetSupportedKey() const override {
        kernel_selector::ParamsKey supported;
        supported.EnableInputDataType(kernel_selector::Datatype::BF16);
        supported.EnableOutputDataType(kernel_selector::Datatype::BF16);
        supported.EnableInputLayout(kernel_selector::DataLayout::bfyx);
        supported.EnableOutputLayout(kernel_selector::DataLayout::bfyx);
        return supported;
    }
};

class FP16PrecisionQueryKernel : public FP32PrecisionQueryKernel {
public:
    std::string GetName() const override { return "fp16_precision_query"; }
    kernel_selector::ParamsKey GetSupportedKey() const override {
        kernel_selector::ParamsKey supported;
        supported.EnableInputDataType(kernel_selector::Datatype::F16);
        supported.EnableOutputDataType(kernel_selector::Datatype::F16);
        return supported;
    }
};

class PrecisionQuerySelector : public kernel_selector::kernel_selector_base {
public:
    explicit PrecisionQuerySelector(bool with_bf16 = false, bool fp16_only = false) {
        if (fp16_only)
            Attach<FP16PrecisionQueryKernel>();
        else
            Attach<FP32PrecisionQueryKernel>();
        if (with_bf16) {
            Attach<BF16PrecisionQueryKernel>();
        }
    }
protected:
    kernel_selector::KernelsData GetBestKernels(const kernel_selector::Params&) const override {
        OPENVINO_THROW("Precision queries must not select kernels");
    }
};

}  // namespace

TEST(KernelPrecisionSupport, SeparatesTypesFromLayoutAndDynamicRestrictions) {
    FP32PrecisionQueryKernel implementation;
    auto supported = implementation.GetSupportedKey();
    kernel_selector::ParamsKey required;
    required.EnableInputDataType(kernel_selector::Datatype::F32);
    required.EnableOutputDataType(kernel_selector::Datatype::F32);
    required.EnableInputLayout(kernel_selector::DataLayout::bfzyx);
    required.EnableOutputLayout(kernel_selector::DataLayout::bfzyx);
    required.EnableDynamicShapesSupport();
    EXPECT_TRUE(supported.SupportDataTypes(required));
    EXPECT_FALSE(supported.Support(required));
}

TEST(KernelPrecisionSupport, ChecksEveryDataAndWeightsRole) {
    kernel_selector::ParamsKey supported;
    supported.EnableInputDataType(kernel_selector::Datatype::F32);
    supported.EnableOutputDataType(kernel_selector::Datatype::F32);
    supported.EnableInputWeightsType(kernel_selector::WeightsType::F16);
    supported.EnableOutputWeightsType(kernel_selector::WeightsType::F16);
    for (const size_t role : {size_t{0}, size_t{1}, size_t{2}, size_t{3}}) {
        kernel_selector::ParamsKey required;
        if (role == 0) required.EnableInputDataType(kernel_selector::Datatype::BF16);
        if (role == 1) required.EnableOutputDataType(kernel_selector::Datatype::BF16);
        if (role == 2) required.EnableInputWeightsType(kernel_selector::WeightsType::BF16);
        if (role == 3) required.EnableOutputWeightsType(kernel_selector::WeightsType::BF16);
        EXPECT_FALSE(supported.SupportDataTypes(required));
    }
}

TEST(KernelPrecisionSupport, DistinguishesMissingTypesFromMissingForcedKernel) {
    PrecisionQueryParams params;
    params.required_key.EnableInputDataType(kernel_selector::Datatype::BF16);
    params.required_key.EnableOutputDataType(kernel_selector::Datatype::BF16);
    PrecisionQuerySelector fp32_selector;
    PrecisionQuerySelector bf16_selector(true);
    EXPECT_EQ(fp32_selector.GetDataTypeSupport(params), kernel_selector::DataTypeSupport::UNSUPPORTED);
    EXPECT_EQ(bf16_selector.GetDataTypeSupport(params), kernel_selector::DataTypeSupport::SUPPORTED);
    params.forceImplementation = "fp32_precision_query";
    EXPECT_EQ(bf16_selector.GetDataTypeSupport(params), kernel_selector::DataTypeSupport::UNSUPPORTED);
    params.forceImplementation = "missing_precision_query";
    EXPECT_EQ(bf16_selector.GetDataTypeSupport(params), kernel_selector::DataTypeSupport::UNKNOWN);
    params.forceImplementation.clear();
    params.required_key.EnableDynamicShapesSupport();
    EXPECT_EQ(bf16_selector.GetDataTypeSupport(params), kernel_selector::DataTypeSupport::SUPPORTED);
}

TEST(KernelPrecisionSupport, NoFP32Alternative) {
    PrecisionQuerySelector selector(false, true);
    PrecisionQueryParams params;
    params.required_key.EnableInputDataType(kernel_selector::Datatype::BF16);
    params.required_key.EnableOutputDataType(kernel_selector::Datatype::BF16);
    EXPECT_EQ(selector.GetDataTypeSupport(params), kernel_selector::DataTypeSupport::UNSUPPORTED);
    params.required_key = {};
    params.required_key.EnableInputDataType(kernel_selector::Datatype::F32);
    params.required_key.EnableOutputDataType(kernel_selector::Datatype::F32);
    EXPECT_EQ(selector.GetDataTypeSupport(params), kernel_selector::DataTypeSupport::UNSUPPORTED);
}

TEST(KernelPrecisionSupport, BF16EligibleDespiteInvalidLayout) {
    PrecisionQuerySelector selector(true);
    PrecisionQueryParams params;
    params.required_key.EnableInputDataType(kernel_selector::Datatype::BF16);
    params.required_key.EnableOutputDataType(kernel_selector::Datatype::BF16);
    params.required_key.EnableInputLayout(kernel_selector::DataLayout::bfzyx);
    params.required_key.EnableOutputLayout(kernel_selector::DataLayout::bfzyx);
    BF16PrecisionQueryKernel kernel;
    EXPECT_FALSE(kernel.GetSupportedKey().Support(params.required_key));
    EXPECT_EQ(selector.GetDataTypeSupport(params), kernel_selector::DataTypeSupport::SUPPORTED);
}

TEST(add_required_reorders, input_reorder_inside_shape_of_subgraph) {
    auto& engine = get_test_engine();
    auto input_layout_dynamic = layout{ov::PartialShape{1, 32, ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                       data_types::f16, format::bfyx};
    auto input = engine.allocate_memory({ov::PartialShape{1, 32, 32, 32}, data_types::f16, format::bfyx});
    auto data_0 = engine.allocate_memory({ ov::PartialShape{}, data_types::i32, format::bfyx });
    auto data_1 = engine.allocate_memory({ ov::PartialShape{}, data_types::f32, format::bfyx });

    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(data("data_0", data_0));
    topology.add(data("data_1", data_1));
    topology.add(shape_of("shape_of", input_info("input"), data_types::i32));
    topology.add(gather("gather0", input_info("shape_of"), input_info("data_0"), 0, {}, {}, 0, true));
    topology.add(eltwise("eltwise0", {input_info("gather0"), input_info("data_1")}, eltwise_mode::prod, data_types::f32));
    topology.add(reshape("reshape0", input_info("eltwise0"), false, {},
                         ov::PartialShape{1}, reshape::reshape_mode::unsqueeze));
    topology.add(gather("gather1", input_info("shape_of"), input_info("data_0"), 0, {}, {}, 0, true));
    topology.add(eltwise("eltwise1", {input_info("gather1"), input_info("data_1")}, eltwise_mode::prod, data_types::f32));
    topology.add(reshape("reshape1", input_info("eltwise1"), false, {},
                         ov::PartialShape{1}, reshape::reshape_mode::unsqueeze));
    topology.add(concatenation("concat0", {input_info("reshape0"), input_info("reshape1")}, 0, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    network.execute();

    auto prog = network.get_program();
    ASSERT_NE(prog, nullptr);

    auto& eltwise_node = prog->get_node("eltwise0");
    auto eltwise_in_layout = eltwise_node.get_input_layout();

    ASSERT_EQ(eltwise_in_layout.data_type, data_types::f32);
}

TEST(add_required_reorders, prevent_input_dt_changing_for_convs) {
    auto& engine = get_test_engine();

    int input_b = 1, input_f = 16, input_y = 3, input_x = 3;
    int output_b = input_b, output_f = 16, output_y = 6, output_x = 6;

    auto input_mem = engine.allocate_memory({ {input_b, input_f, input_y, input_x}, data_types::u8, format::bs_fs_yx_bsv16_fsv32 });
    auto input2_mem = engine.allocate_memory({ {input_b, input_f, input_y, input_x}, data_types::u8, format::bs_fs_yx_bsv16_fsv32 });
    auto weights_mem = engine.allocate_memory({ {16, 16, 1, 1}, data_types::i8, format::bfyx });

    auto input = input_layout("input", input_mem->get_layout());
    auto input_const = data("input_const", input2_mem);
    auto weights = data("weights", weights_mem);
    auto eltwise1 = eltwise("eltwise1", input_info("input"), input_info("input_const"), eltwise_mode::sum);
    auto conv1 = convolution("conv1", input_info("eltwise1"), "weights", "", 1, { 1, 1 }, { 1, 1 }, { 1, 1 }, { 2, 2 }, false);
    auto output_reorder = reorder("reorder", input_info("conv1"), { data_types::f32, format::bfyx, { output_b, output_f, output_y, output_x } });

    topology topology_test(input, input_const, eltwise1, weights, conv1, output_reorder);

    ExecutionConfig config_test = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc conv1_impl_test = { format::bfyx, "", impl_types::ocl };
    config_test.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv1", conv1_impl_test } }));

    auto prog = program::build_program(engine, topology_test, config_test, false, true);
    program_wrapper::apply_opt_pass<add_required_reorders>(*prog);

    ASSERT_NE(prog, nullptr);
    ASSERT_TRUE(prog->has_node("conv1"));
    ASSERT_EQ(prog->get_node("conv1").get_input_layout(0).data_type, data_types::u8);
}

TEST(add_required_reorders, prevent_users_invalidation) {
    auto& engine = get_test_engine();

    // Create padded input memory
    auto input_mem_padded = engine.allocate_memory({ {1, 16, 8, 8}, data_types::f16, format::bfyx, padding{{0, 0, 2, 2}, 0} });
    auto weights_mem = engine.allocate_memory({ {16, 16, 1, 1}, data_types::f16, format::bfyx });

    auto input = input_layout("input", input_mem_padded->get_layout());
    auto weights = data("weights", weights_mem);
    auto conv = convolution("conv", input_info("input"), "weights", "", 1, { 1, 1 }, { 1, 1 }, { 1, 1 }, { 2, 2 }, false);

    auto prog = program::build_program(engine,
                                       topology(input, weights, conv),
                                       get_test_default_config(engine),
                                       false,
                                       true);

    ASSERT_NE(prog, nullptr);
    ASSERT_TRUE(prog->has_node("conv"));

    const auto& conv_node = prog->get_node("conv");

    // Force OneDNN impl type to insert padded_layout -> non_padded_layout reorder
    prog->get_node("conv").set_forced_impl_type(impl_types::onednn);

    program_wrapper::apply_opt_pass<add_required_reorders>(*prog);

    const auto& conv_input = conv_node.get_dependency(0);

    ASSERT_TRUE(conv_input.is_type<reorder>());
    ASSERT_TRUE(conv_input.is_valid_output_layout());
}

TEST(add_required_reorders, skip_adding_reorder_batch_axis_padding) {
    auto& engine = get_test_engine();

    auto in_input = engine.allocate_memory({ data_types::i32, format::bfzyx, tensor{ 3, 6, 2, 2, 2 } });

    layout reorder_layout(data_types::f32, format::bfyx, { 1, 6, 4, 2 });

    tests::set_random_values<int32_t>(in_input);

    topology topology;
    topology.add(input_layout("Input", in_input->get_layout()));
    topology.add(reorder("reorder_input", input_info("Input"), format::bfzyx, data_types::f32));
    topology.add(crop("crop1", input_info("reorder_input"), tensor{1, 6, 2, 2, 2}, tensor(1, 0, 0, 0, 0)));
    topology.add(reorder("crop1_reorder", input_info("crop1"), reorder_layout));
    topology.add(reshape("reshape1", input_info("crop1_reorder"), tensor(6, 2, 2, 2)));
    topology.add(crop("crop2", input_info("reorder_input"), tensor{1, 6, 2, 2, 2}, tensor(2, 0, 0, 0, 0)));
    topology.add(reorder("crop2_reorder", input_info("crop2"), reorder_layout));
    topology.add(reshape("reshape2", input_info("crop2_reorder"), tensor(6, 2, 2, 2)));
    topology.add(concatenation("concat", { input_info("reshape1"), input_info("reshape2") }, 1, data_types::f32));
    topology.add(reorder("reorder_output", input_info("concat"), format::bfyx, data_types::i8));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    cldnn::network network(engine, topology, config);

    network.set_input_data("Input", in_input);
    auto outputs = network.execute();
    auto output = outputs.at("reorder_output").get_memory();
    cldnn::mem_lock<int8_t> output_ptr(output, get_test_stream());

    ExecutionConfig ref_config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(false));
    cldnn::network ref_network(engine, topology, ref_config);

    ref_network.set_input_data("Input", in_input);
    auto ref_outputs = ref_network.execute();
    auto ref_output = ref_outputs.at("reorder_output").get_memory();
    cldnn::mem_lock<int8_t> ref_output_ptr(ref_output, get_test_stream());

    int crop_batch_num = 6;
    int crop_feature_num = 2;
    int crop_y_size = 2;
    int crop_x_size = 2;
    for (int b = 0; b < crop_batch_num; ++b) {
        for (int f = 0; f < crop_feature_num; ++f) {
            for (int y = 0; y < crop_y_size; ++y) {
                for (int x = 0; x < crop_x_size; ++x) {
                    int output_linear_id = x + crop_x_size * (y + crop_y_size * (f + crop_feature_num * b));
                    ASSERT_EQ(output_ptr[output_linear_id], ref_output_ptr[output_linear_id]);
                }
            }
        }
    }

    // optimized reorder, concate
    auto crop_prim = network.get_primitive("crop1");
    ASSERT_EQ(crop_prim->can_be_optimized(), true);
    crop_prim = network.get_primitive("crop2");
    ASSERT_EQ(crop_prim->can_be_optimized(), true);
    auto reorder_prim = network.get_primitive("crop1_reorder");
    ASSERT_EQ(reorder_prim->can_be_optimized(), false);
    reorder_prim = network.get_primitive("crop2_reorder");
    ASSERT_EQ(reorder_prim->can_be_optimized(), false);
    auto concate = network.get_primitive("concat");
    ASSERT_EQ(concate->can_be_optimized(), false);
}

TEST(add_required_reorders, prevent_runtime_error_for_mvn_requiring_alignment) {
    auto& engine = get_test_engine();
    auto input_layout_dyn = layout{ ov::PartialShape{ov::Dimension::dynamic(), 24, 24, 2048},
                                    data_types::f16,
                                    format::bfyx };
    auto conv_weights = engine.allocate_memory({ {2048, 2048, 1, 1}, data_types::f16, format::bfyx });
    auto fc_weights = engine.allocate_memory({ {8192, 2048}, data_types::f32, format::bfyx });
    auto input_mem = engine.allocate_memory({ {1, 24, 24, 2048}, data_types::f16, format::bfyx });

    topology topology;
    topology.add(input_layout("input", input_layout_dyn));
    topology.add(data("weights", conv_weights));
    topology.add(data("fc_weights", fc_weights));
    topology.add(mvn("mvn1", input_info("input"), true, 1e-10f, false, {1}));
    topology.add(permute("permute1", input_info("mvn1"), {0, 3, 1, 2}));
    topology.add(convolution("conv", input_info("permute1"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    topology.add(permute("permute2", input_info("conv"), {0, 2, 3, 1}));
    topology.add(mvn("mvn2", input_info("permute2"), true, 1e-10f, false, {3}));
    topology.add(fully_connected("fc", input_info("mvn2"), "fc_weights", "", data_types::f32, 4, 2));
    topology.add(reorder("reorder", input_info("fc"), layout{ ov::PartialShape{1, 8192, 24, 24}, data_types::f16, format::bfyx }));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    network.set_input_data("input", input_mem);
    ASSERT_NO_THROW(network.execute());

    auto prog = network.get_program();
    ASSERT_NE(prog, nullptr);

    std::shared_ptr<primitive_inst> prim = nullptr;
    for (auto& item : network.get_executed_primitives()) {
        if (network.get_primitive(item.first)->get_node().is_type<fully_connected>())
            prim = network.get_primitive(item.first);
    }

    ASSERT_NE(prim, nullptr);
    const auto& fc_node = prim->get_node();
    ASSERT_TRUE(fc_node.is_all_valid_output_layouts());
}

class align_shape_for_numpy_broadcast_test: public ::testing::Test {
public:
    void test_eltwise_broadcast_static(format small_reshape_format) {
        // Topology :
        //   Input0 -> reorder -> (b_fs_yx_fsv16 / 5dims) -> Eltwise <- (bfyx / 3dims) <- Input1
        // Expected :
        //   Input0 -> reorder -> (b_fs_yx_fsv16 / 5dims) -> Eltwise <- (bfyx / 3dims) <- Input1
        auto& engine = get_test_engine();

        topology topology;
        topology.add(input_layout("input", layout{ ov::PartialShape{ 1, 32, 2, 4, 8 }, data_types::f32, format::bfzyx }));
        topology.add(reorder("reorder_input", input_info("input"), format::b_fs_zyx_fsv16, data_types::f16));
        topology.add(input_layout("eltw_input", layout{ ov::PartialShape{ 2, 4, 8 }, data_types::f16, small_reshape_format }));
        topology.add(eltwise("eltwise", input_info("eltw_input"), input_info("reorder_input"), eltwise_mode::sum, ov::op::AutoBroadcastType::NUMPY));
        topology.add(reorder("reorder_output", input_info("eltwise"), format::b_fs_zyx_fsv16, data_types::f32));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));

        program::ptr prog = nullptr;
        OV_ASSERT_NO_THROW(prog = program::build_program(engine, topology, config));
        ASSERT_NE(prog, nullptr);

        auto prog_impl = prog.get();
        auto& eltwise_node = prog_impl->get_node("eltwise");
        auto input_layouts = eltwise_node.get_input_layouts();

        ASSERT_EQ(input_layouts[0].format, small_reshape_format);
        ASSERT_EQ(input_layouts[1].format, format::b_fs_zyx_fsv16);
        ASSERT_EQ(eltwise_node.get_input_pshape(0).rank(), 3);
        ASSERT_EQ(eltwise_node.get_input_pshape(1).rank(), 5);
        ASSERT_EQ(eltwise_node.get_output_layout().format, format::b_fs_zyx_fsv16);
    }

    void test_eltwise_broadcast_dynamic(format small_reshape_format) {
        // Topology :
        //   Input0 -> reorder -> (b_fs_yx_fsv16 / 5dims) -> Eltwise <- (bfyx / 3dims) <- Input1
        // Expected :
        //   Input0 -> reorder -> (b_fs_yx_fsv16 / 5dims) -> Eltwise <- (bfyx / 3dims) <- Input1
        tests::random_generator rg(GET_SUITE_NAME);
        auto& engine = get_test_engine();

        // data input0
        auto dyn_input_layout = layout{ ov::PartialShape{ -1, 32, -1, -1, 8 }, data_types::f32, format::bfzyx };
        auto input_pshape = ov::PartialShape{ 1, 32, 2, 4, 8 };
        auto input_mem = engine.allocate_memory({ input_pshape, data_types::f32, format::bfzyx });
        auto input_data = rg.generate_random_1d<float>(input_mem->count(), -16.0, 16.0);
        set_values<float>(input_mem, input_data);

        // input1
        auto dyn_eltw_layout = layout{ ov::PartialShape{ -1, -1, 8 }, data_types::f16, small_reshape_format };
        auto elt_pshape = ov::PartialShape{ 2, 4, 8 };
        auto elt_mem = engine.allocate_memory({ elt_pshape, data_types::f16, small_reshape_format });
        auto elt_data = rg.generate_random_1d<ov::float16>(elt_mem->count(), -16.0f, 16.0f);
        set_values<ov::float16>(elt_mem, elt_data);

        topology topology;
        topology.add(input_layout("input", dyn_input_layout));
        topology.add(reorder("reorder_input", input_info("input"), format::b_fs_zyx_fsv16, data_types::f16));
        topology.add(input_layout("eltw_input", dyn_eltw_layout));
        topology.add(eltwise("eltwise", input_info("eltw_input"), input_info("reorder_input"), eltwise_mode::sum, ov::op::AutoBroadcastType::NUMPY));
        topology.add(reorder("reorder_output", input_info("eltwise"), format::b_fs_zyx_fsv16, data_types::f32));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        program::ptr prog = nullptr;
        OV_ASSERT_NO_THROW(prog = program::build_program(engine, topology, config));
        ASSERT_NE(prog, nullptr);

        auto prog_impl = prog.get();
        auto& eltwise_node = prog_impl->get_node("eltwise");
        auto input_layouts = eltwise_node.get_input_layouts();

        ASSERT_EQ(input_layouts[0].format, small_reshape_format);
        ASSERT_EQ(input_layouts[1].format, format::b_fs_zyx_fsv16);
        ASSERT_EQ(eltwise_node.get_input_pshape(0).rank(), 3);
        ASSERT_EQ(eltwise_node.get_input_pshape(1).rank(), 5);
        ASSERT_EQ(eltwise_node.get_output_layout().format, format::b_fs_zyx_fsv16);
    }
};

TEST_F(align_shape_for_numpy_broadcast_test, test_simple_format) {
    this->test_eltwise_broadcast_static(format::bfyx);
}

TEST_F(align_shape_for_numpy_broadcast_test, test_imcompatible_blocked_format) {
    this->test_eltwise_broadcast_static(format::b_fs_yx_fsv16);
}

TEST_F(align_shape_for_numpy_broadcast_test, test_simple_format_dynamic) {
    this->test_eltwise_broadcast_dynamic(format::bfyx);
}

TEST_F(align_shape_for_numpy_broadcast_test, test_imcompatible_blocked_format_dynamic) {
    this->test_eltwise_broadcast_dynamic(format::b_fs_yx_fsv16);
}
