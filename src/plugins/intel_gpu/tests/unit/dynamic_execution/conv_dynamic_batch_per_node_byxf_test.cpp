// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/convolution.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/mvn.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <map>
#include <string>
#include <vector>

#include "layout_optimizer.h"
#include "primitive_inst.h"
#include "random_generator.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

struct conv_geometry {
    int64_t in_features;
    int64_t out_features;
    int64_t in_size;
    int64_t kernel;
    int64_t stride;
    int64_t pad;

    int64_t out_size() const {
        return (in_size + 2 * pad - kernel) / stride + 1;
    }
};

std::string geometry_name(const conv_geometry& g) {
    return "ic" + std::to_string(g.in_features) + "_oc" + std::to_string(g.out_features) + "_in" + std::to_string(g.in_size) + "_k" + std::to_string(g.kernel) +
           "_s" + std::to_string(g.stride);
}

void PrintTo(const conv_geometry& g, std::ostream* os) {
    *os << geometry_name(g);
}

// ResNet-18 trunk at a 28x28 stage input. With b_fs_yx_fsv16, oneDNN only has its reference impl for these at a batch of 16 or more.
const std::vector<conv_geometry> trunk_geometries = {
    {64, 64, 28, 3, 1, 1},
    {64, 128, 28, 3, 2, 1},
    {64, 128, 28, 1, 2, 0},
    {128, 128, 14, 3, 1, 1},
    {128, 256, 14, 3, 2, 1},
    {128, 256, 14, 1, 2, 0},
    {256, 256, 7, 3, 1, 1},
    {256, 512, 7, 3, 2, 1},
    {256, 512, 7, 1, 2, 0},
    {512, 512, 4, 3, 1, 1},
};

class conv_dynamic_batch_per_node_byxf : public ::testing::Test {
public:
    void SetUp() override {
        if (!get_test_engine().get_device_info().supports_immad) {
            GTEST_SKIP() << "oneDNN convolutions are only used on devices with immad support";
        }
        rg.set_seed(GET_SUITE_NAME);
    }

protected:
    struct conv_branch {
        primitive_id id;
        conv_geometry geometry;
        memory::ptr weights;
        memory::ptr bias;
    };

    using tensors = std::map<primitive_id, memory::ptr>;
    using results = std::map<primitive_id, std::vector<float>>;

    // All convolutions read the same input, so they must share in_features and in_size.
    void add_conv(const primitive_id& id, const conv_geometry& g, bool with_bias) {
        auto& engine = get_test_engine();
        const layout weights_layout{ov::PartialShape{g.out_features, g.in_features, g.kernel, g.kernel}, data_types::f16, format::bfyx};
        // Keeps outputs near unit scale for every fan-in.
        const double range = 3.0 / std::sqrt(static_cast<double>(g.in_features * g.kernel * g.kernel));
        auto weights = engine.allocate_memory(weights_layout);
        set_values(weights, rg.generate_random_1d<ov::float16>(weights_layout.count(), -range, range, 64));

        memory::ptr bias;
        if (with_bias) {
            const layout bias_layout{ov::PartialShape{1, g.out_features, 1, 1}, data_types::f16, format::bfyx};
            bias = engine.allocate_memory(bias_layout);
            set_values(bias, rg.generate_random_1d<ov::float16>(bias_layout.count(), -1, 1));
        }
        branches.push_back({id, g, weights, bias});
    }

    topology build_topology(const ov::Dimension& batch, const ov::Dimension& size) const {
        const auto& in = branches.front().geometry;
        topology topo;
        topo.add(input_layout("input", layout{ov::PartialShape{batch, in.in_features, size, size}, data_types::f16, format::bfyx}));
        // MVN is not in the byxf_onednn_convolution whitelist, so only the per-node rule can select byxf.
        topo.add(mvn("mvn", input_info("input"), true, 1e-5f, false, {2, 3}));
        topo.add(reorder("mvn_out", input_info("mvn"), format::bfyx, data_types::f32));
        for (const auto& branch : branches) {
            const auto& g = branch.geometry;
            const auto stride = static_cast<size_t>(g.stride);
            topo.add(data(branch.id + "_weights", branch.weights));
            primitive_id bias_id;
            if (branch.bias) {
                bias_id = branch.id + "_bias";
                topo.add(data(bias_id, branch.bias));
            }
            topo.add(convolution(branch.id,
                                 input_info("input"),
                                 branch.id + "_weights",
                                 bias_id,
                                 1,
                                 {stride, stride},
                                 {1, 1},
                                 {g.pad, g.pad},
                                 {g.pad, g.pad},
                                 false));
            primitive_id last = branch.id;
            if (residual_relu) {
                const ov::PartialShape out_shape{batch, g.out_features, g.out_size(), g.out_size()};
                topo.add(input_layout(branch.id + "_residual", layout{out_shape, data_types::f16, format::bfyx}));
                topo.add(eltwise(branch.id + "_add", {input_info(branch.id), input_info(branch.id + "_residual")}, eltwise_mode::sum));
                topo.add(activation(branch.id + "_relu", input_info(branch.id + "_add"), activation_func::relu));
                last = branch.id + "_relu";
            }
            topo.add(reorder(branch.id + "_out", input_info(last), format::bfyx, data_types::f32));
        }
        return topo;
    }

    // MVN is not in the byxf_onednn_convolution whitelist, so this branch keeps the network-wide byxf rule off.
    static void add_mvn_side_branch(topology& topo) {
        topo.add(input_layout("side", layout{ov::PartialShape{1, 16, 8, 8}, data_types::f16, format::bfyx}));
        topo.add(mvn("side_mvn", input_info("side"), true, 1e-5f, false, {2, 3}));
        topo.add(reorder("side_out", input_info("side_mvn"), format::bfyx, data_types::f32));
    }

    static ExecutionConfig make_config() {
        auto config = get_test_default_config(get_test_engine());
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        return config;
    }

    network::ptr build_network(const ov::Dimension& batch, const ov::Dimension& size) const {
        return std::make_shared<network>(get_test_engine(), build_topology(batch, size), make_config());
    }

    network::ptr build_network(const ov::Dimension& batch) const {
        return build_network(batch, ov::Dimension(branches.front().geometry.in_size));
    }

    tensors make_inputs(int64_t batch) {
        auto& engine = get_test_engine();
        tensors inputs;
        auto add_input = [&](const primitive_id& id, int64_t features, int64_t size) {
            const layout mem_layout{ov::PartialShape{batch, features, size, size}, data_types::f16, format::bfyx};
            auto mem = engine.allocate_memory(mem_layout);
            set_values(mem, rg.generate_random_1d<ov::float16>(mem_layout.count(), -1, 1));
            inputs[id] = mem;
        };

        const auto& in = branches.front().geometry;
        add_input("input", in.in_features, in.in_size);
        if (residual_relu) {
            for (const auto& branch : branches) {
                add_input(branch.id + "_residual", branch.geometry.out_features, branch.geometry.out_size());
            }
        }
        return inputs;
    }

    static results run(network& net, const tensors& inputs) {
        for (const auto& [id, mem] : inputs) {
            net.set_input_data(id, mem);
        }
        auto outputs = net.execute();
        results values;
        for (auto& [id, output] : outputs) {
            auto mem = output.get_memory();
            mem_lock<float, mem_lock_type::read> lock(mem, get_test_stream());
            // Dynamic outputs may sit in a larger, preallocated buffer.
            values[id].assign(lock.data(), lock.data() + output.get_layout().count());
        }
        return values;
    }

    // Reference: oneDNN disabled and the OCL reference kernel in bfyx with the same static shapes.
    void compare_with_reference(const results& actual, int64_t batch, const tensors& inputs) const {
        ov::intel_gpu::ImplForcingMap forcing;
        for (const auto& branch : branches) {
            forcing[branch.id] = {format::bfyx, "convolution_gpu_ref", impl_types::ocl};
        }
        auto config = make_config();
        config.set_property(ov::intel_gpu::force_implementations(forcing));
        config.set_property(ov::intel_gpu::use_onednn(false));
        const auto size = branches.front().geometry.in_size;
        network reference(get_test_engine(), build_topology(ov::Dimension(batch), ov::Dimension(size)), config);
        const auto expected = run(reference, inputs);

        for (const auto& branch : branches) {
            const auto out_id = branch.id + "_out";
            const auto& exp = expected.at(out_id);
            const auto& act = actual.at(out_id);
            ASSERT_EQ(exp.size(), act.size()) << out_id << ", batch " << batch;
            for (size_t i = 0; i < exp.size(); ++i) {
                ASSERT_NEAR(exp[i], act[i], 0.02f + 0.01f * std::abs(exp[i])) << out_id << ", batch " << batch << ", index " << i;
            }
        }
    }

    void check_onednn_conv_formats(network& net, format expected) const {
        ASSERT_EQ(net.get_program()->get_layout_optimizer().get_optimization_attributes().byxf_onednn_convolution, 0);
        for (const auto& branch : branches) {
            const auto& node = net.get_primitive(branch.id)->get_node();
            ASSERT_TRUE(node.is_dynamic()) << branch.id;
            ASSERT_EQ(node.get_preferred_impl_type(), impl_types::onednn) << branch.id;
            ASSERT_EQ(node.get_input_layout(0).format, expected) << branch.id;
            ASSERT_EQ(node.get_output_layout(0).format, expected) << branch.id;
        }
    }

    void run_and_check(network& net, int64_t batch) {
        const auto inputs = make_inputs(batch);
        const auto actual = run(net, inputs);

        for (const auto& branch : branches) {
            const auto* impl = net.get_primitive(branch.id)->get_impl();
            ASSERT_NE(impl, nullptr) << branch.id;
            const auto& name = impl->get_kernel_name();
            EXPECT_TRUE(impl->is_onednn()) << branch.id << ", batch " << batch << ": " << name;
            EXPECT_EQ(name.rfind("jit:ir", 0), 0u) << branch.id << ", batch " << batch << ": " << name;
        }
        compare_with_reference(actual, batch, inputs);
    }

    tests::random_generator rg;
    std::vector<conv_branch> branches;
    bool residual_relu = false;
};

TEST_F(conv_dynamic_batch_per_node_byxf, fused_residual_relu_uses_byxf_jit) {
    add_conv("conv", trunk_geometries[0], true);
    residual_relu = true;
    auto net = build_network(ov::Dimension::dynamic());
    ASSERT_NO_FATAL_FAILURE(check_onednn_conv_formats(*net, format::byxf));
    ASSERT_EQ(net->get_primitive("conv")->get_node().get_fused_primitives().size(), 2u);

    ASSERT_NO_FATAL_FAILURE(run_and_check(*net, 25));
    run_and_check(*net, 150);
}

TEST_F(conv_dynamic_batch_per_node_byxf, byxf_jit_at_every_runtime_batch) {
    add_conv("conv_3x3", trunk_geometries[1], true);
    add_conv("conv_1x1", trunk_geometries[2], false);
    auto net = build_network(ov::Dimension::dynamic());
    ASSERT_NO_FATAL_FAILURE(check_onednn_conv_formats(*net, format::byxf));

    for (const int64_t batch : {1, 8, 16, 25, 150, 25}) {
        ASSERT_NO_FATAL_FAILURE(run_and_check(*net, batch)) << "batch " << batch;
    }
}

// Convolutions with a static batch keep the blocked format, even when other dimensions are dynamic.
TEST_F(conv_dynamic_batch_per_node_byxf, static_batch1_conv_keeps_fsv16) {
    add_conv("conv", trunk_geometries[0], true);
    auto net = build_network(ov::Dimension(1), ov::Dimension::dynamic());
    ASSERT_NO_FATAL_FAILURE(check_onednn_conv_formats(*net, format::b_fs_yx_fsv16));

    run_and_check(*net, 1);
}

TEST_F(conv_dynamic_batch_per_node_byxf, int8_dynamic_batch_conv_keeps_fsv32) {
    auto& engine = get_test_engine();
    auto weights = engine.allocate_memory({ov::PartialShape{64, 64, 3, 3}, data_types::i8, format::bfyx});
    set_values(weights, rg.generate_random_1d<int8_t>(weights->get_layout().count(), -4, 4));

    topology topo(input_layout("input", layout{ov::PartialShape{ov::Dimension::dynamic(), 64, 28, 28}, data_types::i8, format::bfyx}),
                  data("weights", weights),
                  convolution("conv", input_info("input"), "weights", "", 1, {1, 1}, {1, 1}, {1, 1}, {1, 1}, false),
                  reorder("out", input_info("conv"), format::bfyx, data_types::f32));
    add_mvn_side_branch(topo);
    network net(engine, topo, make_config());

    ASSERT_EQ(net.get_program()->get_layout_optimizer().get_optimization_attributes().byxf_onednn_convolution, 0);
    const auto& node = net.get_primitive("conv")->get_node();
    ASSERT_EQ(node.get_preferred_impl_type(), impl_types::onednn);
    EXPECT_EQ(node.get_input_layout(0).format, format::b_fs_yx_fsv32);
}

TEST_F(conv_dynamic_batch_per_node_byxf, rank5_dynamic_batch_conv_keeps_fsv16) {
    auto& engine = get_test_engine();
    auto weights = engine.allocate_memory({ov::PartialShape{32, 32, 3, 3, 3}, data_types::f16, format::bfzyx});
    set_values(weights, rg.generate_random_1d<ov::float16>(weights->get_layout().count(), -1, 1));

    topology topo(input_layout("input", layout{ov::PartialShape{ov::Dimension::dynamic(), 32, 8, 8, 8}, data_types::f16, format::bfzyx}),
                  data("weights", weights),
                  convolution("conv", input_info("input"), "weights", "", 1, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, false),
                  reorder("out", input_info("conv"), format::bfzyx, data_types::f32));
    add_mvn_side_branch(topo);
    network net(engine, topo, make_config());

    ASSERT_EQ(net.get_program()->get_layout_optimizer().get_optimization_attributes().byxf_onednn_convolution, 0);
    const auto& node = net.get_primitive("conv")->get_node();
    ASSERT_EQ(node.get_preferred_impl_type(), impl_types::onednn);
    EXPECT_EQ(node.get_input_layout(0).format, format::b_fs_zyx_fsv16);
    EXPECT_EQ(node.get_output_layout(0).format, format::b_fs_zyx_fsv16);
}

TEST_F(conv_dynamic_batch_per_node_byxf, rank3_dynamic_batch_conv_keeps_fsv16) {
    auto& engine = get_test_engine();
    auto weights = engine.allocate_memory({ov::PartialShape{64, 64, 3}, data_types::f16, format::bfyx});
    set_values(weights, rg.generate_random_1d<ov::float16>(weights->get_layout().count(), -1, 1));

    topology topo(input_layout("input", layout{ov::PartialShape{ov::Dimension::dynamic(), 64, 100}, data_types::f16, format::bfyx}),
                  data("weights", weights),
                  convolution("conv", input_info("input"), "weights", "", 1, {1}, {1}, {1}, {1}, false),
                  reorder("out", input_info("conv"), format::bfyx, data_types::f32));
    add_mvn_side_branch(topo);
    network net(engine, topo, make_config());

    ASSERT_EQ(net.get_program()->get_layout_optimizer().get_optimization_attributes().byxf_onednn_convolution, 0);
    const auto& node = net.get_primitive("conv")->get_node();
    ASSERT_EQ(node.get_preferred_impl_type(), impl_types::onednn);
    EXPECT_EQ(node.get_input_layout(0).format, format::b_fs_yx_fsv16);
    EXPECT_EQ(node.get_output_layout(0).format, format::b_fs_yx_fsv16);
}

// MVN has no fast byxf kernel, so a convolution next to it keeps the blocked format.
TEST_F(conv_dynamic_batch_per_node_byxf, mvn_user_keeps_fsv16) {
    auto& engine = get_test_engine();
    auto weights = engine.allocate_memory({ov::PartialShape{64, 64, 3, 3}, data_types::f16, format::bfyx});
    set_values(weights, rg.generate_random_1d<ov::float16>(weights->get_layout().count(), -1, 1));

    topology topo(input_layout("input", layout{ov::PartialShape{ov::Dimension::dynamic(), 64, 28, 28}, data_types::f16, format::bfyx}),
                  data("weights", weights),
                  convolution("conv", input_info("input"), "weights", "", 1, {1, 1}, {1, 1}, {1, 1}, {1, 1}, false),
                  mvn("mvn", input_info("conv"), true, 1e-5f, false, {2, 3}),
                  reorder("out", input_info("mvn"), format::bfyx, data_types::f32));
    network net(engine, topo, make_config());

    ASSERT_EQ(net.get_program()->get_layout_optimizer().get_optimization_attributes().byxf_onednn_convolution, 0);
    const auto& node = net.get_primitive("conv")->get_node();
    ASSERT_EQ(node.get_preferred_impl_type(), impl_types::onednn);
    EXPECT_EQ(node.get_input_layout(0).format, format::b_fs_yx_fsv16);
    EXPECT_EQ(node.get_output_layout(0).format, format::b_fs_yx_fsv16);
}

class conv_dynamic_batch_per_node_byxf_geometry : public conv_dynamic_batch_per_node_byxf, public ::testing::WithParamInterface<conv_geometry> {};

TEST_P(conv_dynamic_batch_per_node_byxf_geometry, uses_byxf_jit_at_batch25) {
    add_conv("conv", GetParam(), true);
    auto net = build_network(ov::Dimension::dynamic());
    ASSERT_NO_FATAL_FAILURE(check_onednn_conv_formats(*net, format::byxf));

    run_and_check(*net, 25);
}

INSTANTIATE_TEST_SUITE_P(resnet18_trunk,
                         conv_dynamic_batch_per_node_byxf_geometry,
                         ::testing::ValuesIn(trunk_geometries),
                         [](const ::testing::TestParamInfo<conv_geometry>& info) {
                             return geometry_name(info.param);
                         });

}  // namespace
