// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/convolution.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/pooling.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <map>
#include <string>
#include <vector>

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

// ResNet-18 trunk at a 28x28 stage input. oneDNN rejects b_fs_yx_fsv16 for all of them at a batch of 16 or more.
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

enum class expected_impl { ocl, onednn_jit, onednn_reference };

bool is_onednn_reference(const primitive_impl& impl) {
    return impl.is_onednn() && impl.get_kernel_name().rfind("ocl:ref", 0) == 0;
}

class conv_dynamic_batch_ref_fallback : public ::testing::Test {
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

    topology build_topology(const ov::Dimension& batch) const {
        const auto& in = branches.front().geometry;
        topology topo;
        topo.add(input_layout("input", layout{ov::PartialShape{batch, in.in_features, in.in_size, in.in_size}, data_types::f16, format::bfyx}));
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
            // Pooling is not in the byxf_onednn_convolution whitelist, so the convolution keeps b_fs_yx_fsv16.
            topo.add(pooling(branch.id + "_pool", input_info(last), pooling_mode::max, {1, 1}, {1, 1}));
            topo.add(reorder(branch.id + "_out", input_info(branch.id + "_pool"), format::bfyx, data_types::f32));
        }
        return topo;
    }

    static ExecutionConfig make_config(const ov::intel_gpu::ImplForcingMap& forcing = {}) {
        auto config = get_test_default_config(get_test_engine());
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        if (!forcing.empty()) {
            config.set_property(ov::intel_gpu::force_implementations(forcing));
        }
        return config;
    }

    network::ptr build_network(const ov::Dimension& batch, const ov::intel_gpu::ImplForcingMap& forcing = {}) const {
        return std::make_shared<network>(get_test_engine(), build_topology(batch), make_config(forcing));
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

    // Reference: oneDNN disabled and the OCL reference kernel in bfyx with the same static batch.
    void compare_with_reference(const results& actual, int64_t batch, const tensors& inputs) const {
        ov::intel_gpu::ImplForcingMap forcing;
        for (const auto& branch : branches) {
            forcing[branch.id] = {format::bfyx, "convolution_gpu_ref", impl_types::ocl};
        }
        auto config = make_config(forcing);
        config.set_property(ov::intel_gpu::use_onednn(false));
        network reference(get_test_engine(), build_topology(ov::Dimension(batch)), config);
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

    // The convolutions must be in the state that gives oneDNN's reference impl without the fallback.
    void check_dynamic_fsv16_onednn_convs(network& net) const {
        for (const auto& branch : branches) {
            const auto& node = net.get_primitive(branch.id)->get_node();
            ASSERT_TRUE(node.is_dynamic()) << branch.id;
            ASSERT_EQ(node.get_preferred_impl_type(), impl_types::onednn) << branch.id;
            ASSERT_EQ(node.get_input_layout(0).format, format::b_fs_yx_fsv16) << branch.id;
            ASSERT_EQ(node.get_output_layout(0).format, format::b_fs_yx_fsv16) << branch.id;
        }
    }

    void run_and_check(network& net, int64_t batch, expected_impl expected) {
        const auto inputs = make_inputs(batch);
        const auto actual = run(net, inputs);

        for (const auto& branch : branches) {
            const auto* impl = net.get_primitive(branch.id)->get_impl();
            ASSERT_NE(impl, nullptr) << branch.id;
            const auto& name = impl->get_kernel_name();
            switch (expected) {
            case expected_impl::ocl:
                EXPECT_FALSE(impl->is_onednn()) << branch.id << ", batch " << batch << ": " << name;
                break;
            case expected_impl::onednn_jit:
                EXPECT_TRUE(impl->is_onednn()) << branch.id << ", batch " << batch << ": " << name;
                EXPECT_EQ(name.rfind("jit:ir", 0), 0u) << branch.id << ", batch " << batch << ": " << name;
                break;
            case expected_impl::onednn_reference:
                EXPECT_TRUE(is_onednn_reference(*impl)) << branch.id << ", batch " << batch << ": " << name;
                break;
            }
        }
        compare_with_reference(actual, batch, inputs);
    }

    tests::random_generator rg;
    std::vector<conv_branch> branches;
    bool residual_relu = false;
};

TEST_F(conv_dynamic_batch_ref_fallback, fused_residual_relu_uses_ocl) {
    add_conv("conv", trunk_geometries[0], true);
    residual_relu = true;
    auto net = build_network(ov::Dimension::dynamic());
    ASSERT_NO_FATAL_FAILURE(check_dynamic_fsv16_onednn_convs(*net));
    ASSERT_EQ(net->get_primitive("conv")->get_node().get_fused_primitives().size(), 2u);

    ASSERT_NO_FATAL_FAILURE(run_and_check(*net, 25, expected_impl::ocl));
    run_and_check(*net, 150, expected_impl::ocl);
}

TEST_F(conv_dynamic_batch_ref_fallback, downsample_convs_use_ocl) {
    add_conv("conv_3x3", trunk_geometries[1], false);
    add_conv("conv_1x1", trunk_geometries[2], false);
    auto net = build_network(ov::Dimension::dynamic());
    ASSERT_NO_FATAL_FAILURE(check_dynamic_fsv16_onednn_convs(*net));

    run_and_check(*net, 25, expected_impl::ocl);
}

TEST_F(conv_dynamic_batch_ref_fallback, impl_follows_batch_and_reuses_cache) {
    add_conv("conv", trunk_geometries[0], true);
    residual_relu = true;
    auto net = build_network(ov::Dimension::dynamic());
    ASSERT_NO_FATAL_FAILURE(check_dynamic_fsv16_onednn_convs(*net));
    ASSERT_EQ(net->get_primitive("conv")->get_node().get_fused_primitives().size(), 2u);

    auto cached_conv_impls = [&net]() {
        const auto keys = net->get_program()->get_implementations_cache().get_all_keys();
        return std::count_if(keys.begin(), keys.end(), [](const kernel_impl_params& params) {
            return params.desc->id == "conv";
        });
    };

    ASSERT_NO_FATAL_FAILURE(run_and_check(*net, 1, expected_impl::onednn_jit));
    ASSERT_NO_FATAL_FAILURE(run_and_check(*net, 25, expected_impl::ocl));
    const auto kernel_at_batch25 = net->get_primitive("conv")->get_impl()->get_kernel_name();
    ASSERT_NO_FATAL_FAILURE(run_and_check(*net, 8, expected_impl::onednn_jit));
    ASSERT_NO_FATAL_FAILURE(run_and_check(*net, 150, expected_impl::ocl));
    const auto cached_before = cached_conv_impls();
    ASSERT_EQ(cached_before, 4);

    ASSERT_NO_FATAL_FAILURE(run_and_check(*net, 25, expected_impl::ocl));
    EXPECT_EQ(cached_conv_impls(), cached_before);
    EXPECT_EQ(net->get_primitive("conv")->get_impl()->get_kernel_name(), kernel_at_batch25);
}

TEST_F(conv_dynamic_batch_ref_fallback, batch8_keeps_onednn_jit) {
    add_conv("conv", trunk_geometries[0], true);
    auto net = build_network(ov::Dimension::dynamic());
    ASSERT_NO_FATAL_FAILURE(check_dynamic_fsv16_onednn_convs(*net));

    run_and_check(*net, 8, expected_impl::onednn_jit);
}

TEST_F(conv_dynamic_batch_ref_fallback, forced_onednn_keeps_reference_impl) {
    add_conv("conv", trunk_geometries[0], true);
    auto net = build_network(ov::Dimension::dynamic(), {{"conv", {format::b_fs_yx_fsv16, "", impl_types::onednn}}});
    ASSERT_NO_FATAL_FAILURE(check_dynamic_fsv16_onednn_convs(*net));

    run_and_check(*net, 25, expected_impl::onednn_reference);
}

TEST_F(conv_dynamic_batch_ref_fallback, static_batch_keeps_onednn) {
    add_conv("conv", trunk_geometries[0], true);
    auto net = build_network(ov::Dimension(25));
    ASSERT_FALSE(net->get_primitive("conv")->get_node().is_dynamic());

    run_and_check(*net, 25, expected_impl::onednn_jit);
}

class conv_dynamic_batch_ref_fallback_geometry : public conv_dynamic_batch_ref_fallback, public ::testing::WithParamInterface<conv_geometry> {};

TEST_P(conv_dynamic_batch_ref_fallback_geometry, uses_ocl_at_batch25) {
    add_conv("conv", GetParam(), true);
    auto net = build_network(ov::Dimension::dynamic());
    ASSERT_NO_FATAL_FAILURE(check_dynamic_fsv16_onednn_convs(*net));

    run_and_check(*net, 25, expected_impl::ocl);
}

INSTANTIATE_TEST_SUITE_P(resnet18_trunk,
                         conv_dynamic_batch_ref_fallback_geometry,
                         ::testing::ValuesIn(trunk_geometries),
                         [](const ::testing::TestParamInfo<conv_geometry>& info) {
                             return geometry_name(info.param);
                         });

}  // namespace
