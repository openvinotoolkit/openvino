// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "primitive_inst.h"
#include "primitive_type_base.h"
#include "registry/implementation_manager.hpp"
#include "registry/registry.hpp"
#include "test_utils.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>

using namespace cldnn;
using namespace ::tests;

namespace cldnn {

// Fake multi-output primitive whose declared alias pairs are set per test case.
struct alias_test_primitive : public primitive_base<alias_test_primitive> {
    CLDNN_DECLARE_PRIMITIVE(alias_test_primitive)

    alias_test_primitive() : primitive_base("", {}) {}
    alias_test_primitive(const primitive_id& id,
                         const std::vector<input_info>& inputs,
                         std::optional<program_node::input_output_alias> alias)
        : primitive_base(id, inputs, 2, {optional_data_type(), optional_data_type()}, {padding(), padding()}),
          alias(alias) {}

    std::optional<program_node::input_output_alias> alias;
};

template <>
struct typed_program_node<alias_test_primitive> : public typed_program_node_base<alias_test_primitive> {
    using parent = typed_program_node_base<alias_test_primitive>;
    using parent::parent;

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }

    std::optional<input_output_alias> get_supported_input_output_alias() const override {
        return get_primitive()->alias;
    }
};

using alias_test_primitive_node = typed_program_node<alias_test_primitive>;

template <>
class typed_primitive_inst<alias_test_primitive> : public typed_primitive_inst_base<alias_test_primitive> {
public:
    using parent = typed_primitive_inst_base<alias_test_primitive>;
    using parent::parent;

    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(alias_test_primitive_node const& /*node*/, const kernel_impl_params& impl_param) {
        return { impl_param.get_input_layout(0), impl_param.get_input_layout(0) };
    }
    static layout calc_output_layout(alias_test_primitive_node const& /*node*/, kernel_impl_params const& impl_param) {
        return impl_param.get_input_layout(0);
    }
    static std::string to_string(alias_test_primitive_node const& /*node*/) { return "alias_test_primitive"; }
};
using alias_test_primitive_inst = typed_primitive_inst<alias_test_primitive>;

GPU_DEFINE_PRIMITIVE_TYPE_ID(alias_test_primitive)

struct alias_test_impl : public typed_primitive_impl<alias_test_primitive> {
    using parent = typed_primitive_impl<alias_test_primitive>;
    using parent::parent;
    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::alias_test_impl)

    alias_test_impl() : parent("alias_test_impl") {}

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<alias_test_impl>(*this);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& /*events*/, alias_test_primitive_inst& instance) override {
        return instance.get_network().get_stream().create_user_event(true);
    }

    void init_kernels(const kernels_cache& /*kernels_cache*/, const kernel_impl_params& /*params*/) override {}

    static std::unique_ptr<primitive_impl> create(const program_node& /*node*/, const kernel_impl_params& /*params*/) {
        return std::make_unique<alias_test_impl>();
    }
};

}  // namespace cldnn

namespace ov::intel_gpu {

using namespace cldnn;

template <>
const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<alias_test_primitive>::get_implementations() {
    static bool initialize = true;
    if (initialize) {
        implementation_map<alias_test_primitive>::add(impl_types::ocl, shape_types::static_shape, alias_test_impl::create, {});
        initialize = false;
    }

    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_GET_INSTANCE_OCL(alias_test_primitive, shape_types::static_shape)
    };
    return impls;
}

}  // namespace ov::intel_gpu

namespace {

constexpr auto in0 = "in0";
constexpr auto in1 = "in1";
constexpr auto in2 = "in2";
constexpr auto producer = "producer";
constexpr auto net_output = "net_output";

class InputOutputAliasTest : public ::testing::Test {
protected:
    layout data_layout{ ov::PartialShape{1, 4}, data_types::f32, format::bfyx };

    topology make_topology(std::optional<program_node::input_output_alias> alias) {
        return topology{
            input_layout(in0, data_layout),
            input_layout(in1, data_layout),
            input_layout(in2, data_layout),
            alias_test_primitive(producer, { input_info(in0), input_info(in1), input_info(in2) }, alias),
        };
    }

    // Builds the network with graph-level optimizations off, so the test's output-facing
    // reorder/pass-through nodes stay exactly as written instead of being folded away.
    network build(topology& topo) {
        ExecutionConfig config = get_test_default_config(get_test_engine());
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        return network(get_test_engine(), topo, config);
    }

    // Builds the network, binds real input memories, then queries can_bind_user_output_memory() for
    // output_id against either the memory bound to candidate_input or (when separate_candidate is set)
    // an entirely separate allocation that doesn't overlap any bound input.
    bool query(topology& topo, const primitive_id& output_id, size_t candidate_input, bool separate_candidate = false) {
        auto net = build(topo);
        const auto input_ids = net.get_input_ids();
        std::vector<memory::ptr> inputs(3);
        for (size_t i = 0; i < 3; ++i) {
            const auto input_id = "in" + std::to_string(i);
            if (std::find(input_ids.begin(), input_ids.end(), input_id) == input_ids.end())
                continue;
            inputs[i] = get_test_engine().allocate_memory(data_layout);
            net.set_input_data(input_id, inputs[i]);
        }
        auto candidate = separate_candidate ? get_test_engine().allocate_memory(data_layout) : inputs.at(candidate_input);
        return net.can_bind_user_output_memory(output_id, *candidate);
    }
};

// Producer is the output and declares {0,0}; a candidate that is the exact same allocation as dependency 0 is accepted.
TEST_F(InputOutputAliasTest, declared_pair_accepts_exact_dependency_memory) {
    auto topo = make_topology(program_node::input_output_alias{0, 0});

    EXPECT_TRUE(query(topo, producer, 0));
}

// Declared dependency index is 1; a candidate matching in1's memory (not in0's) must also be accepted.
TEST_F(InputOutputAliasTest, declared_pair_accepts_other_declared_dependency) {
    auto topo = make_topology(program_node::input_output_alias{1, 0});

    EXPECT_TRUE(query(topo, producer, 1));
}

// The candidate is the same allocation as in1, but the producer only declares dependency 0 as aliasable.
TEST_F(InputOutputAliasTest, different_input_memory_is_not_aliasable) {
    auto topo = make_topology(program_node::input_output_alias{0, 0});

    EXPECT_FALSE(query(topo, producer, 1));
}

// The network output is producer port 0, but only port 1 is declared aliasable.
TEST_F(InputOutputAliasTest, undeclared_output_port_is_not_aliasable) {
    auto topo = make_topology(program_node::input_output_alias{0, 1});

    EXPECT_FALSE(query(topo, producer, 0));
}

// A non-optimized reorder writes the output, so the producer's declaration doesn't cover that write.
TEST_F(InputOutputAliasTest, declaration_of_non_writer_is_ignored) {
    auto topo = make_topology(program_node::input_output_alias{0, 0});
    topo.add(reorder(net_output, input_info(producer, 0), format::bfyx, data_types::f32));

    EXPECT_FALSE(query(topo, net_output, 0));
}

// Another node also reads the aliased input and may run after the writer overwrote it.
TEST_F(InputOutputAliasTest, other_reader_of_aliased_input_is_not_aliasable) {
    auto topo = make_topology(program_node::input_output_alias{0, 0});
    topo.add(reorder("other_reader", input_info(in0), format::bfyx, data_types::f16));

    EXPECT_FALSE(query(topo, producer, 0));
}

// The writer reads the aliased input through an undeclared dependency index as well.
TEST_F(InputOutputAliasTest, aliased_input_on_undeclared_dependency_is_not_aliasable) {
    topology topo{
        input_layout(in0, data_layout),
        alias_test_primitive(producer, { input_info(in0), input_info(in0) }, program_node::input_output_alias{0, 0}),
    };

    EXPECT_FALSE(query(topo, producer, 0));
}

// An extra node sits between the declaring producer and the network output; the direct-connection
// contract means this is conservatively rejected rather than traversed.
TEST_F(InputOutputAliasTest, intermediate_node_is_not_traversed) {
    auto topo = make_topology(program_node::input_output_alias{0, 0});
    topo.add(alias_test_primitive("fwd", { input_info(producer, 0) }, std::nullopt));
    topo.add(reorder(net_output, input_info("fwd", 0), format::bfyx, data_types::f32));

    EXPECT_FALSE(query(topo, net_output, 0));
}

// A primitive that declares no alias pair at all must never authorize sharing its output buffer.
TEST_F(InputOutputAliasTest, primitive_without_declaration_is_not_aliasable) {
    auto topo = make_topology(std::nullopt);
    topo.add(reorder(net_output, input_info(producer, 0), format::bfyx, data_types::f32));

    EXPECT_FALSE(query(topo, net_output, 0));
}

// No declaring primitive is involved at all (plain input -> converting reorder -> output); still no
// aliasing. The reorder is left as a real convert (f32 to f16) rather than an identity pass-through.
TEST_F(InputOutputAliasTest, plain_output_without_declaring_producer_is_not_aliasable) {
    topology topo{
        input_layout(in0, data_layout),
        reorder(net_output, input_info(in0), format::bfyx, data_types::f16),
    };

    EXPECT_FALSE(query(topo, net_output, 0));
}

// A candidate that doesn't overlap any bound network input memory is always safe to bind directly,
// regardless of any declared alias -- generic non-overlapping zero-copy stays available.
TEST_F(InputOutputAliasTest, non_overlapping_memory_is_allowed_for_ordinary_producer) {
    auto topo = make_topology(program_node::input_output_alias{0, 0});

    EXPECT_TRUE(query(topo, producer, 0, true));
}

}  // namespace

