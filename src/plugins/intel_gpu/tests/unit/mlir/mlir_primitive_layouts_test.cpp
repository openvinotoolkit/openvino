// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <map>
#include <memory>
#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/mlir_primitive.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "mlir_primitive_inst.h"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

// mlir_primitive without an MLIROp (and so without a compiled MLIR program). This is enough to check
// the layout handling at program build time
mlir_primitive make_mlir_primitive(const std::vector<input_info>& inputs, std::map<size_t, format::type> required_fmts) {
    auto shape_infer = [](const std::vector<ov::PartialShape>& shapes) {
        return std::vector<ov::PartialShape>{shapes[0]};
    };
    mlir_primitive prim(/*id=*/"mlir",
                        /*inputs=*/inputs,
                        /*op=*/nullptr,
                        /*shape_infer_f=*/shape_infer,
                        /*num_outputs=*/1,
                        /*out_types=*/std::vector<optional_data_type>{data_types::f16});
    prim.required_input_fmts = std::move(required_fmts);
    return prim;
}

program::ptr build(const topology& topology) {
    auto& engine = get_test_engine();
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    return program::build_program(engine, topology, config, false, false);
}

}  // namespace

// The formats required by the MLIR program are delivered both for runtime inputs
// (query_formats -> reorder_inputs) and for constant ones (explicit reorder at the end of reorder_inputs,
// folded by propagate_constants).
TEST(mlir_primitive_layouts, required_formats_are_delivered) {
    auto& engine = get_test_engine();
    const layout l{ov::PartialShape{2, 8, 4}, data_types::f16, format::bfyx};

    // mlir_primitive inputs are: 'bfyx',
    // and the primitive requires 'byfx' for both inputs.
    topology topology;
    topology.add(input_layout("input", l));
    topology.add(data("scale", engine.allocate_memory(l)));
    topology.add(make_mlir_primitive({input_info("input"), input_info("scale")}, {{0, format::byfx}, {1, format::byfx}}));

    auto prog = build(topology);
    ASSERT_NE(prog, nullptr);

    auto& node = prog->get_node("mlir");
    // Verify that both inputs now have the required 'byfx' format
    EXPECT_EQ(node.get_input_layout(0).format, format::byfx);
    EXPECT_EQ(node.get_input_layout(1).format, format::byfx);
    EXPECT_TRUE(node.get_dependency(1).is_constant());
    EXPECT_EQ(node.get_output_layout().format, format::bfyx);
}

// Without requirements every input is plain, the graph is left as is.
TEST(mlir_primitive_layouts, plain_by_default) {
    auto& engine = get_test_engine();
    const layout l{ov::PartialShape{2, 8, 4}, data_types::f16, format::bfyx};

    topology topology;
    topology.add(input_layout("input", l));
    topology.add(data("scale", engine.allocate_memory(l)));
    topology.add(make_mlir_primitive({input_info("input"), input_info("scale")}, {}));

    auto prog = build(topology);
    ASSERT_NE(prog, nullptr);

    auto& node = prog->get_node("mlir");
    EXPECT_EQ(node.get_input_layout(0).format, format::bfyx);
    EXPECT_EQ(node.get_input_layout(1).format, format::bfyx);
    EXPECT_EQ(node.get_dependency(0).id(), "input");
}

// Without requirements every input must be plain: non-plain inputs (both runtime and constant ones)
// are reordered to the default format.
TEST(mlir_primitive_layouts, non_plain_inputs_are_reordered_to_plain) {
    auto& engine = get_test_engine();
    const layout l{ov::PartialShape{2, 8, 4}, data_types::f16, format::byfx};

    topology topology;
    topology.add(input_layout("input", l));
    topology.add(data("scale", engine.allocate_memory(l)));
    topology.add(make_mlir_primitive({input_info("input"), input_info("scale")}, {}));

    auto prog = build(topology);
    ASSERT_NE(prog, nullptr);

    auto& node = prog->get_node("mlir");
    EXPECT_EQ(node.get_input_layout(0).format, format::bfyx);
    EXPECT_EQ(node.get_input_layout(1).format, format::bfyx);
    EXPECT_TRUE(node.get_dependency(1).is_constant());
    EXPECT_EQ(node.get_output_layout().format, format::bfyx);
}

// Rank-5 required format: bzyxf is not a "simple" data format, it's accepted only via
// the special case in mlir_primitive::is_strided_format.
TEST(mlir_primitive_layouts, rank5_bzyxf_required_format_is_delivered) {
    auto& engine = get_test_engine();
    const layout l{ov::PartialShape{2, 8, 4, 3, 5}, data_types::f16, format::bfzyx};

    topology topology;
    topology.add(input_layout("input", l));
    topology.add(data("scale", engine.allocate_memory(l)));
    topology.add(make_mlir_primitive({input_info("input"), input_info("scale")}, {{0, format::bzyxf}, {1, format::bzyxf}}));

    auto prog = build(topology);
    ASSERT_NE(prog, nullptr);

    auto& node = prog->get_node("mlir");
    EXPECT_EQ(node.get_input_layout(0).format, format::bzyxf);
    EXPECT_EQ(node.get_input_layout(1).format, format::bzyxf);
    EXPECT_TRUE(node.get_dependency(1).is_constant());
    EXPECT_EQ(node.get_output_layout().format, format::bfzyx);
}

TEST(mlir_primitive_layouts, format_from_dim_order) {
    EXPECT_EQ(mlir_primitive::format_from_dim_order({1, 0}), format::fbyx);
    EXPECT_EQ(mlir_primitive::format_from_dim_order({0, 2, 1}), format::byfx);
    EXPECT_EQ(mlir_primitive::format_from_dim_order({0, 2, 3, 1}), format::byxf);
    EXPECT_EQ(mlir_primitive::format_from_dim_order({0, 2, 3, 4, 1}), format::bzyxf);

    // Plain orders give the default format of their rank.
    EXPECT_EQ(mlir_primitive::format_from_dim_order({0, 1}), format::bfyx);
    EXPECT_EQ(mlir_primitive::format_from_dim_order({0, 1, 2}), format::bfyx);
    EXPECT_EQ(mlir_primitive::format_from_dim_order({0, 1, 2, 3}), format::bfyx);
    EXPECT_EQ(mlir_primitive::format_from_dim_order({0, 1, 2, 3, 4}), format::bfzyx);
    EXPECT_EQ(mlir_primitive::format_from_dim_order({0, 1, 2, 3, 4, 5}), format::bfwzyx);

    // No plain data format with this order: {3, 2, 1, 0} exists only as a weights format, {0, 0, 1} isn't a permutation.
    EXPECT_THROW(mlir_primitive::format_from_dim_order({3, 2, 1, 0}), ov::Exception);
    EXPECT_THROW(mlir_primitive::format_from_dim_order({0, 0, 1}), ov::Exception);
    // The rank comes from the order, so an empty one is rejected.
    EXPECT_THROW(mlir_primitive::format_from_dim_order({}), ov::Exception);
}
