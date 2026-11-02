// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_MLIR_FOR_GPU

#    include <memory>
#    include <vector>

#    include "intel_gpu/graph/program.hpp"
#    include "intel_gpu/primitives/data.hpp"
#    include "intel_gpu/primitives/input_layout.hpp"
#    include "intel_gpu/primitives/mlir_primitive.hpp"
#    include "intel_gpu/runtime/engine.hpp"
#    include "mlir_primitive_inst.h"
#    include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

// mlir_primitive without an MLIROp (and so without a compiled MLIR program). This is enough to check
// the layout handling at program build time
mlir_primitive make_mlir_primitive(const std::vector<input_info>& inputs, std::vector<format::type> required_fmts) {
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
    topology.add(make_mlir_primitive({input_info("input"), input_info("scale")}, {format::byfx, format::byfx}));

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

#endif  // ENABLE_MLIR_FOR_GPU
