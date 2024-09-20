// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/input_layout.hpp>
#include "test_utils.h"
#include "program_wrapper.h"

#include "primitive_inst.h"
#include "shape_of_inst.h"
#include "select_inst.h"
#include "broadcast_inst.h"
#include "eltwise_inst.h"
#include "fully_connected_inst.h"
#include "gemm_inst.h"
#include "gather_inst.h"

using namespace cldnn;
using namespace ::tests;

namespace {

// first - input shape, second - expected input shape after canonicalization, third - expected output shape after canonicalization
using Shapes = std::tuple<std::vector<ov::PartialShape>, std::vector<ov::PartialShape>, std::vector<ov::PartialShape>>;

void canonicalization_test(cldnn::topology topology, std::string prim_name,
                           const std::vector<ov::PartialShape>& expected_input_pshapes,
                           const std::vector<ov::PartialShape>& expected_output_pshapes,
                           bool enable_fusing = false) {
    auto& engine = get_test_engine();

    ExecutionConfig config({ov::intel_gpu::optimize_data(true),
                            ov::intel_gpu::allow_new_shape_infer(true)});

    auto prog = program::build_program(engine, topology, config, false, true);
    if (enable_fusing) {
        program_wrapper::apply_opt_pass<prepare_primitive_fusing>(*prog);
    }
    program_wrapper::run_graph_compilation(*prog);

    auto& node = prog->get_node(prim_name);
    auto impl = node.get_selected_impl();
    ASSERT_TRUE(impl != nullptr);

    auto impl_param = node.get_kernel_impl_params();
    auto canonicalized_impl_param = impl->canonicalize_shapes(*impl_param);

    for (size_t i = 0; i < canonicalized_impl_param.input_layouts.size(); ++i) {
        EXPECT_TRUE(canonicalized_impl_param.input_layouts[i].get_partial_shape() == expected_input_pshapes[i]);
    }

    for (size_t i = 0; i < canonicalized_impl_param.output_layouts.size(); ++i) {
        EXPECT_TRUE(canonicalized_impl_param.output_layouts[i].get_partial_shape() == expected_output_pshapes[i]);
    }
};

layout create_default_layout(const ov::PartialShape& pshape) {
    return layout {pshape, data_types::f32, format::bfyx};
}

std::vector<Shapes> select_shapes {
    {{{2, 2}, {1, 2}, {2, 1}}, {{1, 1, 2, 2}, {1, 1, 2, 2}, {1, 1, 2, 2}}, {{1, 1, 2, 2}}}
};

TEST(canonicalization, select) {
    for (const auto& shapes : select_shapes) {
        layout input0_layout = create_default_layout(std::get<0>(shapes)[0]);
        layout input1_layout = create_default_layout(std::get<1>(shapes)[0]);
        layout input2_layout = create_default_layout(std::get<2>(shapes)[0]);

        topology topology;
        topology.add(input_layout("mask", input0_layout));
        topology.add(input_layout("input1", input1_layout));
        topology.add(input_layout("input2", input2_layout));
        topology.add(cldnn::select("select", input_info("mask"), input_info("input1"), input_info("input2")));

        canonicalization_test(topology, "select", std::get<1>(shapes), std::get<2>(shapes));
    }
}

struct broadcast_params {
    ov::Shape target_shape;
    ov::AxisSet axes_mapping;
    ov::op::BroadcastModeSpec broadcast_mode;
};

std::vector<std::pair<Shapes, broadcast_params>> broadcast_shapes_with_params {
    {{{{5}}, {{1, 1, 5, 1}}, {{3, 1, 5, 1}}}, {{3, 1, 5}, {}, ov::op::BroadcastType::NUMPY}},
    {{{{5}}, {{1, 1, 1, 1, 5}}, {{1, 2, 3, 4, 5}}}, {{1, 2, 3, 4, 5}, {}, ov::op::BroadcastType::NUMPY}},
    {{{{3, 1}}, {{1, 1, 3, 1}}, {{1, 2, 3, 4}}}, {{1, 2, 3, 4}, {}, {ov::op::BroadcastType::PDPD, 2}}},
    {{{{4, 1, 6}}, {{1, 1, 1, 4, 1, 6}}, {{1, 2, 3, 4, 5, 6}}}, {{1, 2, 3, 4, 5, 6}, {}, {ov::op::BroadcastType::PDPD, 3}}}
};

TEST(canonicalization, broadcast) {
    for (const auto& params : broadcast_shapes_with_params) {
        layout input0_layout = create_default_layout(std::get<0>(params.first)[0]);

        topology topology;
        topology.add(input_layout("input", input0_layout));
        topology.add(broadcast("broadcast", input_info("input"), params.second.target_shape,
                               params.second.axes_mapping, params.second.broadcast_mode));

        canonicalization_test(topology, "broadcast", std::get<1>(params.first), std::get<2>(params.first));
    }
}

std::vector<Shapes> eltwise_shapes {
    {{{2, 2, 3}, {2, 3}}, {{2, 2, 3, 1}, {1, 2, 3, 1}}, {{2, 2, 3, 1}}},
    {{{6}, {2, 3, 4, 5, 6}}, {{1, 1, 1, 1, 6}, {2, 3, 4, 5, 6}}, {{2, 3, 4, 5, 6}}}
};

TEST(canonicalization, eltwise) {
    for (const auto& shapes : eltwise_shapes) {
        layout input0_layout = create_default_layout(std::get<0>(shapes)[0]);
        layout input1_layout = create_default_layout(std::get<0>(shapes)[1]);

        topology topology;
        topology.add(input_layout("input0", input0_layout));
        topology.add(input_layout("input1", input1_layout));
        topology.add(eltwise("eltwise", { input_info("input0"), input_info("input1") }, eltwise_mode::sum));

        canonicalization_test(topology, "eltwise", std::get<1>(shapes), std::get<2>(shapes));
    }
}

std::vector<Shapes> fully_connected_shapes {
    {{{5, 2}, {5, 2}}, {{5, 2, 1, 1}, {5, 2, 1, 1}}, {{5, 5, 1, 1}}}
};

TEST(canonicalization, fully_connected) {
    auto& engine = get_test_engine();
    for (const auto& shapes : fully_connected_shapes) {
        layout input0_layout = create_default_layout(std::get<0>(shapes)[0]);
        auto weights_prim = engine.allocate_memory(create_default_layout(std::get<0>(shapes)[1]));

        size_t input_rank = input0_layout.get_partial_shape().size();
        size_t weights_rank = weights_prim->get_layout().get_partial_shape().size();

        topology topology;
        topology.add(input_layout("input", input0_layout));
        topology.add(data("weights", weights_prim));
        topology.add(fully_connected("fully_connected", input_info("input"), "weights", "", input_rank, weights_rank));

        canonicalization_test(topology, "fully_connected", std::get<1>(shapes), std::get<2>(shapes));
    }
}

std::vector<Shapes> gemm_shapes {
    {{{1, 5}, {5, 2}}, {{1, 1, 1, 5}, {1, 1, 5, 2}}, {{1, 1, 1, 2}}}
};

TEST(canonicalization, gemm) {
    for (const auto& shapes : gemm_shapes) {
        layout input0_layout = create_default_layout(std::get<0>(shapes)[0]);
        layout input1_layout = create_default_layout(std::get<0>(shapes)[1]);

        size_t input_rank = input0_layout.get_partial_shape().size();
        size_t weights_rank = input1_layout.get_partial_shape().size();

        topology topology;
        topology.add(input_layout("input0", input0_layout));
        topology.add(input_layout("input1", input1_layout));
        topology.add(gemm("gemm", {input_info("input0"), input_info("input1")},
                          data_types::f32, false, false, 1.0f, 0.0f, input_rank, weights_rank));

        canonicalization_test(topology, "gemm", std::get<1>(shapes), std::get<2>(shapes));
    }
}

struct gather_params {
    int64_t axis;
    int64_t batch_dim;
    bool support_neg_ind;
};

std::vector<std::pair<Shapes, gather_params>> gather_shapes_with_params {
    {
        {{{8, 2, 3}, {}}, {{8, 2, 3, 1}, {1, 1, 1, 1}}, {{1, 2, 3, 1}}},
        {0, 0, false}
    },
    {
        {{{8, -1, -1, 2}, {}}, {{8, -1, -1, 2}, {1, 1, 1, 1}}, {{1, -1, -1, 2}}},
        {0, 0, false}
    },
    {
        {{{8, 2, 3}, {1}}, {{8, 2, 3, 1}, {1, 1, 1, 1}}, {{1, 2, 3, 1}}},
        {0, 0, false}
    },
    {
        {{{8, 2, 3, 4}, {8}}, {{8, 2, 3, 4}, {8, 1, 1, 1}}, {{8, 2, 1, 4}}},
        {2, 1, false}
    }
};

TEST(canonicalization, gather) {
    for (const auto& params : gather_shapes_with_params) {
        layout data_layout = create_default_layout(std::get<0>(params.first)[0]);
        layout indices_layout = create_default_layout(std::get<0>(params.first)[1]);

        topology topology;
        topology.add(input_layout("data", data_layout));
        topology.add(input_layout("indices", indices_layout));
        topology.add(gather("gather", input_info("data"), input_info("indices"), params.second.axis,
                            0, ov::Shape{}, params.second.batch_dim, params.second.support_neg_ind));

        canonicalization_test(topology, "gather", std::get<1>(params.first), std::get<2>(params.first));
    }
}

struct fusing_gather_eltwise_params {
    ov::PartialShape data_shape;
    ov::Shape out_shape;
    int64_t axis;
    int64_t batch_dim;
    bool support_neg_ind;
};

std::vector<std::pair<Shapes, fusing_gather_eltwise_params>> fusing_gather_eltwise_shapes_with_params {
    {
        {{{}, {}}, {{4624, 4, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {4624, 1, 1, 1}}, {{4624, 1, 1, 1}}},
        {{4624, 4}, {4624}, 1, 0, true}
    }
};

TEST(canonicalization, fusing_gather_eltwise) {
    for (const auto& shapes : fusing_gather_eltwise_shapes_with_params) {
        layout input_gather_layout = create_default_layout(shapes.second.data_shape);
        layout indices_layout_first = create_default_layout(std::get<0>(shapes.first)[0]);
        layout indices_layout_second = create_default_layout(std::get<0>(shapes.first)[0]);
        layout input_mul_layout = create_default_layout(std::get<0>(shapes.first)[1]);

        topology topology;
        topology.add(input_layout("input", input_gather_layout));
        topology.add(input_layout("indices_first", indices_layout_first));
        topology.add(input_layout("indices_second", indices_layout_second));
        topology.add(input_layout("data", input_mul_layout));
        topology.add(gather("gather_first", input_info("input"), input_info("indices_first"), shapes.second.axis,
                            shapes.second.data_shape.rank().get_length(), shapes.second.out_shape, shapes.second.batch_dim, shapes.second.support_neg_ind));
        topology.add(gather("gather_second", input_info("input"), input_info("indices_second"), shapes.second.axis,
                            shapes.second.data_shape.rank().get_length(), shapes.second.out_shape, shapes.second.batch_dim, shapes.second.support_neg_ind));
        topology.add(eltwise("mul", {input_info("gather_first"), input_info("data")}, eltwise_mode::prod));
        topology.add(eltwise("add", {input_info("gather_second"), input_info("mul")}, eltwise_mode::sum));
        topology.add(reorder("out_reorder", input_info("add"), format::bfyx, data_types::f32));

        canonicalization_test(topology, "gather_first", std::get<1>(shapes.first), std::get<2>(shapes.first), true);
    }
}

struct fusing_gemm_eltwise_params {
    ov::PartialShape input_gemm_first;
    ov::PartialShape weights_gemm_first;
    ov::PartialShape input_gemm_second;
    ov::PartialShape weights_gemm_second;
};

std::vector<std::pair<Shapes, fusing_gemm_eltwise_params>> fusing_gemm_eltwise_shapes_with_params {
    {
        {{/* placeholder */}, {{1, 1, 1, 4, 4}}, {{1, 1, 1, 4, 4}}},
        {{1, 1, 1, 4, 5}, {1, 1, 1, 5, 4}, {1, 1, 4, 5}, {1, 1, 5, 4}}
    }
};

TEST(canonicalization, fusing_gemm_eltwise) {
    for (const auto& shapes : fusing_gemm_eltwise_shapes_with_params) {
        layout input_layout_first = create_default_layout(shapes.second.input_gemm_first);
        layout weights_layout_first = create_default_layout(shapes.second.weights_gemm_first);

        layout input_layout_second = create_default_layout(shapes.second.input_gemm_second);
        layout weights_layout_second = create_default_layout(shapes.second.weights_gemm_second);

        size_t input_rank_first = input_layout_first.get_partial_shape().size();
        size_t weights_rank_first = weights_layout_first.get_partial_shape().size();

        size_t input_rank_second = input_layout_second.get_partial_shape().size();
        size_t weights_rank_second = weights_layout_second.get_partial_shape().size();

        size_t out_rank = std::max(std::max(input_rank_first, weights_rank_first),
                                   std::max(input_rank_second, weights_rank_second));

        topology topology;
        topology.add(input_layout("input_first", input_layout_first));
        topology.add(input_layout("weights_first", weights_layout_first));
        topology.add(input_layout("input_second", input_layout_second));
        topology.add(input_layout("weights_second", weights_layout_second));

        topology.add(gemm("gemm_first", {input_info("input_first"), input_info("weights_first")},
                          data_types::f32, false, false, 1.0f, 0.0f, input_rank_first, weights_rank_first));

        topology.add(gemm("gemm_second", {input_info("input_second"), input_info("weights_second")},
                          data_types::f32, false, false, 1.0f, 0.0f, input_rank_second, weights_rank_second));

        topology.add(eltwise("sum", {input_info("gemm_first"), input_info("gemm_second")}, eltwise_mode::sum));
        topology.add(reorder("out_reorder", input_info("sum"), format::get_default_format(out_rank), data_types::f32));

        canonicalization_test(topology, "out_reorder", std::get<1>(shapes.first), std::get<2>(shapes.first), true);
    }
}

} // namespace
