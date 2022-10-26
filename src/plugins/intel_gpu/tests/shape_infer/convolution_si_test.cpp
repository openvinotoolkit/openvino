// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/convolution.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "convolution_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct convolution_test_params {
    ov::PartialShape input_shape;
    ov::PartialShape weight_shape;
    uint32_t groups;
    ov::Strides stride;
    ov::Strides dilation;
    ov::CoordinateDiff pad_above;
    ov::CoordinateDiff pad_below;
    layout expected_layout;
    bool with_output_size;
    tensor output_size;
    bool weights_have_group_dim;
};

class convolution_si_test : public testing::TestWithParam<convolution_test_params> { };

TEST_P(convolution_si_test, shape_infer) {
    auto p = GetParam();
    auto& engine = get_test_engine();
    auto input_data_layout = layout{p.input_shape, data_types::f32, format::bfyx};
    auto weight_layout = layout{p.weight_shape, data_types::f32, format::bfyx};
    std::vector<cldnn::primitive_id> weights = {"weight"};
    std::vector<cldnn::primitive_id> bias = {};
    auto input_prim = std::make_shared<input_layout>("data", input_data_layout);
    auto weight_prim = std::make_shared<input_layout>("weight", weight_layout);
    std::shared_ptr<convolution> conv_prim = nullptr;
    if (p.with_output_size) {
        conv_prim = std::make_shared<convolution>("conv", "data", weights, bias, p.groups,
                                                  p.stride, p.pad_above, p.dilation, p.output_size,
                                                  data_types::f32, p.weights_have_group_dim);
    } else {
        conv_prim = std::make_shared<convolution>("conv", "data", weights, bias, p.groups, p.stride, p.pad_above, p.dilation, p.pad_above, p.pad_below);
    }
    
    cldnn::program prog(engine);

    auto& input_node = prog.get_or_create(input_prim);
    auto& weight_node = prog.get_or_create(weight_prim);
    auto& conv_node = prog.get_or_create(conv_prim);
    program_wrapper::add_connection(prog, input_node, conv_node);
    program_wrapper::add_connection(prog, weight_node, conv_node);

    auto params = conv_node.get_kernel_impl_params();
    auto res = convolution_inst::calc_output_layouts<ov::PartialShape>(conv_node, *params);

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_CASE_P(smoke_wo_out_size, convolution_si_test,
    testing::ValuesIn(std::vector<convolution_test_params>{
        // conv, symmetric pad
        {
            ov::PartialShape{1, 3, 10, 10}, ov::PartialShape{4, 3, 3, 3},          /* input, weight shape */
            1, {1, 1}, {1, 1},                                                     /* groups, stride, dilation */
            std::vector<ptrdiff_t>{1, 1}, std::vector<ptrdiff_t>{1, 1},            /* pad_above, pad_below */
            layout{ov::PartialShape{1, 4, 10, 10}, data_types::f32, format::bfyx}, /* expected_layout */
            false, tensor(0, 0, 0, 0),                                             /* predefined output size */   
            false                                                                  /* weight_has_groups */  
        },
        // conv, symmetric pad
        {
            ov::PartialShape{1, 3, 10, 10}, ov::PartialShape{4, 3, 3, 3},          /* input, weight shape */
            1, {1, 1}, {1, 1},                                                     /* groups, stride, dilation */
            std::vector<ptrdiff_t>{2, 1}, std::vector<ptrdiff_t>{2, 1},            /* pad_above, pad_below */
            layout{ov::PartialShape{1, 4, 12, 10}, data_types::f32, format::bfyx}, /* expected_layout */
            false, tensor(0, 0, 0, 0),                                             /* predefined output size */   
            false                                                                  /* weight_has_groups */  
        },
        // conv, asymmetric pad
        {
            ov::PartialShape{1, 3, 10, 10}, ov::PartialShape{4, 3, 3, 3},          /* input, weight shape */
            1, {1, 1}, {1, 1},                                                     /* groups, stride, dilation */
            std::vector<ptrdiff_t>{2, 1}, std::vector<ptrdiff_t>{1, 2},            /* pad_above, pad_below */
            layout{ov::PartialShape{1, 4, 11, 11}, data_types::f32, format::bfyx}, /* expected_layout */
            false, tensor(0, 0, 0, 0),                                             /* predefined output size */   
            false                                                                  /* weight_has_groups */  
        },
        // conv, dynamic shape
        {
            ov::PartialShape::dynamic(4), ov::PartialShape{4, 3, 3, 3},           /* input, weight shape */
            1, {1, 1}, {1, 1},                                                    /* groups, stride, dilation */
            std::vector<ptrdiff_t>{1, 2}, std::vector<ptrdiff_t>{2, 1},           /* pad_above, pad_below */
            layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx},  /* expected_layout */
            false, tensor(0, 0, 0, 0),                                            /* predefined output size */
            false                                                                 /* weight_has_groups */
        },
        // groupconv, symmetric pad
        {
            ov::PartialShape{1, 12, 224, 224}, ov::PartialShape{4, 1, 3, 5, 5},       /* input, weight shape */
            4, {1, 1}, {1, 1},                                                        /* groups, stride, dilation */
            std::vector<ptrdiff_t>{2, 2}, std::vector<ptrdiff_t>{2, 2},               /* pad_above, pad_below */
            layout{ov::PartialShape{1, 4, 224, 224}, data_types::f32, format::bfyx},  /* expected_layout */
            false, tensor(0, 0, 0, 0),                                                /* predefined output size */   
            true                                                                      /* weight_has_groups */  
        },
        // groupconv, asymmetric pad
        {
            ov::PartialShape{1, 12, 224, 224}, ov::PartialShape{4, 1, 3, 5, 5},       /* input, weight shape */
            4, {1, 1}, {1, 1},                                                        /* groups, stride, dilation */
            std::vector<ptrdiff_t>{2, 1}, std::vector<ptrdiff_t>{1, 2},               /* pad_above, pad_below */
            layout{ov::PartialShape{1, 4, 223, 223}, data_types::f32, format::bfyx},  /* expected_layout */
            false, tensor(0, 0, 0, 0),                                                /* predefined output size */   
            true                                                                      /* weight_has_groups */  
        },
    }));

INSTANTIATE_TEST_CASE_P(smoke_with_out_size, convolution_si_test,
    testing::ValuesIn(std::vector<convolution_test_params>{
        // conv, symmetric pad
        {
            ov::PartialShape{1, 3, 10, 10}, ov::PartialShape{4, 3, 3, 3},            /* input, weight shape */
            1, {1, 1}, {1, 1},                                                       /* groups, stride, dilation */
            std::vector<ptrdiff_t>{1, 1}, std::vector<ptrdiff_t>{1, 1},              /* pad_above, pad_below */
            layout{ov::PartialShape{1, 4, 10, 10}, data_types::f32, format::bfyx},   /* expected_layout */
            true, tensor(1, 4, 10, 10),                                              /* predefined output size */
            false                                                                    /* weight_has_groups */
        },
        // conv, asymmetric pad
        {
            ov::PartialShape{1, 3, 10, 10}, ov::PartialShape{4, 3, 3, 3},            /* input, weight shape */
            1, {1, 1}, {1, 1},                                                       /* groups, stride, dilation */
            std::vector<ptrdiff_t>{2, 1}, std::vector<ptrdiff_t>{1, 2},              /* pad_above, pad_below */
            layout{ov::PartialShape{1, 4, 12, 12}, data_types::f32, format::bfyx},   /* expected_layout */
            true, tensor(1, 4, 12, 12),                                              /* predefined output size */
            false                                                                    /* weight_has_groups */
        },
        // groupconv, asymmetric pad
        {
            ov::PartialShape{1, 12, 224, 224}, ov::PartialShape{4, 1, 3, 5, 5},       /* input, weight shape */
            4, {1, 1}, {1, 1},                                                        /* groups, stride, dilation */
            std::vector<ptrdiff_t>{2, 1}, std::vector<ptrdiff_t>{1, 2},               /* pad_above, pad_below */
            layout{ov::PartialShape{1, 4, 225, 225}, data_types::f32, format::bfyx},  /* expected_layout */
            true, tensor(1, 4, 225, 225),                                             /* predefined output size */   
            true                                                                      /* weight_has_groups */  
        },
    }));

}