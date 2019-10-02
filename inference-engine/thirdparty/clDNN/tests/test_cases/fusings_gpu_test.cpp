/*
// Copyright (c) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <gtest/gtest.h>
#include "api/memory.hpp"
#include "api/input_layout.hpp"
#include "api/convolution.hpp"
#include "api/quantize.hpp"
#include "api/topology.hpp"
#include "api/tensor.hpp"
#include "api/network.hpp"
#include "api/eltwise.hpp"
#include "api/fully_connected.hpp"
#include "api/binary_convolution.hpp"
#include "api/engine.hpp"
#include "api/data.hpp"

#include "test_utils/test_utils.h"

#include <cmath>

using namespace cldnn;
using namespace tests;

struct bc_test_params {
    tensor in_shape;
    tensor out_shape;
    tensor kernel;
    tensor stride;
    tensor pad;
    tensor dilation;
    uint32_t groups;
    data_types data_type;
    format input_format;
    data_types weights_type;
    format weights_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

class BaseFusingTest : public ::testing::TestWithParam<bc_test_params> {
public:
    cldnn::engine engine;
    cldnn::topology topology;
    cldnn::build_options bo_fused;
    cldnn::build_options bo_not_fused;

    float tolerance = 0.0f;

    static const int min_random = -200;
    static const int max_random = 200;

    void SetUp() override {
        bo_fused.set_option(build_option::optimize_data(true));
        bo_not_fused.set_option(build_option::optimize_data(false));
    }

    void execute(bc_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology, bo_not_fused);
        network network_fused(this->engine, this->topology, bo_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
    }

    void compare(const network& not_fused, const network& fused, bc_test_params& p) {
        auto outputs_ref = not_fused.execute();
        auto outputs_fused = fused.execute();

        ASSERT_EQ(fused.get_executed_primitives().size(), p.expected_fused_primitives);
        ASSERT_EQ(not_fused.get_executed_primitives().size(), p.expected_not_fused_primitives);
        ASSERT_EQ(outputs_ref.size(), outputs_fused.size());
        ASSERT_EQ(outputs_ref.size(), size_t(1));

        auto output_not_fused_prim = outputs_ref.begin()->second.get_memory();
        auto output_fused_prim = outputs_fused.begin()->second.get_memory();
        if (output_not_fused_prim.get_layout().data_type == data_types::f32) {
            auto ref = output_not_fused_prim.pointer<float>();
            auto output_ptr = output_fused_prim.pointer<float>();
            for (size_t i = 0; i < output_fused_prim.get_layout().count(); i++) {
                ASSERT_NEAR(ref[i], output_ptr[i], tolerance) << "i = " << i;
            }
        } else {
            auto ref = output_not_fused_prim.pointer<int16_t>();
            auto output_ptr = output_fused_prim.pointer<int16_t>();
            for (size_t i = 0; i < output_fused_prim.get_layout().count(); i++) {
                ASSERT_NEAR(float16_to_float32(ref[i]), float16_to_float32(output_ptr[i]), tolerance) << "i = " << i;
            }
        }
    }

    cldnn::memory get_mem(cldnn::layout l) {
        auto prim = memory::allocate(engine, l);
        tensor s = l.size;
        if (l.data_type == data_types::bin) {
            VF<int32_t> rnd_vec = generate_random_1d<int32_t>(s.count()/32, min_random, max_random);
            set_values(prim, rnd_vec);
        } else {
            VVVVF<float> rnd = generate_random_4d<float>(s.batch[0], s.feature[0], s.spatial[1], s.spatial[0],
                                                         min_random, max_random);
            VF<float> rnd_vec = flatten_4d<float>(format::bfyx, rnd);
            set_values(prim, rnd_vec);
        }

        return prim;
    }

    cldnn::memory get_mem(cldnn::layout l, float fill_value) {
        auto prim = memory::allocate(engine, l);
        tensor s = l.size;
        if (l.data_type == data_types::bin) {
            VF<int32_t> rnd_vec(s.count()/32, static_cast<int32_t>(fill_value));
            set_values(prim, rnd_vec);
        } else {
            VF<float> rnd_vec(s.count(), fill_value);
            set_values(prim, rnd_vec);
        }

        return prim;
    }

    cldnn::memory get_mem(cldnn::layout l, int min, int max) {
        auto prim = memory::allocate(engine, l);
        tensor s = l.size;
        if (l.data_type == data_types::f32) {
            VF<float> rnd_vec = generate_random_1d<float>(s.count(), min, max);
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::i8) {
            VF<int8_t> rnd_vec = generate_random_1d<int8_t>(s.count(), min, max);
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::bin) {
            VF<int32_t> rnd_vec = generate_random_1d<int32_t>(s.count()/32, min, max);
            set_values(prim, rnd_vec);
        }

        return prim;
    }

    layout get_input_layout(bc_test_params& p) {
        auto pad = p.pad.negate();
        std::vector<int> pad_ = { 0, 0, pad.spatial[0], pad.spatial[1] };
        return layout{p.data_type, p.input_format, p.in_shape, padding{pad_}};
    }

    layout get_output_layout(bc_test_params& p) {
        return layout{p.data_type, p.input_format, p.out_shape};
    }

    layout get_weights_layout(bc_test_params& p) {
        return layout{p.weights_type, p.weights_format, tensor{p.out_shape.feature[0],
                                                               static_cast<int32_t>(p.in_shape.feature[0] / p.groups),
                                                               p.kernel.spatial[0], p.kernel.spatial[1]}};
    }

    layout get_bias_layout(bc_test_params& p) {
        return layout{p.data_type, p.default_format, tensor{1, p.out_shape.feature[0], 1, 1}};
    }

    layout get_per_channel_layout(bc_test_params& p) {
        return layout{p.default_type, p.default_format, tensor{1, p.out_shape.feature[0], 1, 1}};
    }
    layout get_single_element_layout(bc_test_params& p) {
        return layout{p.default_type, p.default_format, tensor{1, 1, 1, 1}};
    }
};

#define CASE_CONV1 {1, 15, 4, 5}, {1, 30, 2, 3}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::bfyx, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV2 {1, 16, 4, 5}, {1, 32, 2, 3}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::bfyx_f16, data_types::f32, format::o_i_yx_i16_o16, data_types::f32, format::bfyx
#define CASE_CONV3 {1, 16, 4, 5}, {1, 32, 4, 5}, {1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::bfyx_f16, data_types::f32, format::o_i_yx_i16_o16, data_types::f32, format::bfyx
#define CASE_CONV4 {1, 32, 4, 5}, {1, 32, 4, 5}, {1, 1, 3, 3}, tensor{1}, tensor{0, 0, -1, -1, 0, 0}, tensor{1}, 32, data_types::f32, format::bfyx_f16, data_types::f32,  format::oiyx_o16, data_types::f32, format::bfyx

#define CASE_BIN_CONV1 {1, 16, 4, 5}, {1, 16, 4, 5}, {1, 1, 3, 3}, tensor{1}, tensor{0, 0, -1, -1, 0, 0}, tensor{1}, 1, data_types::bin, format::b_fs_yx_32fp, data_types::bin, format::os_is_yx_osv32_isv32p, data_types::f32, format::bfyx
#define CASE_BIN_CONV2 {1, 16, 4, 5}, {1, 30, 4, 5}, {1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::bin, format::b_fs_yx_32fp, data_types::bin, format::os_is_yx_osv32_isv32p, data_types::f32, format::bfyx
#define CASE_BIN_CONV3 {1, 184, 12, 21}, {1, 224, 12, 21}, {1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::bin, format::b_fs_yx_32fp, data_types::bin, format::os_is_yx_osv32_isv32p, data_types::f32, format::bfyx

class conv_activation : public BaseFusingTest {};
TEST_P(conv_activation, basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 activation("activation", "conv_prim", activation_func::abs),
                 reorder("reorder_bfyx", "activation", p.default_format, data_types::f32)
    );

    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_activation, ::testing::ValuesIn(std::vector<bc_test_params>{
                                                                           bc_test_params{CASE_CONV1, 3, 4},
                                                                           bc_test_params{CASE_CONV2, 3, 4},
                                                                           bc_test_params{CASE_CONV3, 3, 4},
                                                                           bc_test_params{CASE_CONV4, 3, 4},
}), );


class conv_fp32_scale : public BaseFusingTest {};
TEST_P(conv_fp32_scale, basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count())),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 scale("scale", "conv_prim", "scale_data"),
                 reorder("reorder_bfyx", "scale", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_fp32_scale,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                             bc_test_params{CASE_CONV1, 4, 4},  // doesn't support this fusing for now
                                             bc_test_params{CASE_CONV2, 3, 4},
                                             bc_test_params{CASE_CONV3, 3, 4},
                                             bc_test_params{CASE_CONV4, 3, 4},
                                             }), );

class conv_fp32_prelu_eltwise : public BaseFusingTest {};
TEST_P(conv_fp32_prelu_eltwise, basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("slope_data", get_mem(get_per_channel_layout(p))),
                 data("eltwise_data", get_mem(get_output_layout(p))),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 activation("activation", "conv_prim", "slope_data", activation_func::relu_negative_slope),
                 eltwise("eltwise", "activation", "eltwise_data", eltwise_mode::sum),
                 reorder("reorder_bfyx", "eltwise", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_fp32_prelu_eltwise,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                             bc_test_params{CASE_CONV1, 5, 5},  // doesn't support this fusing for now
                                             bc_test_params{CASE_CONV2, 3, 5},
                                             bc_test_params{CASE_CONV3, 3, 5},
                                             bc_test_params{CASE_CONV4, 3, 5},
                                             }), );

class conv_bin_activation : public BaseFusingTest {};
TEST_P(conv_bin_activation, basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p), -127, 127)),
                 binary_convolution("bin_conv_prim", "input", {"weights"}, p.stride, p.pad, p.dilation, p.out_shape, p.groups),
                 activation("activation", "bin_conv_prim", activation_func::relu),
                 reorder("reorder_bfyx", "activation", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_bin_activation,
                        ::testing::ValuesIn(std::vector<bc_test_params>{bc_test_params{CASE_BIN_CONV1, 3, 4},
                                            }), );

class conv_bin_scale_activation : public BaseFusingTest {};
TEST_P(conv_bin_scale_activation, basic) {
    auto p = GetParam();
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p), -127, 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count())),
                 binary_convolution("bin_conv_prim", "input", {"weights"}, p.stride, p.pad, p.dilation, p.out_shape, p.groups),
                 scale("scale", "bin_conv_prim", "scale_data"),
                 activation("activation", "scale", activation_func::relu),
                 reorder("reorder_bfyx", "activation", p.default_format, data_types::f32)
    );
    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_bin_scale_activation,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                            bc_test_params{CASE_BIN_CONV1, 3, 5},
                            bc_test_params{CASE_BIN_CONV2, 3, 5},
                                            }), );

class conv_bin_quantize_bin : public BaseFusingTest {};
TEST_P(conv_bin_quantize_bin, channel_wise_quantize) {
    auto p = GetParam();
    auto in_thresh = get_mem(get_per_channel_layout(p), min_random, max_random);
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p), -127, 127)),
                 data("in_lo", in_thresh),
                 data("in_hi", in_thresh),
                 data("out_lo", get_mem(get_per_channel_layout(p), -1)),
                 data("out_hi", get_mem(get_per_channel_layout(p),  1)),
                 binary_convolution("bin_conv_prim", "input", {"weights"}, p.stride, p.pad, p.dilation, p.out_shape, p.groups),
                 quantize("quantize_data", "bin_conv_prim", "in_lo", "in_hi", "out_lo", "out_hi", 2),
                 reorder("reorder_bfyx", "quantize_data", p.default_format, data_types::f32)
    );
    tolerance = 1e-5f;
    execute(p);
}

TEST_P(conv_bin_quantize_bin, blob_wise_quantize) {
    auto p = GetParam();
    auto in_thresh = get_mem(get_single_element_layout(p), min_random, max_random);
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p), -127, 127)),
                 data("in_lo", in_thresh),
                 data("in_hi", in_thresh),
                 data("out_lo", get_mem(get_single_element_layout(p), -1)),
                 data("out_hi", get_mem(get_single_element_layout(p), 1)),
                 binary_convolution("bin_conv_prim", "input", {"weights"}, p.stride, p.pad, p.dilation, p.out_shape, p.groups),
                 quantize("quantize_data", "bin_conv_prim", "in_lo", "in_hi", "out_lo", "out_hi", 2),
                 reorder("reorder_bfyx", "quantize_data", p.default_format, data_types::f32)
    );
    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_bin_quantize_bin,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                            bc_test_params{CASE_BIN_CONV1, 3, 4},
                            bc_test_params{CASE_BIN_CONV2, 3, 4},
                                            }), );

class conv_bin_scale_conv_dw : public BaseFusingTest {};
TEST_P(conv_bin_scale_conv_dw, dw_kernel_3x3_stride2) {
    auto p = GetParam();
    auto dw_weights_layout = layout{p.default_type, p.default_format, tensor{p.out_shape.feature[0],
                                                                             1, 3, 3}};

    auto dw_stride = tensor{1, 1, 2, 2};
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p), -127, 127)),
                 data("weights_dw", get_mem(dw_weights_layout, -127, 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1e-1f)),
                 binary_convolution("bin_conv_prim", "input", {"weights"}, p.stride, p.pad, p.dilation, p.out_shape, p.groups),
                 scale("scale", "bin_conv_prim", "scale_data"),
                 convolution("conv_dw", "scale", {"weights_dw"}, p.out_shape.feature[0], dw_stride, p.pad, p.dilation),
                 reorder("reorder_bfyx", "conv_dw", p.default_format, data_types::f32)
    );
    tolerance = 1e-5f;
    execute(p);
}

TEST_P(conv_bin_scale_conv_dw, dw_kernel_3x3_stride1) {
    auto p = GetParam();
    auto dw_weights_layout = layout{p.default_type, p.default_format, tensor{p.out_shape.feature[0],
                                                                             1, 3, 3}};

    auto dw_stride = tensor{1, 1, 1, 1};
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p), -127, 127)),
                 data("weights_dw", get_mem(dw_weights_layout, -127, 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1e-1f)),
                 binary_convolution("bin_conv_prim", "input", {"weights"}, p.stride, p.pad, p.dilation, p.out_shape, p.groups),
                 scale("scale", "bin_conv_prim", "scale_data"),
                 convolution("conv_dw", "scale", {"weights_dw"}, p.out_shape.feature[0], dw_stride, p.pad, p.dilation),
                 reorder("reorder_bfyx", "conv_dw", p.default_format, data_types::f32)
    );
    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_bin_scale_conv_dw,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                            bc_test_params{CASE_BIN_CONV2, 4, 5},
                            bc_test_params{CASE_BIN_CONV3, 4, 5},
                                            }), );

class conv_bin_scale_conv_dw_prelu : public BaseFusingTest {};
TEST_P(conv_bin_scale_conv_dw_prelu, dw_kernel_3x3_stride2) {
    auto p = GetParam();
    auto dw_weights_layout = layout{p.default_type, p.default_format, tensor{p.out_shape.feature[0],
                                                                             1, 3, 3}};

    auto dw_stride = tensor{1, 1, 2, 2};
    auto in_thresh = get_mem(get_per_channel_layout(p), min_random, max_random);
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p), -127, 127)),
                 data("weights_dw", get_mem(dw_weights_layout, -127, 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1e-1f)),
                 binary_convolution("bin_conv_prim", "input", {"weights"}, p.stride, p.pad, p.dilation, p.out_shape, p.groups),
                 scale("scale", "bin_conv_prim", "scale_data"),
                 convolution("conv_dw", "scale", {"weights_dw"}, p.out_shape.feature[0], dw_stride, p.pad, p.dilation),
                 data("slope_data", get_mem(get_per_channel_layout(p))),
                 activation("activation", "conv_dw", "slope_data", activation_func::relu_negative_slope),
                 reorder("reorder_bfyx", "activation", p.default_format, data_types::f32)
    );
    tolerance = 1e-5f;
    execute(p);
}

TEST_P(conv_bin_scale_conv_dw_prelu, dw_kernel_3x3_stride1) {
    auto p = GetParam();
    auto dw_weights_layout = layout{p.default_type, p.default_format, tensor{p.out_shape.feature[0],
                                                                             1, 3, 3}};

    auto dw_stride = tensor{1, 1, 1, 1};
    auto in_thresh = get_mem(get_per_channel_layout(p), min_random, max_random);
    topology.add(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p), -127, 127)),
                 data("weights_dw", get_mem(dw_weights_layout, -127, 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1e-1f)),
                 binary_convolution("bin_conv_prim", "input", {"weights"}, p.stride, p.pad, p.dilation, p.out_shape, p.groups),
                 scale("scale", "bin_conv_prim", "scale_data"),
                 convolution("conv_dw", "scale", {"weights_dw"}, p.out_shape.feature[0], dw_stride, p.pad, p.dilation),
                 data("slope_data", get_mem(get_per_channel_layout(p))),
                 activation("activation", "conv_dw", "slope_data", activation_func::relu_negative_slope),
                 reorder("reorder_bfyx", "activation", p.default_format, data_types::f32)
    );
    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_CASE_P(fusings_gpu, conv_bin_scale_conv_dw_prelu,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                            bc_test_params{CASE_BIN_CONV2, 4, 6},
                            bc_test_params{CASE_BIN_CONV3, 4, 6},
                                            }), );
