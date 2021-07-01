// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <cldnn/primitives/input_layout.hpp>
#include <cldnn/primitives/convolution.hpp>
#include <cldnn/primitives/quantize.hpp>
#include <cldnn/primitives/eltwise.hpp>
#include <cldnn/primitives/fully_connected.hpp>
#include <cldnn/primitives/gemm.hpp>
#include <cldnn/primitives/binary_convolution.hpp>
#include <cldnn/primitives/data.hpp>
#include <cldnn/primitives/resample.hpp>
#include <cldnn/primitives/mvn.hpp>
#include <cldnn/primitives/deconvolution.hpp>
#include <cldnn/primitives/permute.hpp>
#include <cldnn/primitives/gather.hpp>
#include <cldnn/primitives/gather_nd.hpp>
#include <cldnn/primitives/scatter_update.hpp>
#include <cldnn/primitives/scatter_nd_update.hpp>
#include <cldnn/primitives/scatter_elements_update.hpp>
#include <cldnn/primitives/depth_to_space.hpp>
#include <cldnn/primitives/space_to_depth.hpp>
#include <cldnn/primitives/batch_to_space.hpp>
#include <cldnn/primitives/space_to_batch.hpp>
#include <cldnn/primitives/reduce.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

struct resample_test_params {
    tensor in_shape;
    tensor out_shape;
    data_types data_type;
    format input_format;
    resample_type type;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

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

struct bc_force_kernel_params {
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
    std::string kernel_name;
};

struct conv_eltw_test_params {
    tensor in_shape;
    tensor out_shape;
    tensor eltw_shape;
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

struct gemm_test_params {
    std::vector<tensor> in_shapes;
    tensor out_shape;
    tensor kernel;
    tensor pad;
    data_types data_type_in0;
    data_types data_type_in1;
    data_types data_type_in2;
    format input_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

struct normalize_test_params {
    tensor in_shape;
    data_types data_type;
    format input_format;
    data_types default_type;
    format default_format;
    bool across_spatial;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

template<typename T>
class BaseFusingTest : public ::testing::TestWithParam<T> {
public:
    cldnn::engine& engine = get_test_engine();
    cldnn::topology topology_fused;
    cldnn::topology topology_non_fused;
    cldnn::build_options bo_fused;
    cldnn::build_options bo_not_fused;

    float tolerance = 0.0f;

    static const int min_random = -200;
    static const int max_random = 200;

    void SetUp() override {
        bo_fused.set_option(build_option::optimize_data(true));
        bo_not_fused.set_option(build_option::optimize_data(false));
        bo_not_fused.set_option(build_option::allow_static_input_reorder(true));
    }

    void compare(network& not_fused, network& fused, T& p, bool count_reorder = false) {
        auto outputs_ref = not_fused.execute();
        auto outputs_fused = fused.execute();
        auto get_reorders_count = [](network& net) -> size_t {
            size_t count = 0;
            for (auto& pi : net.get_primitives_info()) {
                if (pi.type_id == "reorder") {
                    auto exec_prims = net.get_executed_primitives();
                    auto it = std::find_if(exec_prims.begin(), exec_prims.end(), [&](const std::pair<primitive_id, event::ptr>& e) -> bool {
                        return e.first == pi.original_id;
                    });
                    // We count executed reorders only
                    if (it != exec_prims.end())
                        count++;
                }
            }
            return count;
        };

        size_t reorders_count_fused = get_reorders_count(fused);
        size_t reorders_count_not_fused = get_reorders_count(not_fused);

        std::stringstream description;
        description << std::endl << "not fused: " << std::endl;
        for (auto i : not_fused.get_primitives_info()) {
            description << "  " << i.original_id << " " << i.kernel_id << std::endl;
        }
        description << "fused: " << std::endl;
        for (auto i : fused.get_primitives_info()) {
            description << "  " << i.original_id << " " << i.kernel_id << std::endl;
        }
        SCOPED_TRACE(description.str());
        // Subtract reorders count to handle execution in different layouts when input/output reorders can be added in the graph
        ASSERT_EQ(fused.get_executed_primitives().size() - (count_reorder ? 0 : reorders_count_fused), p.expected_fused_primitives);
        ASSERT_EQ(not_fused.get_executed_primitives().size() - (count_reorder ? 0 : reorders_count_not_fused), p.expected_not_fused_primitives);
        ASSERT_EQ(outputs_ref.size(), outputs_fused.size());
        ASSERT_EQ(outputs_ref.size(), size_t(1));

        auto output_not_fused_prim = outputs_ref.begin()->second.get_memory();
        auto output_fused_prim = outputs_fused.begin()->second.get_memory();
        if (output_not_fused_prim->get_layout().data_type == data_types::f32) {
            cldnn::mem_lock<float> ref(output_not_fused_prim, get_test_stream());
            cldnn::mem_lock<float> output_ptr(output_fused_prim, get_test_stream());
            for (size_t i = 0; i < output_fused_prim->get_layout().count(); i++) {
                ASSERT_NEAR(ref[i], output_ptr[i], tolerance) << "i = " << i;
            }
        } else {
            cldnn::mem_lock<int16_t> ref(output_not_fused_prim, get_test_stream());
            cldnn::mem_lock<int16_t> output_ptr(output_fused_prim, get_test_stream());
            for (size_t i = 0; i < output_fused_prim->get_layout().count(); i++) {
                ASSERT_NEAR(float16_to_float32(ref[i]), float16_to_float32(output_ptr[i]), tolerance) << "i = " << i;
            }
        }
    }

    cldnn::memory::ptr get_mem(cldnn::layout l) {
        auto prim = engine.allocate_memory(l);
        tensor s = l.size;
        if (l.data_type == data_types::bin) {
            VF<int32_t> rnd_vec = generate_random_1d<int32_t>(s.count() / 32, min_random, max_random);
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::i8 || l.data_type == data_types::u8) {
            VF<uint8_t> rnd_vec = generate_random_1d<uint8_t>(s.count(), min_random, max_random);
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::f16) {
            VF<uint16_t> rnd_vec = generate_random_1d<uint16_t>(s.count(), -1, 1);
            set_values(prim, rnd_vec);
        } else {
            VF<float> rnd_vec = generate_random_1d<float>(s.count(), -1, 1);
            set_values(prim, rnd_vec);
        }

        return prim;
    }

    cldnn::memory::ptr get_mem(cldnn::layout l, float fill_value) {
        auto prim = engine.allocate_memory(l);
        tensor s = l.size;
        if (l.data_type == data_types::bin) {
            VF<int32_t> rnd_vec(s.count() / 32, static_cast<int32_t>(fill_value));
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::f16) {
            VF<uint16_t> rnd_vec(s.count(), float32_to_float16(fill_value));
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::f32) {
            VF<float> rnd_vec(s.count(), fill_value);
            set_values(prim, rnd_vec);
        } else {
            throw std::runtime_error("get_mem: Unsupported precision");
        }

        return prim;
    }

    cldnn::memory::ptr get_repeatless_mem(cldnn::layout l, int min, int max) {
        auto prim = engine.allocate_memory(l);
        tensor s = l.size;
        if (l.data_type == data_types::f32) {
            VF<float> rnd_vec = generate_random_norepetitions_1d<float>(s.count(), min, max);
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::f16) {
            VF<FLOAT16> rnd_vec = generate_random_norepetitions_1d<FLOAT16>(s.count(), min, max);
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::i8) {
            VF<int8_t> rnd_vec = generate_random_norepetitions_1d<int8_t>(s.count(), min, max);
            set_values(prim, rnd_vec);
        }
        else if (l.data_type == data_types::bin) {
            VF<int32_t> rnd_vec = generate_random_norepetitions_1d<int32_t>(s.count(), min, max);
            set_values(prim, rnd_vec);
        }

        return prim;
    }

    cldnn::memory::ptr get_mem(cldnn::layout l, int min, int max) {
        auto prim = engine.allocate_memory(l);
        tensor s = l.size;
        if (l.data_type == data_types::f32) {
            VF<float> rnd_vec = generate_random_1d<float>(s.count(), min, max);
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::f16) {
            VF<FLOAT16> rnd_vec = generate_random_1d<FLOAT16>(s.count(), min, max);
            set_values(prim, rnd_vec);
        } else if (l.data_type == data_types::i8) {
            VF<int8_t> rnd_vec = generate_random_1d<int8_t>(s.count(), min, max);
            set_values(prim, rnd_vec);
        }
        else if (l.data_type == data_types::bin) {
            VF<int32_t> rnd_vec = generate_random_1d<int32_t>(s.count() / 32, min, max);
            set_values(prim, rnd_vec);
        }

        return prim;
    }

    layout get_output_layout(T& p) {
        return layout{ p.data_type, p.input_format, p.out_shape };
    }

    layout get_weights_layout(T& p, const int32_t /* split */ = 1) {
        cldnn::tensor weights_tensor;
        if (p.groups == 1) {
            weights_tensor = cldnn::tensor(batch(p.out_shape.feature[0]), feature(p.in_shape.feature[0]),
                                           spatial(p.kernel.spatial[0], p.kernel.spatial[1], p.kernel.spatial[2]));
        } else {
            weights_tensor = cldnn::tensor(group(p.groups), batch(p.out_shape.feature[0] / p.groups), feature(p.in_shape.feature[0] / p.groups),
                                           spatial(p.kernel.spatial[0], p.kernel.spatial[1], p.kernel.spatial[2]));
        }
        return layout{p.weights_type, p.weights_format, weights_tensor};
    }

    layout get_weights_layout(T& p, const int32_t /* split */, cldnn::format f) {
        cldnn::tensor weights_tensor;
        weights_tensor = cldnn::tensor(batch(p.out_shape.feature[0]), feature(static_cast<int32_t>(p.in_shape.feature[0] / p.groups)),
                                       spatial(p.kernel.spatial[0], p.kernel.spatial[1], p.kernel.spatial[2]));
        return layout{p.weights_type, f, weights_tensor};
    }

    layout get_bias_layout(T& p) {
        return layout{ p.default_type, format::bfyx, tensor{1, p.out_shape.feature[0], 1, 1} };
    }

    layout get_weights_zp_layout(T& p) {
        return layout{ p.weights_type, p.default_format, tensor{p.out_shape.feature[0], 1, 1, 1} };
    }

    layout get_activations_zp_layout(T& p) {
        return layout{ p.data_type, p.default_format, tensor{1, p.in_shape.feature[0], 1, 1} };
    }



    layout get_single_element_layout(T& p) {
        return layout{ p.default_type, p.default_format, tensor{1, 1, 1, 1} };
    }

    template <class... Args>
    void create_topologies(Args const&... args) {
        topology_fused.add(args...);
        topology_non_fused.add(args...);
    }
};

template <typename T>
class WeightsPrimitiveFusingTest : public ::BaseFusingTest<T> {
public:

    void execute(T& p) {
        auto input_prim = this->get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, this->bo_not_fused);
        network network_fused(this->engine, this->topology_fused, this->bo_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        this->compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(T& p) {
        auto pad = p.pad.negate();
        std::vector<int> pad_ = { 0, 0, pad.spatial[0], pad.spatial[1] };
        return layout{ p.data_type, p.input_format, p.in_shape, padding{pad_} };
    }

    layout get_per_channel_layout(T& p) {
        return layout{ p.default_type, p.default_format, tensor{1, p.out_shape.feature[0], 1, 1} };
    }

    size_t get_fc_output_dim_size(bc_test_params& p) {
        size_t size = 2;
        for (auto i : p.out_shape.spatial) {
            if (i > 1)
                size++;
        }
        return size;
    }

    layout get_fc_weights_layout(T& p) {
        cldnn::tensor weights_tensor;
        if (p.out_shape.spatial[1] > 1) {
            // 3d case
            weights_tensor = cldnn::tensor(p.kernel.batch[0], p.kernel.feature[0], 1, 1);
        }
        else {
            weights_tensor = cldnn::tensor(batch(p.out_shape.feature[0]), feature(p.in_shape.feature[0]),
                   spatial(p.kernel.spatial[0], p.kernel.spatial[1], p.kernel.spatial[2]));
        }
        return layout{p.weights_type, p.weights_format, weights_tensor};
    }

    layout get_fc_bias_layout(T& p) {
        if (p.out_shape.spatial[1] > 1) {
            // 3d case
            return layout{ p.default_type, format::bfyx, tensor{1, p.out_shape.spatial[1], 1, 1} };
        }
        else {
            return layout{ p.default_type, format::bfyx, tensor{1, p.out_shape.feature[0], 1, 1} };
        }
    }
};

class ResamplePrimitiveFusingTest : public ::BaseFusingTest<resample_test_params> {
public:

    void execute(resample_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(resample_test_params& p) {
        return layout{ p.data_type, p.input_format, p.in_shape, padding{} };
    }

    layout get_per_channel_layout(resample_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{1, p.out_shape.feature[0], 1, 1} };
    }
};

class GemmFusingTest : public ::BaseFusingTest<gemm_test_params> {
public:

    void execute(gemm_test_params& p) {
        auto input0_prim = get_mem(get_input_layout(p, 0));
        auto input1_prim = get_mem(get_input_layout(p, 1));

        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);
        network_fused.set_input_data("input0", input0_prim);
        network_not_fused.set_input_data("input0", input0_prim);
        network_fused.set_input_data("input1", input1_prim);
        network_not_fused.set_input_data("input1", input1_prim);
        if (p.in_shapes.size() > 2) {
            auto input2_prim = get_mem(get_input_layout(p, 2));
            network_fused.set_input_data("input2", input2_prim);
            network_not_fused.set_input_data("input2", input2_prim);
        }

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(gemm_test_params& p, int in_no) {
        auto pad = p.pad.negate();
        std::vector<int> pad_ = { 0, 0, pad.spatial[0], pad.spatial[1] };
        if (in_no == 0)
            return layout{ p.data_type_in0, p.input_format, p.in_shapes.at(0), padding{pad_} };
        else if (in_no == 1)
            return layout{ p.data_type_in1, p.input_format, p.in_shapes.at(1), padding{pad_} };
        else
            return layout{ p.data_type_in2, p.input_format, p.in_shapes.at(2), padding{pad_} };
    }

    layout get_per_channel_layout(gemm_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{1, p.in_shapes.at(0).feature[0], 1, 1} };
    }

    layout get_output_layout(gemm_test_params& p) {
        return layout{ p.default_type, p.input_format, p.out_shape };
    }
};

class ConvEltwTest : public ::BaseFusingTest<conv_eltw_test_params> {
public:

    void execute(conv_eltw_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
        auto find_prim = [](primitive_info& p) -> bool {
            // Add more ids when needed
            if (p.original_id == "deconv_prim")
                return true;
            return false;
        };

        auto pi_fused = network_fused.get_primitives_info();
        auto info_fused = std::find_if(pi_fused.begin(), pi_fused.end(), find_prim);
        if (info_fused != pi_fused.end())
            std::cout << "kernel: " << info_fused->kernel_id << std::endl;
    }

    layout get_input_layout(conv_eltw_test_params& p) {
        auto pad = p.pad.negate();
        std::vector<int> pad_ = { 0, 0, pad.spatial[0], pad.spatial[1] };
        return layout{ p.data_type, p.input_format, p.in_shape, padding{pad_} };
    }

    layout get_per_channel_layout(conv_eltw_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{1, p.out_shape.feature[0], 1, 1} };
    }
};

// in_shape; out_shape; kernel; stride; pad; dilation; groups; data_type; input_format; weights_type; weights_format; default_type; default_format;
#define CASE_CONV_FP32_1 {1, 15, 4, 5}, {1, 30, 2, 3}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::bfyx, data_types::f32, format::oiyx, data_types::f32, format::bfyx
#define CASE_CONV_FP32_2 {1, 16, 4, 5}, {1, 32, 2, 3}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::os_is_yx_isv16_osv16, data_types::f32, format::bfyx
#define CASE_CONV_FP32_3 {1, 16, 4, 5}, {1, 32, 4, 5}, {1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::os_is_yx_isv16_osv16, data_types::f32, format::bfyx
#define CASE_CONV_FP32_4 {1, 32, 4, 5}, {1, 32, 4, 5}, {1, 1, 3, 3}, tensor{1}, tensor{0, 0, -1, -1, 0, 0}, tensor{1}, 32, data_types::f32, format::b_fs_yx_fsv16, data_types::f32,  format::gs_oiyx_gsv16, data_types::f32, format::bfyx
#define CASE_CONV_FP32_5 {1, 15, 4, 5}, {1, 30, 2, 3}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_FP32_6 {1, 16, 4, 5, 4}, {1, 16, 2, 3, 2}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_CONV_FP32_7 {1, 16, 4, 5, 4}, {1, 32, 2, 3, 2}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_CONV_FP32_8 {1, 32, 4, 5, 4}, {1, 16, 2, 3, 2}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 2, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::g_os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_CONV_FP32_9 {1, 32, 4, 5, 4}, {1, 32, 2, 3, 2}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 2, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::g_os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_CONV_FP32_10 {32, 16, 4, 5, 4}, {32, 32, 4, 5, 4}, {1, 1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::bs_fs_zyx_bsv16_fsv16, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_CONV_FP32_11 {1, 32, 4, 5, 4}, {1, 16, 2, 3, 2}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 2, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::g_os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_CONV_FP32_12 {1, 16, 4, 5, 4}, {1, 16, 2, 3, 2}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 2, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::g_os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_CONV_FP32_13 {1, 16, 18, 5, 4}, {1, 16, 16, 3, 2}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 2, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::g_os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_CONV_FP32_14 {1, 3, 4, 5}, {1, 30, 2, 3}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::bfyx, data_types::f32, format::bfyx, data_types::f32, format::bfyx


#define CASE_CONV_FP16_1 {1, 15, 4, 5}, {1, 30, 2, 3}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f16, format::bfyx, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_CONV_FP16_2 {1, 16, 4, 5}, {1, 32, 2, 3}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f16, format::b_fs_yx_fsv16, data_types::f16, format::os_is_yx_isv16_osv16, data_types::f16, format::bfyx
#define CASE_CONV_FP16_3 {1, 16, 4, 5}, {1, 32, 4, 5}, {1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f16, format::b_fs_yx_fsv16, data_types::f16, format::os_is_yx_isv16_osv16, data_types::f16, format::bfyx
#define CASE_CONV_FP16_4 {1, 32, 4, 5}, {1, 32, 4, 5}, {1, 1, 3, 3}, tensor{1}, tensor{0, 0, -1, -1, 0, 0}, tensor{1}, 32, data_types::f16, format::b_fs_yx_fsv16, data_types::f16,  format::gs_oiyx_gsv16, data_types::f16, format::bfyx
#define CASE_CONV_FP16_5 {1, 15, 4, 5}, {1, 30, 2, 3}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f16, format::bfyx, data_types::i8, format::bfyx, data_types::f16, format::bfyx
#define CASE_CONV_FP16_6 {1, 16, 4, 5, 4}, {1, 16, 2, 3, 2}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::os_is_zyx_isv16_osv16, data_types::f16, format::bfzyx
#define CASE_CONV_FP16_7 {1, 16, 4, 5, 4}, {1, 32, 2, 3, 2}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::os_is_zyx_isv16_osv16, data_types::f16, format::bfzyx
#define CASE_CONV_FP16_8 {1, 32, 4, 5, 4}, {1, 16, 2, 3, 2}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 2, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::g_os_is_zyx_isv16_osv16, data_types::f16, format::bfzyx
#define CASE_CONV_FP16_9 {1, 32, 4, 5, 4}, {1, 32, 2, 3, 2}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 2, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::g_os_is_zyx_isv16_osv16, data_types::f16, format::bfzyx
#define CASE_CONV_FP16_10 {32, 16, 4, 5, 4}, {32, 32, 2, 3, 2}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f16, format::bs_fs_zyx_bsv16_fsv16, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_CONV_FP16_11 {1, 32, 4, 5, 4}, {1, 16, 2, 3, 2}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 2, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::g_os_is_zyx_isv16_osv16, data_types::f16, format::bfzyx
#define CASE_CONV_FP16_12 {1, 16, 4, 5, 4}, {1, 16, 2, 3, 2}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 2, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::g_os_is_zyx_isv16_osv16, data_types::f16, format::bfzyx
#define CASE_CONV_FP16_13 {16, 32, 4, 5}, {16, 64, 2, 3}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f16, format::fs_b_yx_fsv32, data_types::f16, format::bfyx, data_types::f16, format::bfyx

#define CASE_CONV_U8S8_1 {1, 15, 4, 5}, {1, 30, 2, 3}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_2 {1, 15, 5, 5}, {1, 30, 3, 3}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_3 {1, 16, 4, 5}, {1, 32, 4, 5}, {1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_4 {1, 17, 4, 5}, {1, 17, 4, 5}, {1, 1, 3, 3}, tensor{1}, tensor{0, 0, -1, -1, 0, 0}, tensor{1}, 17, data_types::u8, format::bfyx, data_types::i8, format::goiyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_5 {1, 16, 5, 5}, {1, 32, 5, 5}, {1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_6 {1, 17, 4, 5}, {1, 17, 4, 5}, {1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 17, data_types::u8, format::bfyx, data_types::i8, format::goiyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_7 {1, 64, 7, 7}, {1, 32, 7, 7}, {1, 1, 3, 3}, tensor{1}, tensor{0, 0, -1, -1, 0, 0}, tensor{1}, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_8 {1, 3, 4, 5}, {1, 32, 4, 5}, {1, 1, 3, 3}, tensor{1}, tensor{0, 0, -1, -1, 0, 0}, tensor{1}, 1, data_types::u8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_9 {16, 32, 5, 5}, {16, 32, 3, 3}, {1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::bs_fs_yx_bsv16_fsv16, data_types::i8, format::os_is_yx_osv16_isv16, data_types::f32, format::bfyx
#define CASE_CONV_U8S8_10 {16, 32, 5, 5}, {16, 32, 3, 3}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::bs_fs_yx_bsv16_fsv16, data_types::i8, format::os_is_yx_osv16_isv16, data_types::f32, format::bfyx

#define CASE_CONV_S8S8_1 {1, 15, 4, 5}, {1, 30, 2, 3}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_2 {1, 15, 5, 5}, {1, 30, 3, 3}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_3 {1, 16, 4, 5}, {1, 32, 4, 5}, {1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_4 {1, 17, 4, 5}, {1, 17, 4, 5}, {1, 1, 3, 3}, tensor{1}, tensor{0, 0, -1, -1, 0, 0}, tensor{1}, 17, data_types::i8, format::bfyx, data_types::i8, format::goiyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_5 {1, 16, 5, 5}, {1, 32, 5, 5}, {1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_6 {1, 17, 4, 5}, {1, 17, 4, 5}, {1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 17, data_types::i8, format::bfyx, data_types::i8, format::goiyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_7  {1, 64, 7, 7}, {1, 32, 7, 7}, {1, 1, 3, 3}, tensor{1}, tensor{0, 0, -1, -1, 0, 0}, tensor{1}, 1, data_types::i8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_8 {1, 3, 4, 5}, {1, 32, 4, 5}, {1, 1, 3, 3}, tensor{1}, tensor{0, 0, -1, -1, 0, 0}, tensor{1}, 1, data_types::i8, format::bfyx, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_9 {16, 32, 5, 5}, {16, 32, 3, 3}, {1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::bs_fs_yx_bsv16_fsv16, data_types::i8, format::os_is_yx_osv16_isv16, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_10 {16, 32, 5, 5}, {16, 32, 3, 3}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::bs_fs_yx_bsv16_fsv16, data_types::i8, format::os_is_yx_osv16_isv16, data_types::f32, format::bfyx
#define CASE_CONV_S8S8_11 {1, 4, 1280, 720}, {1, 4, 1280, 720}, {1, 1, 5, 5}, tensor{1}, tensor{0, 0, -2, -2}, tensor{1}, 1, data_types::i8, format::b_fs_yx_fsv4, data_types::i8, format::os_is_yx_osv16_isv4, data_types::f32, format::bfyx

#define CASE_CONV3D_U8S8_1 {1, 15, 5, 4, 5}, {1, 30, 3, 2, 3}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_U8S8_2 {1, 15, 5, 5, 5}, {1, 30, 3, 3, 3}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_U8S8_3 {1, 16, 5, 4, 5}, {1, 32, 5, 4, 5}, {1, 1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_U8S8_4 {1, 17, 5, 4, 5}, {1, 17, 5, 4, 5}, {1, 1, 3, 3, 3}, tensor{1}, tensor{{0, 0, -1, -1, -1}, 0}, tensor{1}, 17, data_types::u8, format::bfzyx, data_types::i8, format::goizyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_U8S8_5 {1, 3, 5, 4, 5},  {1, 32, 5, 4, 5}, {1, 1, 3, 3, 3}, tensor{1}, tensor{{0, 0, -1, -1, -1}, 0}, tensor{1}, 1, data_types::u8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx

#define CASE_CONV3D_S8S8_1 {1, 15, 5, 4, 5}, {1, 30, 3, 2, 3}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_S8S8_2 {1, 15, 5, 5, 5}, {1, 30, 3, 3, 3}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_S8S8_3 {1, 16, 5, 4, 5}, {1, 32, 5, 4, 5}, {1, 1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_S8S8_4 {1, 17, 5, 4, 5}, {1, 17, 5, 4, 5}, {1, 1, 3, 3, 3}, tensor{1}, tensor{{0, 0, -1, -1, -1}, 0}, tensor{1}, 17, data_types::i8, format::bfzyx, data_types::i8, format::goizyx, data_types::f32, format::bfzyx
#define CASE_CONV3D_S8S8_5 {1, 3, 5, 4, 5},  {1, 18, 5, 4, 5}, {1, 1, 3, 3, 3}, tensor{1}, tensor{{0, 0, -1, -1, -1}, 0}, tensor{1}, 1, data_types::i8, format::bfzyx, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx

// in_shape; out_shape; eltw_shape; kernel; stride; pad; dilation; groups; data_type; input_format; weights_type; weights_format; default_type; default_format;
#define CASE_CONV_ELTW_FP32_1 {1, 16, 4, 5}, {1, 32, 2, 3}, {1, 32, 1, 1}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::oiyx, data_types::f32, format::bfyx
#define CASE_CONV_ELTW_FP32_2 {1, 16, 4, 5}, {1, 32, 2, 3}, {1, 1, 1, 1}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::os_is_yx_isv16_osv16, data_types::f32, format::bfyx
#define CASE_CONV_ELTW_FP32_3 {1, 16, 4, 5}, {1, 32, 4, 5}, {1, 32, 4, 5}, {1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::os_is_yx_isv16_osv16, data_types::f32, format::bfyx
#define CASE_CONV_ELTW_FP32_4 {1, 32, 4, 5}, {1, 32, 4, 5}, {1, 32, 1, 1}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 32, data_types::f32, format::b_fs_yx_fsv16, data_types::f32,  format::gs_oiyx_gsv16, data_types::f32, format::bfyx
#define CASE_CONV_ELTW_FP32_5 {1, 32, 4, 5, 4}, {1, 32, 2, 3, 2}, {1, 32, 2, 1, 1}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 2, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::g_os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_CONV_ELTW_FP32_6 {1, 32, 4, 5, 4}, {1, 16, 2, 3, 2}, {1, 16, 2, 1, 1}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 2, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::g_os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_CONV_ELTW_FP32_7 {1, 16, 3, 5}, {1, 32, 1, 3}, {1, 32, 3, 1}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::os_is_yx_isv16_osv16, data_types::f32, format::bfyx
#define CASE_CONV_ELTW_FP32_8 {1, 32, 3, 5, 4}, {1, 16, 1, 3, 2}, {1, 1, 2, 1, 1}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 2, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::g_os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx

#define CASE_CONV_ELTW_i8_1 {1, 16, 3, 5}, {1, 32, 1, 3}, {1, 32, 3, 1}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::b_fs_yx_fsv16, data_types::i8, format::os_is_yx_osv16_isv16, data_types::f32, format::bfyx
#define CASE_CONV_ELTW_i8_2 {1, 16, 3, 5, 3}, {1, 32, 2, 4, 2}, {1, 1, 2, 4, 2}, {1, 1, 2, 2, 2}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::bfzyx, data_types::i8, format::oiyx, data_types::f32, format::bfzyx
#define CASE_CONV_ELTW_i8_3 {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::bfzyx, data_types::i8, format::oiyx, data_types::f32, format::bfzyx
#define CASE_CONV_ELTW_i8_4 {1, 16, 1, 4}, {1, 16, 1, 2}, {1, 16, 1, 1}, {1, 1, 1, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::b_fs_yx_fsv16, data_types::i8, format::os_is_yx_osv16_isv16, data_types::f32, format::bfyx
#define CASE_CONV_ELTW_i8_5 {1, 16, 1, 4, 1}, {1, 16, 1, 2, 1}, {1, 16, 2, 1, 1}, {1, 1, 1, 3, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::bfzyx, data_types::i8, format::oiyx, data_types::f32, format::bfzyx

#define CASE_BIN_CONV1 {1, 16, 4, 5}, {1, 16, 4, 5}, {1, 1, 3, 3}, tensor{1}, tensor{0, 0, -1, -1, 0, 0}, tensor{1}, 1, data_types::bin, format::b_fs_yx_32fp, data_types::bin, format::os_is_yx_osv32_isv32p, data_types::f32, format::bfyx
#define CASE_BIN_CONV2 {1, 16, 4, 5}, {1, 30, 4, 5}, {1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::bin, format::b_fs_yx_32fp, data_types::bin, format::os_is_yx_osv32_isv32p, data_types::f32, format::bfyx
#define CASE_BIN_CONV3 {1, 184, 12, 21}, {1, 224, 12, 21}, {1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::bin, format::b_fs_yx_32fp, data_types::bin, format::os_is_yx_osv32_isv32p, data_types::f32, format::bfyx

#define CASE_FC_FP32_1 {1, 1, 3, 1}, {1, 4, 1, 1}, {4, 1, 3, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::bfyx, data_types::f32, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP32_2 {2, 1, 3, 1}, {2, 4, 1, 1}, {4, 1, 3, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::yxfb, data_types::f32, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP32_3 {2, 32, 1, 1}, {2, 16, 1, 1}, {16, 32, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_FP32_3D_1 {5, 3, 1, 3}, {5, 3, 1, 5}, {5, 3, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::bfyx, data_types::f32, format::os_iyx_osv16, data_types::f32, format::bfyx
#define CASE_FC_FP32_3D_2 {2, 1, 1, 1}, {2, 1, 1, 32}, {32, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::bfyx, data_types::f32, format::os_iyx_osv16, data_types::f32, format::bfyx
#define CASE_FC_FP32_3D_3 {2, 32, 1, 32}, {2, 32, 1, 16}, {16, 32, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::bfyx, data_types::f32, format::os_iyx_osv16, data_types::f32, format::bfyx

#define CASE_FC_U8S8_1 {1, 1, 3, 1}, {1, 4, 1, 1}, {4, 1, 3, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_2 {2, 1, 3, 1}, {2, 4, 1, 1}, {4, 1, 3, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::b_fs_yx_fsv4, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3 {2, 32, 1, 1}, {2, 16, 1, 1}, {16, 32, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::b_fs_yx_fsv4, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3D_1 {2, 32, 1, 3}, {2, 32, 1, 16}, {16, 3, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3D_2 {1, 1, 1, 3}, {1, 1, 1, 32}, {32, 3, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_FC_U8S8_3D_3 {2, 3, 1, 1}, {2, 3, 1, 15}, {15, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx

#define CASE_NORMALIZE_I8_1 {1, 2, 3, 3}, data_types::u8, format::bfyx, data_types::f32, format::bfyx

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- FP32 convolution cases ------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
/* ----------- NOTE: A part of tests is disabled until all FP kernels don't support fusings ------------ */
class ConvFusingTest : public WeightsPrimitiveFusingTest<bc_test_params> {
public:
    void execute(bc_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
        auto find_conv = [](primitive_info& p) -> bool {
            if (p.original_id == "conv_prim")
                return true;
            return false;
        };

        auto pi_fused = network_fused.get_primitives_info();
        auto info_fused = std::find_if(pi_fused.begin(), pi_fused.end(), find_conv);
        if (info_fused != pi_fused.end())
            std::cout << "kernel: " << info_fused->kernel_id << std::endl;
    }
};

class conv_fp32_activation : public ConvFusingTest {};
TEST_P(conv_fp32_activation, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 activation("activation", "conv_prim", activation_func::abs),
                 reorder("reorder_bfyx", "activation", p.default_format, data_types::f32)
    );

    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_activation, ::testing::ValuesIn(std::vector<bc_test_params>{
                                                                           bc_test_params{CASE_CONV_FP32_1, 2, 3},
                                                                           bc_test_params{CASE_CONV_FP32_2, 2, 3},
                                                                           bc_test_params{CASE_CONV_FP32_3, 2, 3},
                                                                           bc_test_params{CASE_CONV_FP32_4, 2, 3},

                                                                           bc_test_params{CASE_CONV_FP16_4, 2, 3},
                                                                           bc_test_params{CASE_CONV_FP16_4, 2, 3},
                                                                           bc_test_params{CASE_CONV_FP16_4, 2, 3},
                                                                           bc_test_params{CASE_CONV_FP16_4, 2, 3},
}));


class conv_fp32_scale : public ConvFusingTest {};
TEST_P(conv_fp32_scale, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
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

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_scale,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                             // bc_test_params{CASE_CONV_FP32_1, 2, 3},
                                             bc_test_params{CASE_CONV_FP32_2, 2, 3},
                                             bc_test_params{CASE_CONV_FP32_3, 2, 3},
                                             bc_test_params{CASE_CONV_FP32_4, 2, 3},
                                             bc_test_params{CASE_CONV_FP32_10, 2, 3},

                                             // bc_test_params{CASE_CONV_FP16_1, 2, 3},
                                             bc_test_params{CASE_CONV_FP16_2, 2, 3},
                                             bc_test_params{CASE_CONV_FP16_3, 2, 3},
                                             bc_test_params{CASE_CONV_FP16_4, 2, 3},
                                             bc_test_params{CASE_CONV_FP16_10, 2, 3},
                                             }));

class conv_fp32_bias : public ConvFusingTest {};
TEST_P(conv_fp32_bias, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 convolution("conv_prim", "input", {"weights"}, std::vector<primitive_id>{}, p.groups, p.stride, p.pad, p.dilation),
                 eltwise("add_bias", {"conv_prim", "bias"}, eltwise_mode::sum),
                 reorder("reorder_bfyx", "add_bias", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_bias,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                             bc_test_params{CASE_CONV_FP32_1, 2, 3},
                                             bc_test_params{CASE_CONV_FP32_2, 2, 3},
                                             bc_test_params{CASE_CONV_FP32_3, 2, 3},
                                             bc_test_params{CASE_CONV_FP32_4, 2, 3},
                                             bc_test_params{CASE_CONV_FP32_10, 2, 3},

                                             bc_test_params{CASE_CONV_FP16_1, 2, 3},
                                             bc_test_params{CASE_CONV_FP16_2, 2, 3},
                                             bc_test_params{CASE_CONV_FP16_3, 2, 3},
                                             bc_test_params{CASE_CONV_FP16_4, 2, 3},
                                             bc_test_params{CASE_CONV_FP16_10, 2, 3},
                                             }));

class conv_fp32_prelu_eltwise : public ConvFusingTest {};
TEST_P(conv_fp32_prelu_eltwise, basic_sum) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
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

TEST_P(conv_fp32_prelu_eltwise, basic_prod) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("slope_data", get_mem(get_per_channel_layout(p))),
                 data("eltwise_data", get_mem(get_output_layout(p))),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 activation("activation", "conv_prim", "slope_data", activation_func::relu_negative_slope),
                 eltwise("eltwise", "activation", "eltwise_data", eltwise_mode::prod),
                 reorder("reorder_bfyx", "eltwise", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(conv_fp32_prelu_eltwise, eltw_broadcast_sum) {
    auto p = GetParam();
    tensor eltw_shape = p.default_format.spatial_num() == 2 ? tensor{1, 1, 1, 1} : tensor{1, 1, 1, 1, 1};
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("slope_data", get_mem(get_per_channel_layout(p))),
                 data("eltwise_data", get_mem(layout{ p.data_type, p.input_format, eltw_shape })),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 activation("activation", "conv_prim", "slope_data", activation_func::relu_negative_slope),
                 eltwise("eltwise", "activation", "eltwise_data", eltwise_mode::sum),
                 reorder("reorder_bfyx", "eltwise", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(conv_fp32_prelu_eltwise, eltw_broadcast_prod) {
    auto p = GetParam();
    tensor eltw_shape = p.default_format.spatial_num() == 2 ? tensor{1, 1, 1, 1} : tensor{1, 1, 1, 1, 1};
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("slope_data", get_mem(get_per_channel_layout(p))),
                 data("eltwise_data", get_mem(layout{ p.data_type, p.input_format, eltw_shape })),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 activation("activation", "conv_prim", "slope_data", activation_func::relu_negative_slope),
                 eltwise("eltwise", "activation", "eltwise_data", eltwise_mode::prod),
                 reorder("reorder_bfyx", "eltwise", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(conv_fp32_prelu_eltwise, vector_ops) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("slope_data", get_mem(get_per_channel_layout(p))),
                 data("eltwise_data", get_mem(get_output_layout(p))),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 activation("activation", "conv_prim", "slope_data", activation_func::relu_negative_slope),
                 eltwise("eltwise", "activation", "eltwise_data", eltwise_mode::sum),
                 reorder("reorder_bfyx", "eltwise", p.default_format, data_types::f32)
    );

    implementation_desc conv_impl = { format::b_fs_yx_fsv16, "" };
    bo_fused.set_option(build_option::force_implementations({ {"conv_prim", conv_impl} }));

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(conv_fp32_prelu_eltwise, vector_ops_mixed_types) {
    auto p = GetParam();
    auto slope_type = p.default_type == data_types::f32 ? data_types::f16 : data_types::f32;
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("slope_data", get_mem(layout{ slope_type, p.default_format, tensor{1, p.out_shape.feature[0], 1, 1} })),
                 data("eltwise_data", get_mem(get_output_layout(p))),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 activation("activation", "conv_prim", "slope_data", activation_func::relu_negative_slope),
                 eltwise("eltwise", "activation", "eltwise_data", eltwise_mode::sum),
                 reorder("reorder_bfyx", "eltwise", p.default_format, data_types::f32)
    );

    implementation_desc conv_impl = { format::b_fs_yx_fsv16, "" };
    bo_fused.set_option(build_option::force_implementations({ {"conv_prim", conv_impl} }));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_prelu_eltwise,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                             // bc_test_params{CASE_CONV_FP32_1, 2, 4},
                                             bc_test_params{CASE_CONV_FP32_2, 2, 4},
                                             bc_test_params{CASE_CONV_FP32_3, 2, 4},
                                             bc_test_params{CASE_CONV_FP32_4, 2, 4},

                                             // bc_test_params{CASE_CONV_FP32_1, 2, 4},
                                             bc_test_params{CASE_CONV_FP16_2, 2, 4},
                                             bc_test_params{CASE_CONV_FP16_3, 2, 4},
                                             bc_test_params{CASE_CONV_FP16_4, 2, 4},
                                             }));

class conv_fp32_multi_eltwise_2 : public ConvFusingTest {};
TEST_P(conv_fp32_multi_eltwise_2, basic) {
    if (engine.get_device_info().supports_immad) {
        return;
    }

    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
        data("eltwise_data", get_mem(get_output_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        data("weights", get_mem(get_weights_layout(p))),
        convolution("conv_prim", "input", { "weights" }, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
        eltwise("eltwise1", "conv_prim", "eltwise_data", eltwise_mode::sum),
        eltwise("eltwise2", "eltwise1", "conv_prim", eltwise_mode::prod),
        reorder("reorder_bfyx", "eltwise2", p.default_format, data_types::f32)
    );
    implementation_desc conv_impl = { format::b_fs_yx_fsv16, ""};
    bo_fused.set_option(build_option::force_implementations({ {"conv_prim", conv_impl} }));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_multi_eltwise_2,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                             bc_test_params{CASE_CONV_FP32_2, 2, 4},
                                             bc_test_params{CASE_CONV_FP32_3, 2, 4},
                                             bc_test_params{CASE_CONV_FP32_4, 2, 4},

                                             bc_test_params{CASE_CONV_FP16_2, 2, 4},
                                             bc_test_params{CASE_CONV_FP16_3, 2, 4},
                                             bc_test_params{CASE_CONV_FP16_4, 2, 4},
                                             }));


class conv_fp32_multi_eltwise_2_clamp : public ConvFusingTest {};
TEST_P(conv_fp32_multi_eltwise_2_clamp, basic) {
    if (engine.get_device_info().supports_immad) {
        return;
    }

    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
        data("eltwise1_data", get_mem(get_output_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        data("weights", get_mem(get_weights_layout(p))),
        convolution("conv_prim", "input", { "weights" }, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
        eltwise("eltwise1", "conv_prim", "eltwise1_data", eltwise_mode::sum),
        activation("activation", "eltwise1", activation_func::clamp, {0.5f, 2.5f}),
        eltwise("eltwise2", "activation", "conv_prim", eltwise_mode::prod),
        reorder("reorder_bfyx", "eltwise2", p.default_format, data_types::f32)
    );
    implementation_desc conv_impl = { format::b_fs_yx_fsv16, ""};
    bo_fused.set_option(build_option::force_implementations({ {"conv_prim", conv_impl} }));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_multi_eltwise_2_clamp,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                             bc_test_params{CASE_CONV_FP32_2, 2, 5},
                                             bc_test_params{CASE_CONV_FP32_3, 2, 5},
                                             bc_test_params{CASE_CONV_FP32_4, 2, 5},

                                             bc_test_params{CASE_CONV_FP16_2, 2, 5},
                                             bc_test_params{CASE_CONV_FP16_3, 2, 5},
                                             bc_test_params{CASE_CONV_FP16_4, 2, 5},
                                             }));


class conv_fp32_multi_eltwise_4_clamp : public ConvFusingTest {};
TEST_P(conv_fp32_multi_eltwise_4_clamp, basic) {
    if (engine.get_device_info().supports_immad) {
        return;
    }

    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
        data("eltwise1_data", get_mem(get_output_layout(p))),
        data("eltwise2_data", get_mem(get_output_layout(p))),
        data("eltwise4_data", get_mem(get_output_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        data("weights", get_mem(get_weights_layout(p))),
        convolution("conv_prim", "input", { "weights" }, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
        eltwise("eltwise1_add", "conv_prim", "eltwise1_data", eltwise_mode::sum),
        activation("activation", "eltwise1_add", activation_func::clamp, {0.5f, 2.5f}),
        eltwise("eltwise2_mul", "activation", "conv_prim", eltwise_mode::prod),
        eltwise("eltwise3_div", "eltwise2_mul", "eltwise2_data", eltwise_mode::prod),
        eltwise("eltwise4_add", "eltwise3_div", "eltwise4_data", eltwise_mode::sum),
        reorder("reorder_bfyx", "eltwise4_add", p.default_format, data_types::f32)
    );
    implementation_desc conv_impl = { format::b_fs_yx_fsv16, ""};
    bo_fused.set_option(build_option::force_implementations({ {"conv_prim", conv_impl} }));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_multi_eltwise_4_clamp,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                             bc_test_params{CASE_CONV_FP32_2, 2, 7},
                                             bc_test_params{CASE_CONV_FP32_3, 2, 7},
                                             bc_test_params{CASE_CONV_FP32_4, 2, 7},

                                             bc_test_params{CASE_CONV_FP16_2, 2, 7},
                                             bc_test_params{CASE_CONV_FP16_3, 2, 7},
                                             bc_test_params{CASE_CONV_FP16_4, 2, 7},
                                             }));


class conv_fp32_multi_eltwise_3_fusing : public ConvFusingTest {};
TEST_P(conv_fp32_multi_eltwise_3_fusing, basic) {
    if (engine.get_device_info().supports_immad) {
        return;
    }

    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
        data("eltwise_data1", get_mem(get_output_layout(p))),
        data("eltwise_data2", get_mem(get_output_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        data("weights", get_mem(get_weights_layout(p))),
        convolution("conv_prim", "input", { "weights" }, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
        eltwise("eltwise1", "conv_prim", "eltwise_data1", eltwise_mode::sum),
        eltwise("eltwise2", "conv_prim", "eltwise_data2", eltwise_mode::sum),
        eltwise("eltwise3", "eltwise1", "eltwise2", eltwise_mode::prod),
        reorder("reorder_bfyx", "eltwise3", p.default_format, data_types::f32)
    );
    implementation_desc conv_impl = { format::b_fs_yx_fsv16, ""};
    bo_fused.set_option(build_option::force_implementations({ {"conv_prim", conv_impl} }));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_multi_eltwise_3_fusing,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                             bc_test_params{CASE_CONV_FP32_2, 2, 5},
                                             bc_test_params{CASE_CONV_FP32_3, 2, 5},
                                             bc_test_params{CASE_CONV_FP32_4, 2, 5},

                                             bc_test_params{CASE_CONV_FP16_2, 2, 5},
                                             bc_test_params{CASE_CONV_FP16_3, 2, 5},
                                             bc_test_params{CASE_CONV_FP16_4, 2, 5},
                                             }));



class conv_fp32_multi_eltwise_quantization : public ConvFusingTest {};
TEST_P(conv_fp32_multi_eltwise_quantization, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                        data("weights", get_mem(get_weights_layout(p))),
                        data("bias", get_mem(get_bias_layout(p))),
                        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                        data("out_lo", get_mem(get_single_element_layout(p), -127)),
                        data("out_hi", get_mem(get_single_element_layout(p), 127)),
                        data("eltwise_data1", get_mem(get_output_layout(p))),
                        convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                        quantize("quantize", "conv_prim", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
                        eltwise("eltwise1", "conv_prim", "eltwise_data1", eltwise_mode::sum),
                        eltwise("eltwise2", "eltwise1", "quantize", eltwise_mode::prod),
                        reorder("reorder_bfyx", "eltwise2", p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_multi_eltwise_quantization,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                             bc_test_params{CASE_CONV_FP32_2, 4, 5},
                                             bc_test_params{CASE_CONV_FP32_4, 4, 5},

                                             bc_test_params{CASE_CONV_FP16_2, 4, 5},
                                             bc_test_params{CASE_CONV_FP16_3, 4, 5},
                                             bc_test_params{CASE_CONV_FP16_4, 4, 5},
                                             }));


class conv_fp32_multi_eltwise_concat : public ConvFusingTest {};
TEST_P(conv_fp32_multi_eltwise_concat, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
        data("eltwise_data1", get_mem(get_output_layout(p))),
        data("eltwise_data2", get_mem(get_output_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        data("weights", get_mem(get_weights_layout(p))),
        convolution("conv_prim", "input", { "weights" }, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
        eltwise("eltwise1", "conv_prim", "eltwise_data1", eltwise_mode::sum),
        eltwise("eltwise2", "conv_prim", "eltwise_data2", eltwise_mode::sum),
        concatenation("concat",
            {"eltwise1", "eltwise2"},
            concatenation::concatenation_axis::along_f,
            data_types::i8,
            padding{{0, 0, 0, 0}, 0}),
        reorder("reorder_bfyx", "concat", p.default_format, data_types::f32)
    );
    implementation_desc conv_impl = { format::b_fs_yx_fsv16, ""};
    bo_fused.set_option(build_option::force_implementations({ {"conv_prim", conv_impl} }));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_multi_eltwise_concat,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                             bc_test_params{CASE_CONV_FP32_2, 5, 5},
                                             bc_test_params{CASE_CONV_FP32_3, 5, 5},
                                             bc_test_params{CASE_CONV_FP32_4, 5, 5},

                                             bc_test_params{CASE_CONV_FP16_2, 5, 5},
                                             bc_test_params{CASE_CONV_FP16_3, 5, 5},
                                             bc_test_params{CASE_CONV_FP16_4, 5, 5},
                                             }));

class conv_fp32_eltwise_b_fs_zyx_fsv16 : public ConvFusingTest {};

TEST_P(conv_fp32_eltwise_b_fs_zyx_fsv16, vector_ops) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("eltwise_data", get_mem(get_output_layout(p))),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 eltwise("eltwise", "conv_prim", "eltwise_data", eltwise_mode::sum),
                 reorder("reorder_bfyx", "eltwise", p.default_format, data_types::f32)
    );

    implementation_desc conv_impl = { format::b_fs_zyx_fsv16, "" };
    bo_fused.set_option(build_option::force_implementations({ {"conv_prim", conv_impl} }));

    tolerance = 1e-5f;
    execute(p);
}

class conv_fp32_swish : public ConvFusingTest {};
TEST_P(conv_fp32_swish, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 activation("sigmoid", "conv_prim", activation_func::logistic),
                 eltwise("mul", {"conv_prim", "sigmoid"}, eltwise_mode::prod),
                 reorder("reorder_bfyx", "mul", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_swish,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                // bc_test_params{CASE_CONV_FP32_1, 2, 4},
                                bc_test_params{CASE_CONV_FP32_2, 2, 4},
                                bc_test_params{CASE_CONV_FP32_3, 2, 4},
                                bc_test_params{CASE_CONV_FP32_4, 2, 4},

                                // bc_test_params{CASE_CONV_FP32_1, 2, 4},
                                bc_test_params{CASE_CONV_FP16_2, 2, 4},
                                bc_test_params{CASE_CONV_FP16_3, 2, 4},
                                bc_test_params{CASE_CONV_FP16_4, 2, 4},
                        }));

TEST_P(conv_fp32_eltwise_b_fs_zyx_fsv16, splitted_vector_ops) {
    auto p = GetParam();

    std::vector<std::string> weights_idx;
    for (size_t w = 0; w < p.groups; w++) {
        create_topologies(data("weights" + std::to_string(w), get_mem(get_weights_layout(p, p.groups))));
        weights_idx.push_back(("weights" + std::to_string(w)));
    }

    create_topologies(input_layout("input", get_input_layout(p)),
                 data("eltwise_data", get_mem(get_output_layout(p))),
                 convolution("conv_prim", "input", weights_idx, {}, 1, p.stride, p.pad, p.dilation),
                 eltwise("eltwise", "conv_prim", "eltwise_data", eltwise_mode::sum),
                 reorder("reorder_bfyx", "eltwise", p.default_format, data_types::f32)
    );

    implementation_desc conv_impl = { format::b_fs_zyx_fsv16, "" };
    bo_fused.set_option(build_option::force_implementations({ {"conv_prim", conv_impl} }));

    tolerance = 1e-5f;
    //  commented because split mode is disabled
    //  execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_eltwise_b_fs_zyx_fsv16,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_FP32_6, 2, 3},
                                bc_test_params{CASE_CONV_FP32_7, 2, 3},
                                bc_test_params{CASE_CONV_FP32_8, 2, 3},
                                bc_test_params{CASE_CONV_FP32_9, 2, 3},
                                bc_test_params{CASE_CONV_FP32_11, 2, 3},
                                bc_test_params{CASE_CONV_FP32_12, 2, 3},
                                // bc_test_params{CASE_CONV_FP32_13, 2, 3}, - leads to mvn_scale_activation_quantize_i8_eltwise_fp32_quantize_i8.basic/11 test failure

                                bc_test_params{CASE_CONV_FP16_6, 2, 3},
                                bc_test_params{CASE_CONV_FP16_7, 2, 3},
                                bc_test_params{CASE_CONV_FP16_8, 2, 3},
                                bc_test_params{CASE_CONV_FP16_9, 2, 3},
                                bc_test_params{CASE_CONV_FP16_11, 2, 3},
                                bc_test_params{CASE_CONV_FP16_12, 2, 3},
                        }));

class conv_fp32_quantize_u8_first_conv : public ConvFusingTest {};
TEST_P(conv_fp32_quantize_u8_first_conv, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), 0)),
                 data("out_hi", get_mem(get_single_element_layout(p), 255)),
                 reorder("reordered_input", "input", format::b_fs_yx_fsv16, p.data_type),
                 convolution("conv_prim", "reordered_input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 quantize("quantize", "conv_prim", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::u8),
                 reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_quantize_u8_first_conv,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_FP32_14, 2, 3},
                        }));

class conv_fp32_quantize_u8 : public ConvFusingTest {};
TEST_P(conv_fp32_quantize_u8, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), 0)),
                 data("out_hi", get_mem(get_single_element_layout(p), 255)),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 quantize("quantize", "conv_prim", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::u8),
                 reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_quantize_u8,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                // For now only b_fs_yx_fsv16 supports this case
                                bc_test_params{CASE_CONV_FP32_2, 2, 3},
                                bc_test_params{CASE_CONV_FP32_3, 2, 3},

                                bc_test_params{CASE_CONV_FP16_2, 2, 3},
                                bc_test_params{CASE_CONV_FP16_3, 2, 3},
                        }));

class conv_fp32_scale_quantize_i8 : public ConvFusingTest {};
TEST_P(conv_fp32_scale_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), -127)),
                 data("out_hi", get_mem(get_single_element_layout(p), 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 scale("scale", "conv_prim", "scale_data"),
                 quantize("quantize", "scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
                 reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );
    // Output elements are in range [-127, 127]
    // 1.0f difference is allowed, since quantize can return different values in ref and scale_shift kernels
    // due to big error of division (in ref kernel).
    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_scale_quantize_i8,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                // For now only b_fs_yx_fsv16 supports this case
                                bc_test_params{CASE_CONV_FP32_2, 2, 4},
                                bc_test_params{CASE_CONV_FP32_3, 2, 4},

                                bc_test_params{CASE_CONV_FP16_2, 2, 4},
                                bc_test_params{CASE_CONV_FP16_3, 2, 4},
                        }));

class conv_fp32_scale_activation_quantize_i8 : public ConvFusingTest {};
TEST_P(conv_fp32_scale_activation_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), -127)),
                 data("out_hi", get_mem(get_single_element_layout(p), 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 scale("scale", "conv_prim", "scale_data"),
                 activation("activation_scale", "scale", activation_func::exp),
                 quantize("quantize", "activation_scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
                 reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_scale_activation_quantize_i8,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                // For now only b_fs_yx_fsv16 supports this case
                                bc_test_params{CASE_CONV_FP32_2, 2, 5},
                                bc_test_params{CASE_CONV_FP32_3, 2, 5},

                                bc_test_params{CASE_CONV_FP16_2, 2, 5},
                                bc_test_params{CASE_CONV_FP16_3, 2, 5},
                        }));

class conv_fp32_scale_activation_quantize_u8_eltwise_fp32 : public ConvFusingTest {};
TEST_P(conv_fp32_scale_activation_quantize_u8_eltwise_fp32, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), 0)),
                 data("out_hi", get_mem(get_single_element_layout(p), 255)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)),
                 data("eltwise_data", get_mem(get_output_layout(p))),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 scale("scale", "conv_prim", "scale_data"),
                 activation("activation_scale", "scale", activation_func::exp),
                 quantize("quantize", "activation_scale", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::u8),
                 eltwise("sum", { "quantize", "eltwise_data"}, eltwise_mode::sum,  p.default_type),
                 reorder("reorder_bfyx", "sum", p.default_format, data_types::f32)
    );
    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_scale_activation_quantize_u8_eltwise_fp32,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                // For now only b_fs_yx_fsv16 supports this case
                                bc_test_params{CASE_CONV_FP32_2, 2, 6},
                                bc_test_params{CASE_CONV_FP32_3, 2, 6},

                                bc_test_params{CASE_CONV_FP16_2, 2, 6},
                                bc_test_params{CASE_CONV_FP16_3, 2, 6},
                        }));

class conv_fp32_scale_activation_quantize_i8_activation : public ConvFusingTest {};
TEST_P(conv_fp32_scale_activation_quantize_i8_activation, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), -127)),
                 data("out_hi", get_mem(get_single_element_layout(p), 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)),
                 data("slope_data", get_mem(get_per_channel_layout(p))),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 scale("scale", "conv_prim", "scale_data"),
                 activation("activation_scale", "scale", "slope_data", activation_func::relu_negative_slope),
                 quantize("quantize", "activation_scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
                 activation("activation_quantize", "quantize", activation_func::relu),
                 reorder("reorder_bfyx", "activation_quantize", p.default_format, data_types::f32)
    );
    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_scale_activation_quantize_i8_activation,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_FP32_2, 2, 6},
                                bc_test_params{CASE_CONV_FP32_3, 2, 6},

                                bc_test_params{CASE_CONV_FP16_2, 2, 6},
                                bc_test_params{CASE_CONV_FP16_3, 2, 6},
                        }));


class conv_fp32_scale_activation_quantize_i8_eltwise_fp32_quantize_i8 : public ConvFusingTest {};
TEST_P(conv_fp32_scale_activation_quantize_i8_eltwise_fp32_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_lo1", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("in_hi1", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), -127)),
                 data("out_lo1", get_mem(get_single_element_layout(p), -127)),
                 data("out_hi", get_mem(get_single_element_layout(p), 127)),
                 data("out_hi1", get_mem(get_single_element_layout(p), 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)),
                 data("eltwise_data", get_mem(layout{data_types::i8, p.input_format, p.out_shape})),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 scale("scale", "conv_prim", "scale_data"),
                 activation("activation_scale", "scale", activation_func::exp),
                 quantize("quantize", "activation_scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
                 eltwise("sum", { "quantize", "eltwise_data"}, eltwise_mode::sum, data_types::f32),
                 quantize("quantize_1", "sum", "in_lo1", "in_hi1", "out_lo1", "out_hi1", 255, data_types::i8),
                 reorder("reorder_bfyx", "quantize_1", p.default_format, data_types::f32)
    );
    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_scale_activation_quantize_i8_eltwise_fp32_quantize_i8,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_FP32_2, 2, 7},
                                bc_test_params{CASE_CONV_FP32_3, 2, 7},
                        }));

class conv_fp32_activation_eltwise_in_u8_fp32 : public WeightsPrimitiveFusingTest<bc_test_params> {};
TEST_P(conv_fp32_activation_eltwise_in_u8_fp32, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("eltwise_data", get_mem(layout{ data_types::i8, p.input_format, p.out_shape })),
                 convolution("conv_prim", "input", { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
                 activation("activation", "conv_prim", activation_func::relu_negative_slope),
                 eltwise("sum", { "activation", "eltwise_data" }, eltwise_mode::sum, data_types::f32),
                 reorder("reorder_bfyx", "sum", p.default_format, data_types::f32)
    );
    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_activation_eltwise_in_u8_fp32,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                // bc_test_params{CASE_CONV_FP32_1, 2, 4}, - eltwise fusing not supported
                                bc_test_params{CASE_CONV_FP32_2, 2, 4},
                                bc_test_params{CASE_CONV_FP32_3, 2, 4},
                                bc_test_params{CASE_CONV_FP32_4, 2, 4},
                                // bc_test_params{CASE_CONV_FP32_5, 2, 4}, - eltwise fusing not supported
                                bc_test_params{CASE_CONV_FP32_6, 2, 4},
                                bc_test_params{CASE_CONV_FP32_7, 2, 4},
                                // bc_test_params{CASE_CONV_FP32_8, 2, 4}, - unknown bug
                                bc_test_params{CASE_CONV_FP32_9, 2, 4},
                                bc_test_params{CASE_CONV_FP32_10, 2, 4},
                        }));

class conv_fp32_activation_eltwise_diff_sizes : public ConvEltwTest {};
TEST_P(conv_fp32_activation_eltwise_diff_sizes, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("eltwise_data", get_mem(layout{ p.data_type, p.input_format, p.eltw_shape })),
                 convolution("conv_prim", "input", { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
                 activation("activation", "conv_prim", activation_func::relu_negative_slope),
                 eltwise("sum", { "activation", "eltwise_data" }, eltwise_mode::sum, data_types::f32),
                 reorder("reorder_bfyx", "sum", p.default_format, data_types::f32)
    );
    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp32_activation_eltwise_diff_sizes,
                        ::testing::ValuesIn(std::vector<conv_eltw_test_params>{
                                conv_eltw_test_params{CASE_CONV_ELTW_FP32_1, 2, 4},
                                conv_eltw_test_params{CASE_CONV_ELTW_FP32_2, 2, 4},
                                conv_eltw_test_params{CASE_CONV_ELTW_FP32_3, 2, 4},
                                conv_eltw_test_params{CASE_CONV_ELTW_FP32_4, 2, 4},
                                conv_eltw_test_params{CASE_CONV_ELTW_FP32_5, 2, 4},
                                conv_eltw_test_params{CASE_CONV_ELTW_FP32_6, 2, 4},
                                conv_eltw_test_params{CASE_CONV_ELTW_FP32_7, 3, 4},
                                conv_eltw_test_params{CASE_CONV_ELTW_FP32_8, 3, 4},
                        }));

class conv_scale_activation_eltwise_fp32_quantize_i8 : public ConvEltwTest {};
TEST_P(conv_scale_activation_eltwise_fp32_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 convolution("conv", "input", { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
                 data("scale_data", get_mem(get_per_channel_layout(p))),
                 scale("scale", "conv", "scale_data"),
                 activation("activation", "scale", activation_func::hyperbolic_tan),
                 data("eltwise_data", get_mem(layout{ p.data_type, p.input_format, p.eltw_shape })),
                 eltwise("eltw", { "activation", "eltwise_data" }, eltwise_mode::sum, data_types::f32),
                 data("in_low", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_high", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_low", get_mem(get_single_element_layout(p), -127, 127)),
                 data("out_high", get_mem(get_single_element_layout(p), -127, 127)),
                 quantize("quant", "eltw", "in_low", "in_high", "out_low", "out_high", 255, data_types::i8),
                 reorder("reorder_bfyx", "quant", p.default_format, data_types::f32)
    );
    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_scale_activation_eltwise_fp32_quantize_i8,
                        ::testing::ValuesIn(std::vector<conv_eltw_test_params>{
                                conv_eltw_test_params{CASE_CONV_ELTW_FP32_1, 2, 6},
                                conv_eltw_test_params{CASE_CONV_ELTW_FP32_2, 2, 6},
                                conv_eltw_test_params{CASE_CONV_ELTW_FP32_3, 2, 6},
                                conv_eltw_test_params{CASE_CONV_ELTW_FP32_4, 2, 6},
                                conv_eltw_test_params{CASE_CONV_ELTW_FP32_5, 3, 6},
                                conv_eltw_test_params{CASE_CONV_ELTW_FP32_6, 3, 6},
                                conv_eltw_test_params{CASE_CONV_ELTW_FP32_7, 3, 6},
                                conv_eltw_test_params{CASE_CONV_ELTW_FP32_8, 3, 6},
                        }));

/* ----------------------------------------------------------------------------------------------------- */
/* -------------------------------------- binary convolution cases ------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

class conv_bin_activation : public ConvFusingTest {};
TEST_P(conv_bin_activation, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p), -127, 127)),
                 binary_convolution("bin_conv_prim", "input", {"weights"}, p.stride, p.pad, p.dilation, p.out_shape, p.groups),
                 activation("activation", "bin_conv_prim", activation_func::relu),
                 reorder("reorder_bfyx", "activation", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_bin_activation,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                            bc_test_params{CASE_BIN_CONV1, 2, 3},
                                            }));

class conv_bin_scale_activation : public ConvFusingTest {};
TEST_P(conv_bin_scale_activation, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
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

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_bin_scale_activation,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                            bc_test_params{CASE_BIN_CONV1, 2, 4},
                            bc_test_params{CASE_BIN_CONV2, 2, 4},
                                            }));

class conv_bin_quantize_bin : public ConvFusingTest {};
TEST_P(conv_bin_quantize_bin, channel_wise_quantize) {
    auto p = GetParam();
    auto in_thresh = get_mem(get_per_channel_layout(p), min_random, max_random);
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p), -127, 127)),
                 data("in_lo", in_thresh),
                 data("in_hi", in_thresh),
                 data("out_lo", get_mem(get_per_channel_layout(p), -1)),
                 data("out_hi", get_mem(get_per_channel_layout(p),  1)),
                 binary_convolution("bin_conv_prim", "input", {"weights"}, p.stride, p.pad, p.dilation, p.out_shape, p.groups),
                 quantize("quantize_data", "bin_conv_prim", "in_lo", "in_hi", "out_lo", "out_hi", 2, data_types::bin),
                 reorder("reorder_bfyx", "quantize_data", p.default_format, data_types::f32)
    );
    tolerance = 1e-5f;
    execute(p);
}

TEST_P(conv_bin_quantize_bin, blob_wise_quantize) {
    auto p = GetParam();
    auto in_thresh = get_mem(get_single_element_layout(p), min_random, max_random);
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p), -127, 127)),
                 data("in_lo", in_thresh),
                 data("in_hi", in_thresh),
                 data("out_lo", get_mem(get_single_element_layout(p), -1)),
                 data("out_hi", get_mem(get_single_element_layout(p), 1)),
                 binary_convolution("bin_conv_prim", "input", {"weights"}, p.stride, p.pad, p.dilation, p.out_shape, p.groups),
                 quantize("quantize_data", "bin_conv_prim", "in_lo", "in_hi", "out_lo", "out_hi", 2, data_types::bin),
                 reorder("reorder_bfyx", "quantize_data", p.default_format, data_types::f32)
    );
    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_bin_quantize_bin,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                            bc_test_params{CASE_BIN_CONV1, 2, 3},
                            bc_test_params{CASE_BIN_CONV2, 2, 3},
                                            }));

class conv_bin_scale_conv_dw : public ConvFusingTest {};
TEST_P(conv_bin_scale_conv_dw, dw_kernel_3x3_stride2) {
    auto p = GetParam();
    auto dw_tensor = cldnn::tensor(group(p.out_shape.feature[0]), batch(1), feature(1), spatial(3, 3));
    auto dw_weights_layout = layout{p.default_type, format::goiyx, dw_tensor};

    auto dw_stride = tensor{1, 1, 2, 2};
    create_topologies(input_layout("input", get_input_layout(p)),
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
    auto dw_tensor = cldnn::tensor(group(p.out_shape.feature[0]), batch(1), feature(1), spatial(3, 3));
    auto dw_weights_layout = layout{p.default_type, format::goiyx, dw_tensor};

    auto dw_stride = tensor{1, 1, 1, 1};
    create_topologies(input_layout("input", get_input_layout(p)),
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

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_bin_scale_conv_dw,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                            bc_test_params{CASE_BIN_CONV2, 3, 4},
                            bc_test_params{CASE_BIN_CONV3, 3, 4},
                                            }));

class conv_bin_scale_conv_dw_prelu : public ConvFusingTest {};
TEST_P(conv_bin_scale_conv_dw_prelu, dw_kernel_3x3_stride2) {
    auto p = GetParam();
    auto dw_tensor = cldnn::tensor(group(p.out_shape.feature[0]), batch(1), feature(1), spatial(3, 3));
    auto dw_weights_layout = layout{p.default_type, format::goiyx, dw_tensor};

    auto dw_stride = tensor{1, 1, 2, 2};
    auto in_thresh = get_mem(get_per_channel_layout(p), min_random, max_random);
    create_topologies(input_layout("input", get_input_layout(p)),
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
    auto dw_tensor = cldnn::tensor(group(p.out_shape.feature[0]), batch(1), feature(1), spatial(3, 3));
    auto dw_weights_layout = layout{p.default_type, format::goiyx, dw_tensor};

    auto dw_stride = tensor{1, 1, 1, 1};
    auto in_thresh = get_mem(get_per_channel_layout(p), min_random, max_random);
    create_topologies(input_layout("input", get_input_layout(p)),
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

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_bin_scale_conv_dw_prelu,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                            bc_test_params{CASE_BIN_CONV2, 3, 5},
                            bc_test_params{CASE_BIN_CONV3, 3, 5},
                                            }));

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- INT8 convolution cases ------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
class conv_int8_scale : public ConvFusingTest {};
TEST_P(conv_int8_scale, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
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

TEST_P(conv_int8_scale, fp16_scale_out) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count())),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 scale("scale", "conv_prim", "scale_data", optional_data_type{data_types::f16}),
                 reorder("reorder_bfyx", "scale", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_scale,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_U8S8_1, 2, 3},
                                bc_test_params{CASE_CONV_U8S8_2, 2, 3},
                                bc_test_params{CASE_CONV_U8S8_3, 2, 3},
                                bc_test_params{CASE_CONV_U8S8_4, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_1, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_2, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_3, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_4, 2, 3},

                                bc_test_params{CASE_CONV3D_U8S8_1, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_2, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_3, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_4, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_5, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_1, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_2, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_3, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_4, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_5, 2, 3},
                        }));

class conv_int8_eltwise : public ConvFusingTest {};
TEST_P(conv_int8_eltwise, fp16_eltwise_out) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count())),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 eltwise("scale", {"conv_prim", "scale_data"}, eltwise_mode::prod, data_types::f16),
                 reorder("reorder_bfyx", "scale", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_eltwise,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_U8S8_1, 2, 3},
                                bc_test_params{CASE_CONV_U8S8_2, 2, 3},
                                bc_test_params{CASE_CONV_U8S8_3, 2, 3},
                                bc_test_params{CASE_CONV_U8S8_4, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_1, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_2, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_3, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_4, 2, 3},

                                bc_test_params{CASE_CONV3D_U8S8_1, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_2, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_3, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_4, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_5, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_1, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_2, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_3, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_4, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_5, 2, 3},
                        }));

class conv_int8_scale_shift_swish : public ConvFusingTest {};
TEST_P(conv_int8_scale_shift_swish, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count())),
                 data("shift_data", get_mem(get_per_channel_layout(p), 1)),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 eltwise("scale0", {"conv_prim", "scale_data"}, eltwise_mode::prod),
                 eltwise("scale1", {"conv_prim", "scale_data"}, eltwise_mode::prod),
                 eltwise("shift0", {"scale0", "shift_data"}, eltwise_mode::sum),
                 eltwise("shift1", {"scale1", "shift_data"}, eltwise_mode::sum),
                 activation("sigmoid", "shift0", activation_func::logistic),
                 eltwise("mul", {"shift1", "sigmoid"}, eltwise_mode::prod),
                 reorder("reorder_bfyx", "mul", p.default_format, data_types::f32)
    );

    tolerance = 1e-4f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_scale_shift_swish,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_U8S8_1, 2, 8},
                                bc_test_params{CASE_CONV_U8S8_2, 2, 8},
                                bc_test_params{CASE_CONV_U8S8_3, 2, 8},
                                bc_test_params{CASE_CONV_U8S8_4, 2, 8},
                                bc_test_params{CASE_CONV_S8S8_1, 2, 8},
                                bc_test_params{CASE_CONV_S8S8_2, 2, 8},
                                bc_test_params{CASE_CONV_S8S8_3, 2, 8},
                                bc_test_params{CASE_CONV_S8S8_4, 2, 8},

                                bc_test_params{CASE_CONV3D_U8S8_1, 2, 8},
                                bc_test_params{CASE_CONV3D_U8S8_2, 2, 8},
                                bc_test_params{CASE_CONV3D_U8S8_3, 2, 8},
                                bc_test_params{CASE_CONV3D_U8S8_4, 2, 8},
                                bc_test_params{CASE_CONV3D_U8S8_5, 2, 8},
                                bc_test_params{CASE_CONV3D_S8S8_1, 2, 8},
                                bc_test_params{CASE_CONV3D_S8S8_2, 2, 8},
                                bc_test_params{CASE_CONV3D_S8S8_3, 2, 8},
                                bc_test_params{CASE_CONV3D_S8S8_4, 2, 8},
                                bc_test_params{CASE_CONV3D_S8S8_5, 2, 8},
                        }));

class conv_int8_prelu_eltwise : public ConvFusingTest {};
TEST_P(conv_int8_prelu_eltwise, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
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

TEST_P(conv_int8_prelu_eltwise, fsv16) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("slope_data", get_mem(get_per_channel_layout(p))),
                 data("eltwise_data", get_mem(get_output_layout(p))),
                 convolution("conv_prim", "input", { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
                 activation("activation", "conv_prim", "slope_data", activation_func::relu_negative_slope),
                 eltwise("eltwise", "activation", "eltwise_data", eltwise_mode::sum),
                 reorder("reorder_bfyx", "eltwise", p.default_format, data_types::f32)
    );

    if (p.default_format.dimension() == 4) {
        implementation_desc conv_impl = { format::b_fs_yx_fsv16, "" };
        bo_fused.set_option(build_option::force_implementations({ {"conv_prim", conv_impl} }));
    } else {
        // TODO Add 5D int8 optimized convolution implementations
        return;
    }

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_prelu_eltwise,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_U8S8_1, 2, 4},
                                bc_test_params{CASE_CONV_U8S8_2, 2, 4},
                                bc_test_params{CASE_CONV_U8S8_3, 2, 4},
                                bc_test_params{CASE_CONV_U8S8_4, 2, 4},
                                bc_test_params{CASE_CONV_U8S8_7, 2, 4},
                                bc_test_params{CASE_CONV_U8S8_8, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_1, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_2, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_3, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_4, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_7, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_8, 2, 4},

                                bc_test_params{CASE_CONV3D_U8S8_1, 2, 4},
                                bc_test_params{CASE_CONV3D_U8S8_2, 2, 4},
                                bc_test_params{CASE_CONV3D_U8S8_3, 2, 4},
                                bc_test_params{CASE_CONV3D_U8S8_4, 2, 4},
                                bc_test_params{CASE_CONV3D_U8S8_5, 2, 4},
                                bc_test_params{CASE_CONV3D_S8S8_1, 2, 4},
                                bc_test_params{CASE_CONV3D_S8S8_2, 2, 4},
                                bc_test_params{CASE_CONV3D_S8S8_3, 2, 4},
                                bc_test_params{CASE_CONV3D_S8S8_4, 2, 4},
                                bc_test_params{CASE_CONV3D_S8S8_5, 2, 4},
                        }));

class conv_int8_activation_eltwise_quantize : public ConvFusingTest {};
TEST_P(conv_int8_activation_eltwise_quantize, fsv16) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("eltwise_data", get_mem(get_output_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), -127)),
                 data("out_hi", get_mem(get_single_element_layout(p), 127)),
                 convolution("conv_prim", "input", { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
                 activation("activation", "conv_prim", activation_func::negative),
                 eltwise("eltwise", "activation", "eltwise_data", eltwise_mode::sum),
                 quantize("quantize", "eltwise", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
                 reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    if (p.default_format.dimension() == 4) {
        implementation_desc conv_impl = { format::b_fs_yx_fsv16, "" };
        bo_fused.set_option(build_option::force_implementations({ {"conv_prim", conv_impl} }));
    } else {
        // TODO Add 5D int8 optimized convolution implementations
        return;
    }

    tolerance = 1.f;
    execute(p);
}

TEST_P(conv_int8_activation_eltwise_quantize, fsv32) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("eltwise_data", get_mem(get_output_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), -127)),
                 data("out_hi", get_mem(get_single_element_layout(p), 127)),
                 convolution("conv_prim", "input", { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
                 activation("activation", "conv_prim", activation_func::negative),
                 eltwise("eltwise", "activation", "eltwise_data", eltwise_mode::sum),
                 quantize("quantize", "eltwise", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
                 reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    if (p.default_format.dimension() == 4) {
        implementation_desc conv_impl = { format::b_fs_yx_fsv32, "" };
        bo_fused.set_option(build_option::force_implementations({ {"conv_prim", conv_impl} }));
    } else {
        // TODO Add 5D int8 optimized convolution implementations
        return;
    }

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_activation_eltwise_quantize,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_U8S8_1, 2, 5},
                                bc_test_params{CASE_CONV_U8S8_2, 2, 5},
                                bc_test_params{CASE_CONV_U8S8_3, 2, 5},
                                bc_test_params{CASE_CONV_U8S8_4, 2, 5},
                                bc_test_params{CASE_CONV_U8S8_7, 2, 5},
                                bc_test_params{CASE_CONV_U8S8_8, 2, 5},
                                bc_test_params{CASE_CONV_S8S8_1, 2, 5},
                                bc_test_params{CASE_CONV_S8S8_2, 2, 5},
                                bc_test_params{CASE_CONV_S8S8_3, 2, 5},
                                bc_test_params{CASE_CONV_S8S8_4, 2, 5},
                                bc_test_params{CASE_CONV_S8S8_7, 2, 5},
                                bc_test_params{CASE_CONV_S8S8_8, 2, 5},
                        }));

class conv_int8_activation_eltwise : public ConvFusingTest {};
TEST_P(conv_int8_activation_eltwise, fsv16) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("eltwise_data", get_mem(get_output_layout(p))),
                 convolution("conv_prim", "input", { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
                 activation("activation", "conv_prim", activation_func::negative),
                 eltwise("eltwise", "activation", "eltwise_data", eltwise_mode::sum),
                 reorder("reorder_bfyx", "eltwise", p.default_format, data_types::f32)
    );

    if (p.default_format.dimension() == 4) {
        implementation_desc conv_impl = { format::b_fs_yx_fsv16, "" };
        bo_fused.set_option(build_option::force_implementations({ {"conv_prim", conv_impl} }));
    } else {
        // TODO Add 5D int8 optimized convolution implementations
        return;
    }

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(conv_int8_activation_eltwise, fsv32) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("eltwise_data", get_mem(get_output_layout(p))),
                 convolution("conv_prim", "input", { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
                 activation("activation", "conv_prim", activation_func::negative),
                 eltwise("eltwise", "activation", "eltwise_data", eltwise_mode::sum),
                 reorder("reorder_bfyx", "eltwise", p.default_format, data_types::f32)
    );

    if (p.default_format.dimension() == 4) {
        implementation_desc conv_impl = { format::b_fs_yx_fsv32, "" };
        bo_fused.set_option(build_option::force_implementations({ {"conv_prim", conv_impl} }));
    } else {
        // TODO Add 5D int8 optimized convolution implementations
        return;
    }

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_activation_eltwise,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_U8S8_1, 2, 4},
                                bc_test_params{CASE_CONV_U8S8_2, 2, 4},
                                bc_test_params{CASE_CONV_U8S8_3, 2, 4},
                                bc_test_params{CASE_CONV_U8S8_4, 2, 4},
                                bc_test_params{CASE_CONV_U8S8_7, 2, 4},
                                bc_test_params{CASE_CONV_U8S8_8, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_1, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_2, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_3, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_4, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_7, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_8, 2, 4},
                        }));

class conv_int8_quantize_u8 : public ConvFusingTest {};
TEST_P(conv_int8_quantize_u8, per_channel) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), 0)),
                 data("out_hi", get_mem(get_single_element_layout(p), 255)),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 quantize("quantize", "conv_prim", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::u8),
                 reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

TEST_P(conv_int8_quantize_u8, per_tensor) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_single_element_layout(p), -10)),
                 data("in_hi", get_mem(get_single_element_layout(p), 10)),
                 data("out_lo", get_mem(get_single_element_layout(p), 0)),
                 data("out_hi", get_mem(get_single_element_layout(p), 255)),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 quantize("quantize", "conv_prim", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::u8),
                 reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_quantize_u8,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_U8S8_1, 2, 3},
                                bc_test_params{CASE_CONV_U8S8_2, 2, 3},
                                bc_test_params{CASE_CONV_U8S8_3, 2, 3},
                                bc_test_params{CASE_CONV_U8S8_4, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_1, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_2, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_3, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_4, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_8, 2, 3},

                                bc_test_params{CASE_CONV3D_U8S8_1, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_2, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_3, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_4, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_5, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_1, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_2, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_3, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_4, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_5, 2, 3},
                        }));

class conv_int8_scale_quantize_i8 : public ConvFusingTest {};
TEST_P(conv_int8_scale_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), -127)),
                 data("out_hi", get_mem(get_single_element_layout(p), 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 scale("scale", "conv_prim", "scale_data"),
                 quantize("quantize", "scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
                 reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );
    // Output elements are in range [-127, 127]
    // 1.0f difference is allowed, since quantize can return different values in ref and scale_shift kernels
    // due to big error of division (in ref kernel).
    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_scale_quantize_i8,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_U8S8_1, 2, 4},
                                bc_test_params{CASE_CONV_U8S8_2, 2, 4},
                                bc_test_params{CASE_CONV_U8S8_3, 2, 4},
                                bc_test_params{CASE_CONV_U8S8_4, 2, 4},
                                bc_test_params{CASE_CONV_U8S8_9, 2, 4},
                                bc_test_params{CASE_CONV_U8S8_10, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_1, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_2, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_3, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_4, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_9, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_10, 2, 4},

                                bc_test_params{CASE_CONV3D_U8S8_1, 2, 4},
                                bc_test_params{CASE_CONV3D_U8S8_2, 2, 4},
                                bc_test_params{CASE_CONV3D_U8S8_3, 2, 4},
                                bc_test_params{CASE_CONV3D_U8S8_4, 2, 4},
                                bc_test_params{CASE_CONV3D_U8S8_5, 2, 4},
                                bc_test_params{CASE_CONV3D_S8S8_1, 2, 4},
                                bc_test_params{CASE_CONV3D_S8S8_2, 2, 4},
                                bc_test_params{CASE_CONV3D_S8S8_3, 2, 4},
                                bc_test_params{CASE_CONV3D_S8S8_4, 2, 4},
                                bc_test_params{CASE_CONV3D_S8S8_5, 2, 4},
                        }));

class conv_int8_scale_quantize_i8_conv_b_fs_yx_fsv4_int8 : public ConvFusingTest {};
TEST_P(conv_int8_scale_quantize_i8_conv_b_fs_yx_fsv4_int8, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), -127)),
                 data("out_hi", get_mem(get_single_element_layout(p), 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / p.kernel.count() / 255)),
                 convolution("conv_prim", "input", { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
                 scale("scale", "conv_prim", "scale_data"),
                 quantize("quantize", "scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
                 reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );
    implementation_desc conv_impl = { format::b_fs_yx_fsv4, "convolution_gpu_b_fs_yx_fsv4_int8" };
    bo_fused.set_option(build_option::force_implementations({ {"conv_prim", conv_impl} }));

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_scale_quantize_i8_conv_b_fs_yx_fsv4_int8,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{ CASE_CONV_S8S8_11, 2, 4 },
                        }));

class conv_int8_relu_quantize : public ConvFusingTest {};
TEST_P(conv_int8_relu_quantize, i8) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), -127)),
                 data("out_hi", get_mem(get_single_element_layout(p), 127)),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 activation("relu", "conv_prim", activation_func::relu),
                 quantize("quantize", "relu", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::i8),
                 reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );
    // Output elements are in range [-127, 127]
    // 1.0f difference is allowed, since quantize can return different values in ref and scale_shift kernels
    // due to big error of division (in ref kernel).
    tolerance = 1.0f;
    execute(p);
}

TEST_P(conv_int8_relu_quantize, u8) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), 0)),
                 data("out_hi", get_mem(get_single_element_layout(p), 255)),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 activation("relu", "conv_prim", activation_func::relu),
                 quantize("quantize", "relu", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::u8),
                 reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );
    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_relu_quantize,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_U8S8_1, 2, 4},
                                bc_test_params{CASE_CONV_U8S8_2, 2, 4},
                                bc_test_params{CASE_CONV_U8S8_3, 2, 4},
                                bc_test_params{CASE_CONV_U8S8_4, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_1, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_2, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_3, 2, 4},
                                bc_test_params{CASE_CONV_S8S8_4, 2, 4},

                                bc_test_params{CASE_CONV3D_U8S8_1, 2, 4},
                                bc_test_params{CASE_CONV3D_U8S8_2, 2, 4},
                                bc_test_params{CASE_CONV3D_U8S8_3, 2, 4},
                                bc_test_params{CASE_CONV3D_U8S8_4, 2, 4},
                                bc_test_params{CASE_CONV3D_U8S8_5, 2, 4},
                                bc_test_params{CASE_CONV3D_S8S8_1, 2, 4},
                                bc_test_params{CASE_CONV3D_S8S8_2, 2, 4},
                                bc_test_params{CASE_CONV3D_S8S8_3, 2, 4},
                                bc_test_params{CASE_CONV3D_S8S8_4, 2, 4},
                                bc_test_params{CASE_CONV3D_S8S8_5, 2, 4},
                        }));

class conv_int8_scale_activation_quantize_i8 : public ConvFusingTest {};
TEST_P(conv_int8_scale_activation_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), -127)),
                 data("out_hi", get_mem(get_single_element_layout(p), 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 scale("scale", "conv_prim", "scale_data"),
                 activation("activation_scale", "scale", activation_func::exp),
                 quantize("quantize", "activation_scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
                 reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_scale_activation_quantize_i8,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_U8S8_1, 2, 5},
                                bc_test_params{CASE_CONV_U8S8_2, 2, 5},
                                bc_test_params{CASE_CONV_U8S8_3, 2, 5},
                                bc_test_params{CASE_CONV_U8S8_4, 2, 5},
                                bc_test_params{CASE_CONV_S8S8_1, 2, 5},
                                bc_test_params{CASE_CONV_S8S8_2, 2, 5},
                                bc_test_params{CASE_CONV_S8S8_3, 2, 5},
                                bc_test_params{CASE_CONV_S8S8_4, 2, 5},

                                bc_test_params{CASE_CONV3D_U8S8_1, 2, 5},
                                bc_test_params{CASE_CONV3D_U8S8_2, 2, 5},
                                bc_test_params{CASE_CONV3D_U8S8_3, 2, 5},
                                bc_test_params{CASE_CONV3D_U8S8_4, 2, 5},
                                bc_test_params{CASE_CONV3D_U8S8_5, 2, 5},
                                bc_test_params{CASE_CONV3D_S8S8_1, 2, 5},
                                bc_test_params{CASE_CONV3D_S8S8_2, 2, 5},
                                bc_test_params{CASE_CONV3D_S8S8_3, 2, 5},
                                bc_test_params{CASE_CONV3D_S8S8_4, 2, 5},
                                bc_test_params{CASE_CONV3D_S8S8_5, 2, 5},
                        }));

class conv_int8_scale_activation_quantize_i8_eltwise_fp32 : public ConvFusingTest {};
TEST_P(conv_int8_scale_activation_quantize_i8_eltwise_fp32, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), -127)),
                 data("out_hi", get_mem(get_single_element_layout(p), 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)),
                 data("eltwise_data", get_mem(get_output_layout(p))),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 scale("scale", "conv_prim", "scale_data"),
                 activation("activation_scale", "scale", activation_func::exp),
                 quantize("quantize", "activation_scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
                 eltwise("sum", { "quantize", "eltwise_data"}, eltwise_mode::sum,  data_types::f32),
                 reorder("reorder_bfyx", "sum", p.default_format, data_types::f32)
    );
    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_scale_activation_quantize_i8_eltwise_fp32,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_U8S8_1, 2, 6},
                                bc_test_params{CASE_CONV_U8S8_2, 2, 6},
                                bc_test_params{CASE_CONV_U8S8_3, 2, 6},
                                bc_test_params{CASE_CONV_U8S8_4, 2, 6},
                                bc_test_params{CASE_CONV_S8S8_1, 2, 6},
                                bc_test_params{CASE_CONV_S8S8_2, 2, 6},
                                bc_test_params{CASE_CONV_S8S8_3, 2, 6},
                                bc_test_params{CASE_CONV_S8S8_4, 2, 6},

                                bc_test_params{CASE_CONV3D_U8S8_1, 2, 6},
                                bc_test_params{CASE_CONV3D_U8S8_2, 2, 6},
                                bc_test_params{CASE_CONV3D_U8S8_3, 2, 6},
                                bc_test_params{CASE_CONV3D_U8S8_4, 2, 6},
                                bc_test_params{CASE_CONV3D_U8S8_5, 2, 6},
                                bc_test_params{CASE_CONV3D_S8S8_1, 2, 6},
                                bc_test_params{CASE_CONV3D_S8S8_2, 2, 6},
                                bc_test_params{CASE_CONV3D_S8S8_3, 2, 6},
                                bc_test_params{CASE_CONV3D_S8S8_4, 2, 6},
                                bc_test_params{CASE_CONV3D_S8S8_5, 2, 6},
                        }));

class conv_int8_scale_activation_quantize_i8_activation : public ConvFusingTest {};
TEST_P(conv_int8_scale_activation_quantize_i8_activation, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), -127)),
                 data("out_hi", get_mem(get_single_element_layout(p), 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)),
                 data("slope_data", get_mem(get_per_channel_layout(p))),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 scale("scale", "conv_prim", "scale_data"),
                 activation("activation_scale", "scale", "slope_data", activation_func::relu_negative_slope),
                 quantize("quantize", "activation_scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
                 activation("activation_quantize", "quantize", activation_func::relu),
                 reorder("reorder_bfyx", "activation_quantize", p.default_format, data_types::f32)
    );
    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_scale_activation_quantize_i8_activation,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_U8S8_1, 2, 6},
                                bc_test_params{CASE_CONV_U8S8_2, 2, 6},
                                bc_test_params{CASE_CONV_U8S8_3, 2, 6},
                                bc_test_params{CASE_CONV_U8S8_4, 2, 6},
                                bc_test_params{CASE_CONV_S8S8_1, 2, 6},
                                bc_test_params{CASE_CONV_S8S8_2, 2, 6},
                                bc_test_params{CASE_CONV_S8S8_3, 2, 6},
                                bc_test_params{CASE_CONV_S8S8_4, 2, 6},

                                bc_test_params{CASE_CONV3D_U8S8_1, 2, 6},
                                bc_test_params{CASE_CONV3D_U8S8_2, 2, 6},
                                bc_test_params{CASE_CONV3D_U8S8_3, 2, 6},
                                bc_test_params{CASE_CONV3D_U8S8_4, 2, 6},
                                bc_test_params{CASE_CONV3D_U8S8_5, 2, 6},
                                bc_test_params{CASE_CONV3D_S8S8_1, 2, 6},
                                bc_test_params{CASE_CONV3D_S8S8_2, 2, 6},
                                bc_test_params{CASE_CONV3D_S8S8_3, 2, 6},
                                bc_test_params{CASE_CONV3D_S8S8_4, 2, 6},
                                bc_test_params{CASE_CONV3D_S8S8_5, 2, 6},
                        }));


class conv_int8_scale_activation_quantize_i8_eltwise_fp32_quantize_i8 : public ConvFusingTest {};
// With some input values accuracy error might be = 2, so the test is disabled.
TEST_P(conv_int8_scale_activation_quantize_i8_eltwise_fp32_quantize_i8, DISABLED_basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_lo1", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("in_hi1", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), -127)),
                 data("out_lo1", get_mem(get_single_element_layout(p), -127)),
                 data("out_hi", get_mem(get_single_element_layout(p), 127)),
                 data("out_hi1", get_mem(get_single_element_layout(p), 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)),
                 data("eltwise_data", get_mem(layout{data_types::i8, p.input_format, p.out_shape})),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 scale("scale", "conv_prim", "scale_data"),
                 activation("activation_scale", "scale", activation_func::exp),
                 quantize("quantize", "activation_scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
                 eltwise("sum", { "quantize", "eltwise_data"}, eltwise_mode::sum, data_types::f32),
                 quantize("quantize_1", "sum", "in_lo1", "in_hi1", "out_lo1", "out_hi1", 255, data_types::i8),
                 reorder("reorder_bfyx", "quantize_1", p.default_format, data_types::f32)
    );
    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_scale_activation_quantize_i8_eltwise_fp32_quantize_i8,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_U8S8_1, 2, 7},
                                bc_test_params{CASE_CONV_U8S8_2, 2, 7},
                                bc_test_params{CASE_CONV_U8S8_3, 2, 7},
                                bc_test_params{CASE_CONV_U8S8_4, 2, 7},
                                bc_test_params{CASE_CONV_S8S8_1, 2, 7},
                                bc_test_params{CASE_CONV_S8S8_2, 2, 7},
                                bc_test_params{CASE_CONV_S8S8_3, 2, 7},
                                bc_test_params{CASE_CONV_S8S8_4, 2, 7},

                                bc_test_params{CASE_CONV3D_U8S8_1, 2, 7},
                                bc_test_params{CASE_CONV3D_U8S8_2, 2, 7},
                                bc_test_params{CASE_CONV3D_U8S8_3, 2, 7},
                                bc_test_params{CASE_CONV3D_U8S8_4, 2, 7},
                                bc_test_params{CASE_CONV3D_U8S8_5, 2, 7},
                                bc_test_params{CASE_CONV3D_S8S8_1, 2, 7},
                                bc_test_params{CASE_CONV3D_S8S8_2, 2, 7},
                                bc_test_params{CASE_CONV3D_S8S8_3, 2, 7},
                                bc_test_params{CASE_CONV3D_S8S8_4, 2, 7},
                                bc_test_params{CASE_CONV3D_S8S8_5, 2, 7},
                        }));

class conv_int8_scale_prelu_quantize_i8_eltwise_fp32_quantize_i8_vec : public ConvFusingTest {};
TEST_P(conv_int8_scale_prelu_quantize_i8_eltwise_fp32_quantize_i8_vec, vector_ops) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_lo1", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("in_hi1", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), -127)),
                 data("out_lo1", get_mem(get_single_element_layout(p), -127)),
                 data("out_hi", get_mem(get_single_element_layout(p), 127)),
                 data("out_hi1", get_mem(get_single_element_layout(p), 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)),
                 data("slope_data", get_mem(get_per_channel_layout(p))),
                 data("eltwise_data", get_mem(layout{data_types::i8, format::b_fs_yx_fsv4, p.out_shape})),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 scale("scale", "conv_prim", "scale_data"),
                 activation("activation_scale", "scale", "slope_data", activation_func::relu_negative_slope),
                 quantize("quantize", "activation_scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
                 eltwise("sum", { "quantize", "eltwise_data"}, eltwise_mode::sum, data_types::f32),
                 quantize("quantize_1", "sum", "in_lo1", "in_hi1", "out_lo1", "out_hi1", 255, data_types::i8),
                 reorder("reorder_bfyx", "quantize_1", p.default_format, data_types::f32)
    );

    implementation_desc conv_impl = { format::b_fs_yx_fsv4, "convolution_gpu_b_fs_yx_fsv4_1x1" };
    bo_fused.set_option(build_option::force_implementations({ {"conv_prim", conv_impl} }));

    tolerance = 1.f;
    execute(p);
}

TEST_P(conv_int8_scale_prelu_quantize_i8_eltwise_fp32_quantize_i8_vec, vector_ops_mixed_types) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_lo1", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("in_hi1", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_lo", get_mem(get_single_element_layout(p), -127)),
                 data("out_lo1", get_mem(get_single_element_layout(p), -127)),
                 data("out_hi", get_mem(get_single_element_layout(p), 127)),
                 data("out_hi1", get_mem(get_single_element_layout(p), 127)),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count()/255)),
                 data("slope_data", get_mem(layout{ data_types::f16, p.default_format, tensor{1, p.out_shape.feature[0], 1, 1} })),
                 data("eltwise_data", get_mem(layout{data_types::u8, format::b_fs_yx_fsv4, p.out_shape})),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 scale("scale", "conv_prim", "scale_data"),
                 activation("activation_scale", "scale", "slope_data", activation_func::relu_negative_slope),
                 quantize("quantize", "activation_scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
                 eltwise("sum", { "quantize", "eltwise_data"}, eltwise_mode::sum, data_types::f32),
                 quantize("quantize_1", "sum", "in_lo1", "in_hi1", "out_lo1", "out_hi1", 255, data_types::i8),
                 reorder("reorder_bfyx", "quantize_1", p.default_format, data_types::f32)
    );

    implementation_desc conv_impl = { format::b_fs_yx_fsv4, "convolution_gpu_b_fs_yx_fsv4_1x1" };
    bo_fused.set_option(build_option::force_implementations({ {"conv_prim", conv_impl} }));

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_scale_prelu_quantize_i8_eltwise_fp32_quantize_i8_vec,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_U8S8_3, 2, 7},
                                bc_test_params{CASE_CONV_U8S8_5, 2, 7},
                                bc_test_params{CASE_CONV_S8S8_3, 2, 7},
                                bc_test_params{CASE_CONV_S8S8_5, 2, 7},
                        }));

class conv_int8_asymmetric_weights : public ConvFusingTest {};
TEST_P(conv_int8_asymmetric_weights, basic) {
    auto p = GetParam();
    auto weights_format = (p.weights_format == format::goiyx) ? format::bfyx : format::bfzyx;
    auto weights_layout = (p.groups > 1) ? get_weights_layout(p, 1, weights_format) :
                                           get_weights_layout(p);
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(weights_layout)),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("w_zp", get_mem(get_weights_zp_layout(p), 1, 127)),
                 eltwise("w_sub", {"weights", "w_zp"}, eltwise_mode::sub, data_types::f32),
                 convolution("conv_prim", "input", {"w_sub"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation, p.out_shape, data_types::f32, false),
                 reorder("reorder_bfyx", "conv_prim", p.default_format, data_types::f32)
    );
    tolerance = 1.f;

    auto input_prim = get_mem(get_input_layout(p));
    network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
    network network_fused(this->engine, this->topology_fused, bo_fused);
    network_fused.set_input_data("input", input_prim);
    network_not_fused.set_input_data("input", input_prim);

    ASSERT_FALSE(network_fused.get_primitives_info().empty());
    ASSERT_FALSE(network_not_fused.get_primitives_info().empty());

    // Search for both conv_prim and reorder_bfyx, as in case of fused topology convolution will be merged with the last reorder
    auto find_conv = [](primitive_info& p) -> bool {
        if (p.original_id == "conv_prim" || p.original_id == "reorder_bfyx")
            return true;
        return false;
    };

    auto pi_fused = network_fused.get_primitives_info();
    auto pi_not_fused = network_not_fused.get_primitives_info();
    auto info_fused = std::find_if(pi_fused.begin(), pi_fused.end(), find_conv);
    auto info_not_fused = std::find_if(pi_not_fused.begin(), pi_not_fused.end(), find_conv);

    ASSERT_TRUE(info_fused != pi_fused.end());
    ASSERT_TRUE(info_not_fused != pi_not_fused.end());

    ASSERT_EQ(info_fused->c_dependencies.size(), 4lu);  // input + weights + bias + w_zp
    ASSERT_EQ(info_not_fused->c_dependencies.size(), 3lu);  // input + weights + bias

    compare(network_not_fused, network_fused, p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_asymmetric_weights,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_U8S8_1, 2, 2},
                                bc_test_params{CASE_CONV_U8S8_2, 2, 2},
                                bc_test_params{CASE_CONV_U8S8_3, 2, 2},
                                bc_test_params{CASE_CONV_U8S8_4, 2, 2},
                                bc_test_params{CASE_CONV_S8S8_1, 2, 2},
                                bc_test_params{CASE_CONV_S8S8_2, 2, 2},
                                bc_test_params{CASE_CONV_S8S8_3, 2, 2},
                                bc_test_params{CASE_CONV_S8S8_4, 2, 2},

                                bc_test_params{CASE_CONV3D_U8S8_1, 2, 2},
                                bc_test_params{CASE_CONV3D_U8S8_2, 2, 2},
                                bc_test_params{CASE_CONV3D_U8S8_3, 2, 2},
                                bc_test_params{CASE_CONV3D_U8S8_4, 2, 2},
                                bc_test_params{CASE_CONV3D_S8S8_1, 2, 2},
                                bc_test_params{CASE_CONV3D_S8S8_2, 2, 2},
                                bc_test_params{CASE_CONV3D_S8S8_3, 2, 2},
                                bc_test_params{CASE_CONV3D_S8S8_4, 2, 2},
                        }));

class conv_int8_asymmetric_data : public ConvFusingTest {};
TEST_P(conv_int8_asymmetric_data, basic) {
    auto p = GetParam();
    auto weights_format = (p.weights_format == format::goiyx) ? format::bfyx : format::bfzyx;
    auto weights_layout = (p.groups > 1) ? get_weights_layout(p, 1, weights_format) :
                          get_weights_layout(p);
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(weights_layout)),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("a_zp", get_mem(get_activations_zp_layout(p), 1, 127)),
                 eltwise("a_sub", {"input", "a_zp"}, eltwise_mode::sub, data_types::f32),
                 convolution("conv_prim", "a_sub", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation, p.out_shape, data_types::f32, false),
                 reorder("reorder_bfyx", "conv_prim", p.default_format, data_types::f32)
    );
    tolerance = 1.f;

    auto input_prim = get_mem(get_input_layout(p));
    network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
    network network_fused(this->engine, this->topology_fused, bo_fused);
    network_fused.set_input_data("input", input_prim);
    network_not_fused.set_input_data("input", input_prim);

    ASSERT_FALSE(network_fused.get_primitives_info().empty());
    ASSERT_FALSE(network_not_fused.get_primitives_info().empty());

    // Search for both conv_prim and reorder_bfyx, as in case of fused topology convolution will be merged with the last reorder
    auto find_conv = [](primitive_info& p) -> bool {
        if (p.original_id == "conv_prim" || p.original_id == "reorder_bfyx")
            return true;
        return false;
    };

    auto pi_fused = network_fused.get_primitives_info();
    auto pi_not_fused = network_not_fused.get_primitives_info();
    auto info_fused = std::find_if(pi_fused.begin(), pi_fused.end(), find_conv);
    auto info_not_fused = std::find_if(pi_not_fused.begin(), pi_not_fused.end(), find_conv);

    ASSERT_TRUE(info_fused != pi_fused.end());
    ASSERT_TRUE(info_not_fused != pi_not_fused.end());

    ASSERT_EQ(info_fused->c_dependencies.size(), 5lu);  // input + weights + bias + a_zp + comp
    ASSERT_EQ(info_not_fused->c_dependencies.size(), 3lu);  // input + weights + bias

    compare(network_not_fused, network_fused, p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_asymmetric_data,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_U8S8_1, 2, 3},
                                bc_test_params{CASE_CONV_U8S8_2, 2, 3},
                                bc_test_params{CASE_CONV_U8S8_3, 2, 3},
                                bc_test_params{CASE_CONV_U8S8_4, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_1, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_2, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_3, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_4, 2, 3},

                                bc_test_params{CASE_CONV3D_U8S8_1, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_2, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_3, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_4, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_5, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_1, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_2, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_3, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_4, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_5, 2, 3},
                        }));

class conv_int8_asymmetric_data_and_weights : public ConvFusingTest {};
TEST_P(conv_int8_asymmetric_data_and_weights, basic) {
    auto p = GetParam();
    auto weights_format = (p.weights_format == format::goiyx) ? format::bfyx : format::bfzyx;
    auto weights_layout = (p.groups > 1) ? get_weights_layout(p, 1, weights_format) :
                          get_weights_layout(p);
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(weights_layout)),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("a_zp", get_mem(get_activations_zp_layout(p), 1, 127)),
                 data("w_zp", get_mem(get_weights_zp_layout(p), 1, 127)),
                 eltwise("a_sub", {"input", "a_zp"}, eltwise_mode::sub, data_types::f32),
                 eltwise("w_sub", {"weights", "w_zp"}, eltwise_mode::sub, data_types::f32),
                 convolution("conv_prim", "a_sub", {"w_sub"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation, p.out_shape, data_types::f32, false),
                 reorder("reorder_bfyx", "conv_prim", p.default_format, data_types::f32)
    );
    tolerance = 1.f;

    auto input_prim = get_mem(get_input_layout(p));
    network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
    network network_fused(this->engine, this->topology_fused, bo_fused);
    network_fused.set_input_data("input", input_prim);
    network_not_fused.set_input_data("input", input_prim);

    ASSERT_FALSE(network_fused.get_primitives_info().empty());
    ASSERT_FALSE(network_not_fused.get_primitives_info().empty());

    // Search for both conv_prim and reorder_bfyx, as in case of fused topology convolution will be merged with the last reorder
    auto find_conv = [](primitive_info& p) -> bool {
        if (p.original_id == "conv_prim" || p.original_id == "reorder_bfyx")
            return true;
        return false;
    };

    auto pi_fused = network_fused.get_primitives_info();
    auto pi_not_fused = network_not_fused.get_primitives_info();
    auto info_fused = std::find_if(pi_fused.begin(), pi_fused.end(), find_conv);
    auto info_not_fused = std::find_if(pi_not_fused.begin(), pi_not_fused.end(), find_conv);

    ASSERT_TRUE(info_fused != pi_fused.end());
    ASSERT_TRUE(info_not_fused != pi_not_fused.end());

    ASSERT_EQ(info_fused->c_dependencies.size(), 6lu);  // input + weights + bias + a_zp + w_zp + comp
    ASSERT_EQ(info_not_fused->c_dependencies.size(), 3lu);  // input + weights + bias

    compare(network_not_fused, network_fused, p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_int8_asymmetric_data_and_weights,
                        ::testing::ValuesIn(std::vector<bc_test_params>{
                                bc_test_params{CASE_CONV_U8S8_1, 2, 3},
                                bc_test_params{CASE_CONV_U8S8_2, 2, 3},
                                bc_test_params{CASE_CONV_U8S8_3, 2, 3},
                                bc_test_params{CASE_CONV_U8S8_4, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_1, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_2, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_3, 2, 3},
                                bc_test_params{CASE_CONV_S8S8_4, 2, 3},

                                bc_test_params{CASE_CONV3D_U8S8_1, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_2, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_3, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_4, 2, 3},
                                bc_test_params{CASE_CONV3D_U8S8_5, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_1, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_2, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_3, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_4, 2, 3},
                                bc_test_params{CASE_CONV3D_S8S8_5, 2, 3},
                        }));


class conv_i8_activation_eltwise_diff_sizes : public ConvEltwTest {};
TEST_P(conv_i8_activation_eltwise_diff_sizes, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 data("eltwise_data", get_mem(layout{ p.data_type, p.input_format, p.eltw_shape })),
                 convolution("conv_prim", "input", { "weights" }, { "bias" }, p.groups, p.stride, p.pad, p.dilation),
                 activation("activation", "conv_prim", activation_func::abs),
                 eltwise("sum", { "activation", "eltwise_data" }, eltwise_mode::sum, data_types::f32),
                 reorder("reorder_bfyx", "sum", p.default_format, data_types::f32)
    );
    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_i8_activation_eltwise_diff_sizes,
                        ::testing::ValuesIn(std::vector<conv_eltw_test_params>{
                                conv_eltw_test_params{CASE_CONV_ELTW_i8_1, 3, 4},
                                conv_eltw_test_params{CASE_CONV_ELTW_i8_2, 2, 4},
                                conv_eltw_test_params{CASE_CONV_ELTW_i8_3, 2, 4},
                                conv_eltw_test_params{CASE_CONV_ELTW_i8_4, 2, 4},
                                conv_eltw_test_params{CASE_CONV_ELTW_i8_5, 3, 4},
                        }));

/* ----------------------------------------------------------------------------------------------------- */
/* ----------------------------------- Force convolution kernel cases ---------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
class ConvFusingForceKernelTest : public ::WeightsPrimitiveFusingTest<bc_force_kernel_params> {
    public:
    void execute(bc_force_kernel_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        build_options options;
        options.set_option(build_option::optimize_data(true));
        implementation_desc conv_impl = {p.input_format, p.kernel_name};
        options.set_option(build_option::force_implementations({{"conv_prim", conv_impl}}));

        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, options);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
        auto find_conv = [](primitive_info& p) -> bool {
            if (p.original_id == "conv_prim")
                return true;
            return false;
        };

        auto pi_fused = network_fused.get_primitives_info();
        auto info_fused = std::find_if(pi_fused.begin(), pi_fused.end(), find_conv);
        if (info_fused != pi_fused.end())
            std::cout << "kernel: " << info_fused->kernel_id << std::endl;
    }
};

class conv_fp16_activation : public ConvFusingForceKernelTest {};
TEST_P(conv_fp16_activation, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("bias", get_mem(get_bias_layout(p))),
                 convolution("conv_prim", "input", {"weights"}, {"bias"}, p.groups, p.stride, p.pad, p.dilation),
                 activation("activation", "conv_prim", activation_func::abs),
                 reorder("reorder_bfyx", "activation", p.default_format, data_types::f32)
    );

    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp16_activation, ::testing::ValuesIn(std::vector<bc_force_kernel_params>{
                                                                           bc_force_kernel_params{CASE_CONV_FP16_13, 2, 3, "convolution_gpu_fs_byx_fsv32"},
                                                                           }));


class conv_fp16_scale : public ConvFusingForceKernelTest {};
TEST_P(conv_fp16_scale, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
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

INSTANTIATE_TEST_SUITE_P(fusings_gpu, conv_fp16_scale,
                        ::testing::ValuesIn(std::vector<bc_force_kernel_params>{
                                             bc_force_kernel_params{CASE_CONV_FP16_13, 2, 3, "convolution_gpu_fs_byx_fsv32"},
                                             }));

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- FC cases --------------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
class FCFusingTest : public WeightsPrimitiveFusingTest<bc_test_params> {};
class fc_fp32_activation : public FCFusingTest {};
TEST_P(fc_fp32_activation, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                data("weights", get_mem(get_fc_weights_layout(p))),
                data("bias", get_mem(get_fc_bias_layout(p))),
                fully_connected("fc_prim", "input", "weights", "bias", padding(), get_fc_output_dim_size(p)),
                activation("activation", "fc_prim", activation_func::abs),
                reorder("reorder_bfyx", "activation", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_fp32_activation, ::testing::ValuesIn(std::vector<bc_test_params>{
                                                                            bc_test_params{ CASE_FC_FP32_1, 2, 3 },
                                                                            bc_test_params{ CASE_FC_FP32_2, 2, 3 },
                                                                            bc_test_params{ CASE_FC_FP32_3, 2, 3 },
                                                                            bc_test_params{ CASE_FC_FP32_3D_1, 2, 3 },
                                                                            bc_test_params{ CASE_FC_FP32_3D_2, 2, 3 },
                                                                            bc_test_params{ CASE_FC_FP32_3D_3, 2, 3 },
}));

class fc_fp32_bias : public FCFusingTest {};
TEST_P(fc_fp32_bias, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                data("weights", get_mem(get_fc_weights_layout(p))),
                data("bias", get_mem(get_fc_bias_layout(p))),
                fully_connected("fc_prim", "input", "weights", ""),
                eltwise("bias_add", {"fc_prim", "bias"}, eltwise_mode::sum),
                reorder("reorder_bfyx", "bias_add", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_fp32_bias, ::testing::ValuesIn(std::vector<bc_test_params>{
                                                                            bc_test_params{ CASE_FC_FP32_1, 2, 3 },
                                                                            bc_test_params{ CASE_FC_FP32_2, 2, 3 },
                                                                            bc_test_params{ CASE_FC_FP32_3, 2, 3 },
}));

class fc_int8_scale : public FCFusingTest {};
TEST_P(fc_int8_scale, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_fc_weights_layout(p))),
        data("bias", get_mem(get_fc_bias_layout(p))),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / p.kernel.count())),
        fully_connected("fc_prim", "input", "weights", "bias", data_types::f32, padding(), get_fc_output_dim_size(p)),
        scale("scale", "fc_prim", "scale_data"),
        reorder("reorder_bfyx", "scale", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(fc_int8_scale, fp16_scale_out) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_fc_weights_layout(p))),
        data("bias", get_mem(get_fc_bias_layout(p))),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / p.kernel.count())),
        fully_connected("fc_prim", "input", "weights", "bias", data_types::f32, padding(), get_fc_output_dim_size(p)),
        scale("scale", "fc_prim", "scale_data", optional_data_type{data_types::f16}),
        reorder("reorder_bfyx", "scale", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_int8_scale,
    ::testing::ValuesIn(std::vector<bc_test_params>{
                        bc_test_params{ CASE_FC_U8S8_1, 2, 3 },
                        bc_test_params{ CASE_FC_U8S8_2, 2, 3 },
                        bc_test_params{ CASE_FC_U8S8_3, 2, 3 },
                        bc_test_params{ CASE_FC_U8S8_3D_1, 2, 3 },
                        bc_test_params{ CASE_FC_U8S8_3D_2, 2, 3 },
                        bc_test_params{ CASE_FC_U8S8_3D_3, 2, 3 },
                        }));

class fc_int8_quantize_u8 : public FCFusingTest {};
TEST_P(fc_int8_quantize_u8, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_fc_weights_layout(p))),
        data("bias", get_mem(get_fc_bias_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        fully_connected("fc_prim", "input", "weights", "bias", data_types::f32, padding(), get_fc_output_dim_size(p)),
        quantize("quantize", "fc_prim", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::u8),
        reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu_fc, fc_int8_quantize_u8,
    ::testing::ValuesIn(std::vector<bc_test_params>{
        bc_test_params{CASE_FC_U8S8_1, 2, 3},
        bc_test_params{CASE_FC_U8S8_2, 2, 3},
        bc_test_params{CASE_FC_U8S8_3, 2, 3},
        bc_test_params{ CASE_FC_U8S8_3D_1, 2, 3 },
        bc_test_params{ CASE_FC_U8S8_3D_2, 2, 3 },
        bc_test_params{ CASE_FC_U8S8_3D_3, 2, 3 },
        }));

class fc_int8_scale_quantize_i8 : public FCFusingTest {};
TEST_P(fc_int8_scale_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_fc_weights_layout(p))),
        data("bias", get_mem(get_fc_bias_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / p.kernel.count() / 255)),
        fully_connected("fc_prim", "input", "weights", "bias", data_types::f32, padding(), get_fc_output_dim_size(p)),
        scale("scale", "fc_prim", "scale_data"),
        quantize("quantize", "scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );
    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_int8_scale_quantize_i8,
    ::testing::ValuesIn(std::vector<bc_test_params>{
        bc_test_params{CASE_FC_U8S8_1, 2, 4},
        bc_test_params{CASE_FC_U8S8_2, 2, 4},
        bc_test_params{CASE_FC_U8S8_3, 2, 4},
        bc_test_params{ CASE_FC_U8S8_3D_1, 2, 4 },
        bc_test_params{ CASE_FC_U8S8_3D_2, 2, 4 },
        bc_test_params{ CASE_FC_U8S8_3D_3, 2, 4 },
        }));



class fc_int8_scale_activation_quantize_i8 : public FCFusingTest {};
TEST_P(fc_int8_scale_activation_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_fc_weights_layout(p))),
        data("bias", get_mem(get_fc_bias_layout(p))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / p.kernel.count() / 255)),
        fully_connected("fc_prim", "input", "weights", "bias", data_types::f32, padding(), get_fc_output_dim_size(p)),
        scale("scale", "fc_prim", "scale_data"),
        activation("activation_scale", "scale", activation_func::exp),
        quantize("quantize", "activation_scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, fc_int8_scale_activation_quantize_i8,
    ::testing::ValuesIn(std::vector<bc_test_params>{
        bc_test_params{CASE_FC_U8S8_1, 2, 5},
        bc_test_params{CASE_FC_U8S8_2, 2, 5},
        bc_test_params{CASE_FC_U8S8_3, 2, 5},

        bc_test_params{ CASE_FC_U8S8_3D_1, 2, 5 },
        bc_test_params{ CASE_FC_U8S8_3D_2, 2, 5 },
        bc_test_params{ CASE_FC_U8S8_3D_3, 2, 5 },

        bc_test_params{ CASE_FC_FP32_3D_1, 3, 5 },
        bc_test_params{ CASE_FC_FP32_3D_2, 3, 5 },
        bc_test_params{ CASE_FC_FP32_3D_3, 3, 5 },
        }));


/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- Gemm cases ------------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
#define CASE_GEMM_3IN_FP32_1 {{1, 1, 2, 2}, {1, 1, 2, 2}, {1, 1, 2, 2}}, {1, 1, 2, 2}, tensor{1}, tensor{0}, data_types::f32, data_types::f32, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_3IN_FP16_1 {{1, 1, 2, 2}, {1, 1, 2, 2}, {1, 1, 2, 2}}, {1, 1, 2, 2}, tensor{1}, tensor{0}, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_3IN_S8S8_1 {{1, 1, 2, 2}, {1, 1, 2, 2}, {1, 1, 2, 2}}, {1, 1, 2, 2}, tensor{1}, tensor{0}, data_types::i8, data_types::i8, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_3IN_S8S8_2 {{1, 2, 64, 128}, {1, 2, 256, 64}, {1, 2, 256, 128}}, {1, 2, 256, 128}, tensor{1}, tensor{0}, data_types::i8, data_types::i8, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_3IN_S8S8_3 {{1, 1, 8, 16}, {1, 1, 32, 8}, {1, 1, 32, 16}}, {1, 1, 32, 16}, tensor{1}, tensor{0}, data_types::i8, data_types::i8, data_types::i8, format::bfyx, data_types::f32, format::bfyx

#define CASE_GEMM_2IN_FP32_1 {{1, 1, 2, 2}, {1, 1, 2, 2}}, {1, 1, 2, 2}, tensor{1}, tensor{0}, data_types::f32, data_types::f32, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_2IN_FP16_1 {{1, 1, 2, 2}, {1, 1, 2, 2}}, {1, 1, 2, 2}, tensor{1}, tensor{0}, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_2IN_U8U8_1 {{1, 1, 2, 2}, {1, 1, 2, 2}}, {1, 1, 2, 2}, tensor{1}, tensor{0}, data_types::u8, data_types::u8, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_2IN_U8U8_2 {{1, 2, 64, 128}, {1, 2, 256, 64}}, {1, 2, 256, 128}, tensor{1}, tensor{0}, data_types::u8, data_types::u8, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_2IN_U8U8_3 {{1, 1, 16, 32}, {1, 1, 32, 16}}, {1, 1, 32, 32}, tensor{1}, tensor{0}, data_types::u8, data_types::u8, data_types::u8, format::bfyx, data_types::f32, format::bfyx

#define CASE_GEMM_2IN_U8S8_1 {{1, 1, 4, 2}, {1, 1, 8, 4}}, {1, 1, 8, 4}, tensor{1}, tensor{0}, data_types::u8, data_types::i8, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_2IN_S8U8_1 {{1, 2, 64, 128}, {1, 2, 256, 64}}, {1, 2, 256, 128}, tensor{1}, tensor{0}, data_types::i8, data_types::u8, data_types::u8, format::bfyx, data_types::f32, format::bfyx

#define CASE_GEMM_ELTWISE_2IN_FP32_1 {{1, 1, 4, 4}, {1, 1, 4, 4}}, {1, 1, 4, 4}, tensor{1}, tensor{0}, data_types::f32, data_types::f32, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_ELTWISE_2IN_FP16_1 {{1, 1, 32, 32}, {1, 1, 32, 32}}, {1, 1, 32, 32}, tensor{1}, tensor{0}, data_types::f16, data_types::f16, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GEMM_ELTWISE_2IN_U8S8_1 {{1, 1, 4, 4}, {1, 1, 4, 4}}, {1, 1, 4, 4}, tensor{1}, tensor{0}, data_types::u8, data_types::i8, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_GEMM_ELTWISE_2IN_S8U8_1 {{1, 1, 32, 32}, {1, 1, 32, 32}}, {1, 1, 32, 32}, tensor{1}, tensor{0}, data_types::i8, data_types::u8, data_types::u8, format::bfyx, data_types::f32, format::bfyx

class gemm_3in_quantize_i8 : public GemmFusingTest {};
TEST_P(gemm_3in_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        input_layout("input2", get_input_layout(p, 2)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        gemm("gemm_prim", { "input0", "input1", "input2" }, data_types::f32),
        quantize("quantize", "gemm_prim", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gemm_3in_quantize_i8,
    ::testing::ValuesIn(std::vector<gemm_test_params>{
                        gemm_test_params{ CASE_GEMM_3IN_FP16_1, 4, 5 },
                        gemm_test_params{ CASE_GEMM_3IN_S8S8_1, 4, 5 },
                        gemm_test_params{ CASE_GEMM_3IN_S8S8_2, 4, 5 },
                        gemm_test_params{ CASE_GEMM_3IN_S8S8_3, 4, 5 },
                      //gemm_test_params{ CASE_GEMM_3IN_FP32_1, 4, 5 },
}));

class gemm_2in_quantize_u8 : public GemmFusingTest {};
TEST_P(gemm_2in_quantize_u8, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        data("in_lo", get_mem(get_per_channel_layout(p), 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        gemm("gemm_prim", { "input0", "input1" }, data_types::f32),
        quantize("quantize", "gemm_prim", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::u8),
        reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gemm_2in_quantize_u8,
    ::testing::ValuesIn(std::vector<gemm_test_params>{
                        gemm_test_params{ CASE_GEMM_2IN_FP16_1, 3, 4 },
                        gemm_test_params{ CASE_GEMM_2IN_U8U8_1, 3, 4 },
                        gemm_test_params{ CASE_GEMM_2IN_U8U8_2, 3, 4 },
                        gemm_test_params{ CASE_GEMM_2IN_U8U8_3, 3, 4 },
                      //gemm_test_params{ CASE_GEMM_2IN_FP32_1, 3, 4 },
}));

class gemm_2in_scale : public GemmFusingTest {};
TEST_P(gemm_2in_scale, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count())),
        gemm("gemm_prim", { "input0", "input1" }, data_types::f32),
        scale("scale", "gemm_prim", "scale_data"),
        reorder("reorder_bfyx", "scale", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(gemm_2in_scale, fp16_scale_out) {
    auto p = GetParam();
    create_topologies(input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count())),
        gemm("gemm_prim", { "input0", "input1" }, data_types::f32),
        scale("scale", "gemm_prim", "scale_data", optional_data_type{data_types::f16}),
        reorder("reorder_bfyx", "scale", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gemm_2in_scale,
    ::testing::ValuesIn(std::vector<gemm_test_params>{
                        gemm_test_params{ CASE_GEMM_2IN_FP32_1, 3, 4 },
                        gemm_test_params{ CASE_GEMM_2IN_FP16_1, 3, 4 },
                        gemm_test_params{ CASE_GEMM_2IN_U8U8_1, 3, 4 },
                        gemm_test_params{ CASE_GEMM_2IN_U8U8_2, 3, 4 },
                        gemm_test_params{ CASE_GEMM_2IN_U8U8_3, 3, 4 },
}));

class gemm_2in_act_scale_quantize_i8 : public GemmFusingTest {};
TEST_P(gemm_2in_act_scale_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / p.kernel.count() / 255)),
        gemm("gemm_prim", { "input0", "input1" }, data_types::f32),
        activation("activation", "gemm_prim", activation_func::exp),
        scale("scale", "activation", "scale_data"),
        quantize("quantize", "scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gemm_2in_act_scale_quantize_i8,
    ::testing::ValuesIn(std::vector<gemm_test_params>{
                        gemm_test_params{ CASE_GEMM_2IN_FP32_1, 3, 6 },
                        gemm_test_params{ CASE_GEMM_2IN_FP16_1, 3, 6 },
                        gemm_test_params{ CASE_GEMM_2IN_U8S8_1, 3, 6 },
                        gemm_test_params{ CASE_GEMM_2IN_S8U8_1, 3, 6 },
}));

class gemm_2in_act_scale_quantize_eltwise_i8 : public GemmFusingTest {};
TEST_P(gemm_2in_act_scale_quantize_eltwise_i8, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / p.kernel.count() / 255)),
        data("eltwise_data", get_mem(get_output_layout(p))),
        gemm("gemm_prim", { "input0", "input1" }, data_types::f32),
        activation("activation", "gemm_prim", activation_func::exp),
        scale("scale", "activation", "scale_data"),
        quantize("quantize", "scale", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        eltwise("sum", { "quantize", "eltwise_data"}, eltwise_mode::sum,  data_types::f32),
        reorder("reorder_bfyx", "sum", p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gemm_2in_act_scale_quantize_eltwise_i8,
    ::testing::ValuesIn(std::vector<gemm_test_params>{
                        gemm_test_params{ CASE_GEMM_ELTWISE_2IN_FP32_1, 3, 7 },
                        gemm_test_params{ CASE_GEMM_ELTWISE_2IN_FP16_1, 3, 7 },
                        gemm_test_params{ CASE_GEMM_ELTWISE_2IN_U8S8_1, 3, 7 },
                        gemm_test_params{ CASE_GEMM_ELTWISE_2IN_S8U8_1, 3, 7 },
}));

class gemm_2in_act_scale_eltwise : public GemmFusingTest {};
TEST_P(gemm_2in_act_scale_eltwise, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / p.kernel.count() / 255)),
        data("eltwise_data", get_mem(get_output_layout(p))),
        gemm("gemm_prim", { "input0", "input1" }, data_types::f32),
        scale("scale", "gemm_prim", "scale_data"),
        activation("activation", "scale", activation_func::negative),
        eltwise("sum", { "activation", "eltwise_data"}, eltwise_mode::sum,  data_types::f32),
        reorder("reorder_bfyx", "sum", p.default_format, data_types::f32)
    );

    tolerance = 1e-4f;
    execute(p);
}

TEST_P(gemm_2in_act_scale_eltwise, broadcast_eltwise) {
    auto p = GetParam();
    create_topologies(input_layout("input0", get_input_layout(p, 0)),
        input_layout("input1", get_input_layout(p, 1)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / p.kernel.count() / 255)),
        data("eltwise_data", get_mem(get_single_element_layout(p))),
        gemm("gemm_prim", { "input0", "input1" }, data_types::f32),
        scale("scale", "gemm_prim", "scale_data"),
        activation("activation", "scale", activation_func::negative),
        eltwise("sum", { "activation", "eltwise_data"}, eltwise_mode::sum,  data_types::f32),
        reorder("reorder_bfyx", "sum", p.default_format, data_types::f32)
    );

    tolerance = 1e-4f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gemm_2in_act_scale_eltwise,
    ::testing::ValuesIn(std::vector<gemm_test_params>{
                        gemm_test_params{ CASE_GEMM_ELTWISE_2IN_FP32_1, 3, 6 },
                        gemm_test_params{ CASE_GEMM_ELTWISE_2IN_FP16_1, 3, 6 },
                        gemm_test_params{ CASE_GEMM_ELTWISE_2IN_U8S8_1, 3, 6 },
                        gemm_test_params{ CASE_GEMM_ELTWISE_2IN_S8U8_1, 3, 6 },
}));

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- Resample cases --------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
#define CASE_RESAMPLE_FP32_1 {1, 15, 4, 5}, {1, 15, 2, 3}, data_types::f32, format::bfyx, resample_type::nearest, data_types::f32, format::bfyx
#define CASE_RESAMPLE_FP32_2 {1, 15, 4, 5}, {1, 15, 2, 3}, data_types::f32, format::bfyx, resample_type::bilinear, data_types::f32, format::bfyx
#define CASE_RESAMPLE_FP32_3 {1, 15, 4, 5}, {1, 15, 2, 3}, data_types::f32, format::bfyx, resample_type::caffe_bilinear, data_types::f32, format::bfyx
#define CASE_RESAMPLE_FP32_4 {1, 16, 4, 5}, {1, 16, 7, 8}, data_types::f32, format::bfyx, resample_type::nearest, data_types::f32, format::bfyx
#define CASE_RESAMPLE_FP32_5 {1, 16, 4, 5}, {1, 16, 7, 8}, data_types::f32, format::bfyx, resample_type::bilinear, data_types::f32, format::bfyx
#define CASE_RESAMPLE_FP32_6 {1, 16, 4, 5}, {1, 16, 7, 8}, data_types::f32, format::bfyx, resample_type::caffe_bilinear, data_types::f32, format::bfyx
#define CASE_RESAMPLE_FP32_7 {1, 16, 4, 5, 4}, {1, 16, 2, 3, 2}, data_types::f32, format::bfzyx, resample_type::nearest, data_types::f32, format::bfzyx
#define CASE_RESAMPLE_FP32_8 {1, 16, 4, 5, 4}, {1, 16, 2, 3, 2}, data_types::f32, format::bfzyx, resample_type::caffe_bilinear, data_types::f32, format::bfzyx
#define CASE_RESAMPLE_FP32_9 {1, 16, 4, 5}, {1, 16, 7, 8}, data_types::f32, format::b_fs_yx_fsv16, resample_type::bilinear, data_types::f32, format::bfyx
#define CASE_RESAMPLE_FP32_10 {1, 16, 4, 5}, {1, 16, 7, 8}, data_types::f32, format::b_fs_yx_fsv16, resample_type::caffe_bilinear, data_types::f32, format::bfyx

#define CASE_RESAMPLE_FP16_1 {1, 15, 4, 5}, {1, 15, 2, 3}, data_types::f16, format::bfyx, resample_type::nearest, data_types::f16, format::bfyx
#define CASE_RESAMPLE_FP16_2 {1, 15, 4, 5}, {1, 15, 2, 3}, data_types::f16, format::bfyx, resample_type::bilinear, data_types::f16, format::bfyx
#define CASE_RESAMPLE_FP16_3 {1, 15, 4, 5}, {1, 15, 2, 3}, data_types::f16, format::bfyx, resample_type::caffe_bilinear, data_types::f16, format::bfyx
#define CASE_RESAMPLE_FP16_4 {1, 16, 4, 5}, {1, 16, 7, 8}, data_types::f16, format::bfyx, resample_type::nearest, data_types::f16, format::bfyx
#define CASE_RESAMPLE_FP16_5 {1, 16, 4, 5}, {1, 16, 7, 8}, data_types::f16, format::bfyx, resample_type::bilinear, data_types::f16, format::bfyx
#define CASE_RESAMPLE_FP16_6 {1, 16, 4, 5}, {1, 16, 7, 8}, data_types::f16, format::bfyx, resample_type::caffe_bilinear, data_types::f16, format::bfyx
#define CASE_RESAMPLE_FP16_7 {1, 16, 4, 5, 4}, {1, 16, 2, 3, 2}, data_types::f16, format::bfzyx, resample_type::nearest, data_types::f16, format::bfzyx
#define CASE_RESAMPLE_FP16_8 {1, 16, 4, 5, 4}, {1, 16, 2, 3, 2}, data_types::f16, format::bfzyx, resample_type::caffe_bilinear, data_types::f16, format::bfzyx
#define CASE_RESAMPLE_FP16_9 {1, 16, 4, 5}, {1, 16, 7, 8}, data_types::f16, format::b_fs_yx_fsv16, resample_type::bilinear, data_types::f16, format::bfyx
#define CASE_RESAMPLE_FP16_10 {2, 32, 4, 5}, {2, 32, 7, 8}, data_types::f16, format::fs_b_yx_fsv32, resample_type::bilinear, data_types::f16, format::bfyx
#define CASE_RESAMPLE_FP16_11 {1, 16, 4, 5}, {1, 16, 7, 8}, data_types::f16, format::b_fs_yx_fsv16, resample_type::caffe_bilinear, data_types::f16, format::bfyx
#define CASE_RESAMPLE_FP16_12 {2, 32, 4, 5}, {2, 32, 7, 8}, data_types::f16, format::fs_b_yx_fsv32, resample_type::caffe_bilinear, data_types::f16, format::bfyx
#define CASE_RESAMPLE_FP16_13 {1, 16, 4, 5}, {1, 16, 7, 8}, data_types::f16, format::b_fs_yx_fsv16, resample_type::caffe_bilinear, data_types::f16, format::bfyx
#define CASE_RESAMPLE_FP16_14 {1, 32, 4, 5}, {1, 32, 2, 3}, data_types::f16, format::fs_b_yx_fsv32, resample_type::caffe_bilinear, data_types::f16, format::bfyx

#define CASE_RESAMPLE_I8_1 {1, 16, 4, 5}, {1, 16, 2, 3}, data_types::i8, format::b_fs_yx_fsv16, resample_type::nearest, data_types::f32, format::bfyx
#define CASE_RESAMPLE_I8_2 {2, 32, 4, 5}, {2, 32, 2, 3}, data_types::i8, format::b_fs_yx_fsv16, resample_type::nearest, data_types::f32, format::bfyx
#define CASE_RESAMPLE_I8_3 {1, 16, 4, 5}, {1, 16, 2, 3}, data_types::i8, format::b_fs_yx_fsv16, resample_type::bilinear, data_types::f32, format::bfyx
#define CASE_RESAMPLE_I8_4 {2, 32, 4, 5}, {2, 32, 2, 3}, data_types::i8, format::b_fs_yx_fsv16, resample_type::bilinear, data_types::f32, format::bfyx

#define CASE_RESAMPLE_U8_1 {1, 16, 4, 5}, {1, 16, 2, 3}, data_types::u8, format::b_fs_yx_fsv16, resample_type::nearest, data_types::f32, format::bfyx
#define CASE_RESAMPLE_U8_2 {2, 32, 4, 5}, {2, 32, 2, 3}, data_types::u8, format::b_fs_yx_fsv16, resample_type::nearest, data_types::f32, format::bfyx
#define CASE_RESAMPLE_U8_3 {1, 16, 4, 5}, {1, 16, 2, 3}, data_types::u8, format::b_fs_yx_fsv16, resample_type::bilinear, data_types::f32, format::bfyx
#define CASE_RESAMPLE_U8_4 {2, 32, 4, 5}, {2, 32, 2, 3}, data_types::u8, format::b_fs_yx_fsv16, resample_type::bilinear, data_types::f32, format::bfyx

class resample_quantize : public ResamplePrimitiveFusingTest {};
TEST_P(resample_quantize, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        resample("resample_prim", "input", p.out_shape, p.in_shape.feature[0], p.type),
        quantize("quantize", "resample_prim", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, resample_quantize,
    ::testing::ValuesIn(std::vector<resample_test_params>{
                        resample_test_params{ CASE_RESAMPLE_FP32_1, 2, 3 },
                        resample_test_params{ CASE_RESAMPLE_FP32_2, 2, 3 },
                        resample_test_params{ CASE_RESAMPLE_FP32_3, 2, 3 },
                        resample_test_params{ CASE_RESAMPLE_FP32_4, 2, 3 },
                        resample_test_params{ CASE_RESAMPLE_FP32_5, 2, 3 },
                        resample_test_params{ CASE_RESAMPLE_FP32_6, 2, 3 },
                        resample_test_params{ CASE_RESAMPLE_FP32_7, 2, 3 },
                        resample_test_params{ CASE_RESAMPLE_FP32_8, 2, 3 },
                        resample_test_params{ CASE_RESAMPLE_FP32_9, 2, 3 },
                        resample_test_params{ CASE_RESAMPLE_FP32_10, 2, 3 },

                        // FQ can't be fused to FP16 primitive for now
                        // resample_test_params{ CASE_RESAMPLE_FP16_1, 2, 3 },
                        // resample_test_params{ CASE_RESAMPLE_FP16_2, 2, 3 },
                        // resample_test_params{ CASE_RESAMPLE_FP16_3, 2, 3 },
                        // resample_test_params{ CASE_RESAMPLE_FP16_4, 2, 3 },
                        // resample_test_params{ CASE_RESAMPLE_FP16_5, 2, 3 },
                        // resample_test_params{ CASE_RESAMPLE_FP16_6, 2, 3 },
                        // resample_test_params{ CASE_RESAMPLE_FP16_7, 2, 3 },
                        // resample_test_params{ CASE_RESAMPLE_FP16_8, 2, 3 },
                        // resample_test_params{ CASE_RESAMPLE_FP16_9, 2, 3 },
}));

class resample_scale_activation_eltwise : public ResamplePrimitiveFusingTest {};
TEST_P(resample_scale_activation_eltwise, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
        data("scale_data", get_mem(get_per_channel_layout(p), -10, 10)),
        data("eltwise_data", get_mem(get_output_layout(p), -10, 10)),
        resample("resample_prim", "input", p.out_shape, p.in_shape.feature[0], p.type),
        scale("scale", "resample_prim", "scale_data"),
        activation("activation", "scale", activation_func::abs),
        eltwise("eltwise", { "activation", "eltwise_data"}, eltwise_mode::sum),
        reorder("reorder_bfyx", "eltwise", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, resample_scale_activation_eltwise,
    ::testing::ValuesIn(std::vector<resample_test_params>{
                        resample_test_params{ CASE_RESAMPLE_FP32_1, 2, 5 },
                        resample_test_params{ CASE_RESAMPLE_FP32_2, 2, 5 },
                        resample_test_params{ CASE_RESAMPLE_FP32_3, 2, 5 },
                        resample_test_params{ CASE_RESAMPLE_FP32_4, 2, 5 },
                        resample_test_params{ CASE_RESAMPLE_FP32_5, 2, 5 },
                        resample_test_params{ CASE_RESAMPLE_FP32_6, 2, 5 },
                        resample_test_params{ CASE_RESAMPLE_FP32_7, 2, 5 },
                        resample_test_params{ CASE_RESAMPLE_FP32_8, 2, 5 },
                        resample_test_params{ CASE_RESAMPLE_FP32_9, 2, 5 },
                        resample_test_params{ CASE_RESAMPLE_FP32_10, 2, 5 },

                        resample_test_params{ CASE_RESAMPLE_FP16_1, 2, 5 },
                        resample_test_params{ CASE_RESAMPLE_FP16_2, 2, 5 },
                        resample_test_params{ CASE_RESAMPLE_FP16_3, 2, 5 },
                        resample_test_params{ CASE_RESAMPLE_FP16_4, 2, 5 },
                        resample_test_params{ CASE_RESAMPLE_FP16_5, 2, 5 },
                        resample_test_params{ CASE_RESAMPLE_FP16_6, 2, 5 },
                        resample_test_params{ CASE_RESAMPLE_FP16_7, 2, 5 },
                        resample_test_params{ CASE_RESAMPLE_FP16_8, 2, 5 },
                        resample_test_params{ CASE_RESAMPLE_FP16_9, 2, 5 },
                        resample_test_params{ CASE_RESAMPLE_FP16_10, 2, 5 },
                        resample_test_params{ CASE_RESAMPLE_FP16_11, 2, 5 },
                        resample_test_params{ CASE_RESAMPLE_FP16_12, 2, 5 },
                        resample_test_params{ CASE_RESAMPLE_FP16_13, 2, 5 },
                        resample_test_params{ CASE_RESAMPLE_FP16_14, 2, 5 },

                        resample_test_params{ CASE_RESAMPLE_I8_1, 2, 5 },
                        resample_test_params{ CASE_RESAMPLE_I8_2, 2, 5 },
                        resample_test_params{ CASE_RESAMPLE_I8_3, 2, 5 },
                        resample_test_params{ CASE_RESAMPLE_I8_4, 2, 5 },

                        resample_test_params{ CASE_RESAMPLE_U8_1, 2, 5 },
                        resample_test_params{ CASE_RESAMPLE_U8_2, 2, 5 },
                        resample_test_params{ CASE_RESAMPLE_U8_3, 2, 5 },
                        resample_test_params{ CASE_RESAMPLE_U8_4, 2, 5 },
}));

class resample_quantize_concat : public ResamplePrimitiveFusingTest {};
TEST_P(resample_quantize_concat, along_f) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        resample("resample1", "input", p.out_shape, p.in_shape.feature[0], p.type),
        data("in_lo_1", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi_1", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo_1", get_mem(get_single_element_layout(p), -128)),
        data("out_hi_1", get_mem(get_single_element_layout(p), 127)),
        quantize("quant1", "resample1", "in_lo_1", "in_hi_1", "out_lo_1", "out_hi_1", 256, data_types::i8),
        resample("resample2", "input", p.out_shape, p.in_shape.feature[0], p.type),
        data("in_lo_2", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi_2", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo_2", get_mem(get_single_element_layout(p), -127)),
        data("out_hi_2", get_mem(get_single_element_layout(p), 127)),
        quantize("quant2", "resample2", "in_lo_2", "in_hi_2", "out_lo_2", "out_hi_2", 255, data_types::i8),
        concatenation("concat", { "quant1", "quant2" }, cldnn::concatenation::along_f),
        reorder("reorder_bfyx", "concat", cldnn::format::bfyx, p.default_type)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, resample_quantize_concat,
    ::testing::ValuesIn(std::vector<resample_test_params>{
                        resample_test_params{ CASE_RESAMPLE_FP32_1, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP32_2, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP32_3, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP32_4, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP32_5, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP32_6, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP32_7, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP32_8, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP32_9, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP32_10, 3, 6 },

                        resample_test_params{ CASE_RESAMPLE_FP16_1, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP16_2, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP16_3, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP16_4, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP16_5, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP16_6, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP16_7, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP16_8, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP16_9, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP16_10, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP16_11, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP16_12, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP16_13, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP16_14, 3, 6 },

                        resample_test_params{ CASE_RESAMPLE_I8_3, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_I8_4, 3, 6 },

                        resample_test_params{ CASE_RESAMPLE_U8_3, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_U8_4, 3, 6 },
}));

class resample_scale_concat : public ResamplePrimitiveFusingTest {};
TEST_P(resample_scale_concat, along_f) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        resample("resample1", "input", p.out_shape, p.in_shape.feature[0], p.type),
        data("scale1_scale", get_mem(get_per_channel_layout(p), -10, 10)),
        data("scale1_shift", get_mem(get_per_channel_layout(p), -10, 10)),
        scale("scale1", "resample1", "scale1_scale", "scale1_shift"),
        resample("resample2", "input", p.out_shape, p.in_shape.feature[0], p.type),
        data("scale2_scale", get_mem(get_per_channel_layout(p), -10, 10)),
        data("scale2_shift", get_mem(get_per_channel_layout(p), -10, 10)),
        scale("scale2", "resample2", "scale2_scale", "scale2_shift"),
        concatenation("concat", { "scale1", "scale2" }, cldnn::concatenation::along_f),
        reorder("reorder_bfyx", "concat", cldnn::format::bfyx, p.default_type)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, resample_scale_concat,
    ::testing::ValuesIn(std::vector<resample_test_params>{
                        resample_test_params{ CASE_RESAMPLE_FP32_1, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP32_2, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP32_3, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP32_4, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP32_5, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP32_6, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP32_7, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP32_8, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP32_9, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP32_10, 3, 6 },

                        resample_test_params{ CASE_RESAMPLE_FP16_1, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP16_2, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP16_3, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP16_4, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP16_5, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP16_6, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP16_7, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP16_8, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP16_9, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP16_10, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP16_11, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP16_12, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP16_13, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_FP16_14, 3, 6 },

                        resample_test_params{ CASE_RESAMPLE_I8_1, 3, 6},
                        resample_test_params{ CASE_RESAMPLE_I8_2, 3, 6},
                        resample_test_params{ CASE_RESAMPLE_I8_3, 3, 6},
                        resample_test_params{ CASE_RESAMPLE_I8_4, 3, 6},

                        resample_test_params{ CASE_RESAMPLE_U8_1, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_U8_2, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_U8_3, 3, 6 },
                        resample_test_params{ CASE_RESAMPLE_U8_4, 3, 6 },
}));

/* ----------------------------------------------------------------------------------------------------- */
/* --------------------------------------- MVN cases --------------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
struct mvn_test_params {
    tensor input_size;
    tensor elwise_size;
    data_types input_type;
    format input_format;
    bool across_channels;
    bool normalize_variance;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

#define CASE_MVN_F32_1      {1, 16, 8, 8},    {1, 16, 8, 8},    data_types::f32, format::bfyx, false, true, data_types::f32, format::bfyx
#define CASE_MVN_F32_2      {2, 16, 8, 8},    {2, 16, 8, 8},    data_types::f32, format::bfyx, true, true, data_types::f32, format::bfyx
#define CASE_MVN_3D_F32_1   {1, 16, 8, 8, 8}, {1, 16, 8, 8, 8}, data_types::f32, format::bfzyx, false, true, data_types::f32, format::bfzyx
#define CASE_MVN_3D_F32_2   {2, 16, 8, 8, 8}, {2, 16, 8, 8, 8}, data_types::f32, format::bfzyx, true, true, data_types::f32, format::bfzyx
#define CASE_MVN_3D_F32_3   {2, 8, 4, 4, 4},  {2, 8, 1, 1, 1},  data_types::f32, format::bfzyx, true, true, data_types::f32, format::bfzyx
#define CASE_MVN_F16_1      {1, 16, 8, 8},    {1, 16, 8, 8},    data_types::f16, format::bfyx, false, true, data_types::f16, format::bfyx
#define CASE_MVN_F16_2      {2, 16, 8, 8},    {2, 16, 8, 8},    data_types::f16, format::bfyx, true, true, data_types::f16, format::bfyx
#define CASE_MVN_3D_F16_1   {1, 16, 8, 8, 8}, {1, 16, 8, 8, 8}, data_types::f16, format::bfzyx, false, true, data_types::f16, format::bfzyx
#define CASE_MVN_3D_F16_2   {2, 16, 8, 8, 8}, {2, 16, 8, 8, 8}, data_types::f16, format::bfzyx, true, true, data_types::f16, format::bfzyx
#define CASE_MVN_I8_1       {1, 16, 8, 8},    {1, 16, 8, 8},    data_types::i8, format::bfyx, false, true, data_types::f32, format::bfyx
#define CASE_MVN_I8_2       {2, 16, 8, 8},    {2, 16, 8, 8},    data_types::i8, format::bfyx, true, true, data_types::f32, format::bfyx
#define CASE_MVN_I8_3       {1, 16, 8, 8},    {1, 16, 8, 8},    data_types::i8, format::b_fs_yx_fsv16, false, true, data_types::f32, format::bfyx
#define CASE_MVN_I8_4       {2, 16, 8, 8},    {2, 16, 8, 8},    data_types::i8, format::b_fs_yx_fsv16, true, true, data_types::f32, format::bfyx
#define CASE_MVN_I8_5       {2, 16, 8, 8},    {1, 1, 1, 8},     data_types::i8, format::b_fs_yx_fsv16, false, true, data_types::f32, format::bfyx
#define CASE_MVN_I8_6       {2, 16, 8, 8},    {1, 1, 1, 1},     data_types::i8, format::b_fs_yx_fsv16, true, true, data_types::f32, format::bfyx
#define CASE_MVN_I8_7       {2, 16, 1, 8},    {1, 1, 8, 1},     data_types::i8, format::b_fs_yx_fsv16, true, true, data_types::f32, format::bfyx
#define CASE_MVN_3D_I8_1    {1, 16, 8, 8, 8}, {1, 16, 8, 8, 8}, data_types::i8, format::bfzyx, false, true, data_types::f32, format::bfzyx
#define CASE_MVN_3D_I8_2    {2, 16, 8, 8, 8}, {2, 16, 8, 8, 8}, data_types::i8, format::bfzyx, true, true, data_types::f32, format::bfzyx
#define CASE_MVN_3D_I8_3    {2, 16, 8, 8, 8}, {2, 1, 8, 8, 1},  data_types::i8, format::bfzyx, true, true, data_types::f32, format::bfzyx
#define CASE_MVN_3D_I8_4    {2, 16, 8, 8, 8}, {2, 16, 8, 1, 8}, data_types::i8, format::bfzyx, false, true, data_types::f32, format::bfzyx
#define CASE_MVN_3D_I8_5    {2, 2, 1, 2, 1},  {2, 2, 2, 2, 2},  data_types::i8, format::bfzyx, false, true, data_types::f32, format::bfzyx
#define CASE_MVN_U8_1       {1, 16, 8, 8},    {1, 16, 8, 8},    data_types::u8, format::bfyx, false, true, data_types::f32, format::bfyx
#define CASE_MVN_U8_2       {2, 16, 8, 8},    {2, 16, 8, 8},    data_types::u8, format::bfyx, true, true, data_types::f32, format::bfyx
#define CASE_MVN_U8_3       {1, 16, 8, 8},    {1, 16, 8, 8},    data_types::u8, format::b_fs_yx_fsv16, false, true, data_types::f32, format::bfyx
#define CASE_MVN_U8_4       {2, 16, 8, 8},    {2, 16, 8, 8},    data_types::u8, format::b_fs_yx_fsv16, true, true, data_types::f32, format::bfyx
#define CASE_MVN_U8_5       {2, 16, 8, 8},    {2, 1, 8, 8},     data_types::u8, format::b_fs_yx_fsv16, false, true, data_types::f32, format::bfyx
#define CASE_MVN_U8_6       {2, 16, 8, 8},    {1, 1, 1, 8},     data_types::u8, format::b_fs_yx_fsv16, true, true, data_types::f32, format::bfyx
#define CASE_MVN_U8_7       {1, 16, 16, 1},   {1, 16, 1, 16},   data_types::u8, format::b_fs_yx_fsv16, true, true, data_types::f32, format::bfyx
#define CASE_MVN_3D_U8_1    {1, 16, 8, 8, 8}, {1, 16, 8, 8, 8}, data_types::u8, format::bfzyx, false, true, data_types::f32, format::bfzyx
#define CASE_MVN_3D_U8_2    {2, 16, 8, 8, 8}, {2, 16, 8, 8, 8}, data_types::u8, format::bfzyx, true, true, data_types::f32, format::bfzyx
#define CASE_MVN_3D_U8_3    {2, 16, 8, 8, 8}, {2, 1, 1, 1, 1},  data_types::u8, format::bfzyx, true, true, data_types::f32, format::bfzyx
#define CASE_MVN_3D_U8_4    {2, 16, 8, 8, 8}, {1, 1, 1, 1, 1},  data_types::u8, format::bfzyx, false, true, data_types::f32, format::bfzyx
#define CASE_MVN_3D_U8_5    {2, 16, 1, 8, 8}, {1, 1, 8, 1, 1},  data_types::u8, format::bfzyx, false, true, data_types::f32, format::bfzyx

class MVNFusingTest : public ::BaseFusingTest<mvn_test_params> {
public:
    void execute(mvn_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));

        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);

        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(mvn_test_params& p) {
        return layout{ p.input_type, p.input_format, p.input_size };
    }

    layout get_per_channel_layout(mvn_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{1, p.input_size.feature[0], 1, 1} };
    }
};

class mvn_activation : public MVNFusingTest {};
TEST_P(mvn_activation, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        mvn("mvn", "input", p.normalize_variance, 1e-10f, false, false),
        activation("act", "mvn", activation_func::hyperbolic_tan),
        reorder("reorder_bfyx", "act", format::bfyx, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, mvn_activation,
    ::testing::ValuesIn(std::vector<mvn_test_params>{
                        mvn_test_params{ CASE_MVN_F32_1, 2, 3 },
                        mvn_test_params{ CASE_MVN_F32_2, 2, 3 },
                        mvn_test_params{ CASE_MVN_3D_F32_1, 2, 3 },
                        mvn_test_params{ CASE_MVN_3D_F32_2, 2, 3 },
                        mvn_test_params{ CASE_MVN_F16_1, 2, 3 },
                        mvn_test_params{ CASE_MVN_F16_2, 2, 3 },
                        mvn_test_params{ CASE_MVN_3D_F16_1, 2, 3 },
                        mvn_test_params{ CASE_MVN_3D_F16_2, 2, 3 },
                        mvn_test_params{ CASE_MVN_I8_1, 2, 3 },
                        mvn_test_params{ CASE_MVN_I8_2, 2, 3 },
                        mvn_test_params{ CASE_MVN_I8_3, 2, 3 },
                        mvn_test_params{ CASE_MVN_I8_4, 2, 3 },
                        mvn_test_params{ CASE_MVN_3D_I8_1, 2, 3 },
                        mvn_test_params{ CASE_MVN_3D_I8_2, 2, 3 },
                        mvn_test_params{ CASE_MVN_U8_1, 2, 3 },
                        mvn_test_params{ CASE_MVN_U8_2, 2, 3 },
                        mvn_test_params{ CASE_MVN_U8_3, 2, 3 },
                        mvn_test_params{ CASE_MVN_U8_4, 2, 3 },
                        mvn_test_params{ CASE_MVN_3D_U8_1, 2, 3 },
                        mvn_test_params{ CASE_MVN_3D_U8_2, 2, 3 },
}));

class mvn_scale_quantize_i8 : public MVNFusingTest {};
TEST_P(mvn_scale_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        mvn("mvn", "input", p.normalize_variance, 1e-10f, false, false),
        data("scale_data", get_mem(get_per_channel_layout(p))),
        scale("scale", "mvn", "scale_data"),
        data("in_low", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_high", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_low", get_mem(get_single_element_layout(p), -127, 127)),
        data("out_high", get_mem(get_single_element_layout(p), -127, 127)),
        quantize("quant", "scale", "in_low", "in_high", "out_low", "out_high", 255, data_types::i8),
        reorder("reorder_bfyx", "quant", format::bfyx, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, mvn_scale_quantize_i8,
    ::testing::ValuesIn(std::vector<mvn_test_params>{
        // Full fusing for fp input not supported yet, it may lead to output padding and non-optimal kernel
        // mvn_test_params{ CASE_MVN_F32_1, 2, 4 },
        // mvn_test_params{ CASE_MVN_F32_2, 2, 4 },
        // mvn_test_params{ CASE_MVN_3D_F32_1, 2, 4 },
        // mvn_test_params{ CASE_MVN_3D_F32_2, 2, 4 },
        // mvn_test_params{ CASE_MVN_F16_1, 2, 4 },
        // mvn_test_params{ CASE_MVN_F16_2, 2, 4 },
        // mvn_test_params{ CASE_MVN_3D_F16_1, 2, 4 },
        // mvn_test_params{ CASE_MVN_3D_F16_2, 2, 4 },
        mvn_test_params{ CASE_MVN_I8_1, 2, 4 },
        mvn_test_params{ CASE_MVN_I8_2, 2, 4 },
        mvn_test_params{ CASE_MVN_I8_3, 2, 4 },
        mvn_test_params{ CASE_MVN_I8_4, 2, 4 },
        mvn_test_params{ CASE_MVN_3D_I8_1, 2, 4 },
        mvn_test_params{ CASE_MVN_3D_I8_2, 2, 4 },
        mvn_test_params{ CASE_MVN_U8_1, 2, 4 },
        mvn_test_params{ CASE_MVN_U8_2, 2, 4 },
        mvn_test_params{ CASE_MVN_U8_3, 2, 4 },
        mvn_test_params{ CASE_MVN_U8_4, 2, 4 },
        mvn_test_params{ CASE_MVN_3D_U8_1, 2, 4 },
        mvn_test_params{ CASE_MVN_3D_U8_2, 2, 4 },
}));

class mvn_scale_activation_eltwise_fp32_quantize_i8 : public MVNFusingTest {};
TEST_P(mvn_scale_activation_eltwise_fp32_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        mvn("mvn", "input", p.normalize_variance, 1e-10f, false, false),
        data("scale_data", get_mem(get_per_channel_layout(p))),
        scale("scale", "mvn", "scale_data"),
        activation("act", "scale", activation_func::hyperbolic_tan),
        data("eltw_data", get_mem(layout{ p.input_type, p.default_format, p.elwise_size })),
        eltwise("eltw", {"act", "eltw_data"}, eltwise_mode::sum, data_types::f32),
        data("in_low", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_high", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_low", get_mem(get_single_element_layout(p), -128)),
        data("out_high", get_mem(get_single_element_layout(p), 127)),
        quantize("quant", "eltw", "in_low", "in_high", "out_low", "out_high", 256, data_types::i8),
        reorder("reorder_bfyx", "quant", format::bfyx, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, mvn_scale_activation_eltwise_fp32_quantize_i8,
    ::testing::ValuesIn(std::vector<mvn_test_params>{
        // Full using for fp input not supported yet, it may lead to output padding and non-optimal kernel
        // mvn_test_params{ CASE_MVN_F32_1, 2, 7 },
        // mvn_test_params{ CASE_MVN_F32_2, 2, 7 },
        // mvn_test_params{ CASE_MVN_3D_F32_1, 2, 7 },
        // mvn_test_params{ CASE_MVN_3D_F32_2, 2, 7 },
        // mvn_test_params{ CASE_MVN_F16_1, 2, 7 },
        // mvn_test_params{ CASE_MVN_F16_2, 2, 7 },
        // mvn_test_params{ CASE_MVN_3D_F16_1, 2, 7 },
        // mvn_test_params{ CASE_MVN_3D_F16_2, 2, 7 },
        mvn_test_params{ CASE_MVN_I8_1, 2, 6 },
        mvn_test_params{ CASE_MVN_I8_2, 2, 6 },
        mvn_test_params{ CASE_MVN_I8_3, 2, 6 },
        mvn_test_params{ CASE_MVN_I8_4, 2, 6 },
        mvn_test_params{ CASE_MVN_I8_5, 2, 6 },
        mvn_test_params{ CASE_MVN_I8_6, 2, 6 },
        mvn_test_params{ CASE_MVN_I8_7, 3, 6 },
        mvn_test_params{ CASE_MVN_3D_I8_1, 2, 6 },
        mvn_test_params{ CASE_MVN_3D_I8_2, 2, 6 },
        mvn_test_params{ CASE_MVN_3D_I8_3, 2, 6 },
        mvn_test_params{ CASE_MVN_3D_I8_4, 2, 6 },
        mvn_test_params{ CASE_MVN_3D_I8_5, 3, 6 },
        mvn_test_params{ CASE_MVN_U8_1, 2, 6 },
        mvn_test_params{ CASE_MVN_U8_2, 2, 6 },
        mvn_test_params{ CASE_MVN_U8_3, 2, 6 },
        mvn_test_params{ CASE_MVN_U8_4, 2, 6 },
        mvn_test_params{ CASE_MVN_U8_5, 2, 6 },
        mvn_test_params{ CASE_MVN_U8_6, 2, 6 },
        mvn_test_params{ CASE_MVN_U8_7, 3, 6 },
        mvn_test_params{ CASE_MVN_3D_U8_1, 2, 6 },
        mvn_test_params{ CASE_MVN_3D_U8_2, 2, 6 },
        mvn_test_params{ CASE_MVN_3D_U8_3, 2, 6 },
        mvn_test_params{ CASE_MVN_3D_U8_4, 2, 6 },
        mvn_test_params{ CASE_MVN_3D_U8_5, 3, 6 },
}));

class mvn_eltwise : public MVNFusingTest {};
TEST_P(mvn_eltwise, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", layout{ p.input_type, p.input_format, p.input_size }),
                 mvn("mvn", "input", p.normalize_variance, 1e-10f, false, false),
                 data("eltw_data", get_mem(layout{ p.input_type, p.default_format, p.elwise_size })),
                 eltwise("eltw", {"mvn", "eltw_data"}, eltwise_mode::sum, data_types::f32),
                 reorder("reorder_bfyx", "eltw", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, mvn_eltwise,
    ::testing::ValuesIn(std::vector<mvn_test_params>{
        mvn_test_params{ CASE_MVN_I8_5, 2, 3 },
        mvn_test_params{ CASE_MVN_I8_6, 2, 3 },
        mvn_test_params{ CASE_MVN_I8_7, 3, 3 },
        mvn_test_params{ CASE_MVN_3D_I8_3, 2, 3 },
        mvn_test_params{ CASE_MVN_3D_I8_4, 2, 3 },
        mvn_test_params{ CASE_MVN_3D_I8_5, 3, 3 },
        mvn_test_params{ CASE_MVN_U8_1, 2, 3 },
        mvn_test_params{ CASE_MVN_U8_2, 2, 3 },
        mvn_test_params{ CASE_MVN_U8_3, 2, 3 },
        mvn_test_params{ CASE_MVN_U8_4, 2, 3 },
        mvn_test_params{ CASE_MVN_U8_5, 2, 3 },
        mvn_test_params{ CASE_MVN_U8_6, 2, 3 },
        mvn_test_params{ CASE_MVN_U8_7, 3, 3 },
        mvn_test_params{ CASE_MVN_3D_U8_1, 2, 3 },
        mvn_test_params{ CASE_MVN_3D_U8_2, 2, 3 },
        mvn_test_params{ CASE_MVN_3D_U8_3, 2, 3 },
        mvn_test_params{ CASE_MVN_3D_U8_4, 2, 3 },
        mvn_test_params{ CASE_MVN_3D_U8_5, 3, 3 },
}));

class mvn_eltwise_f16 : public MVNFusingTest {};
TEST_P(mvn_eltwise_f16, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", layout{ p.input_type, p.input_format, p.input_size }),
                 mvn("mvn", "input", p.normalize_variance, 1e-10f, false, false),
                 data("eltw_data", get_mem(layout{ p.input_type, p.default_format, p.elwise_size })),
                 eltwise("eltw", {"mvn", "eltw_data"}, eltwise_mode::sum, data_types::f16),
                 reorder("reorder_bfyx", "eltw", p.default_format, data_types::f32)
    );

    tolerance = 0.1f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, mvn_eltwise_f16,
    ::testing::ValuesIn(std::vector<mvn_test_params>{
        mvn_test_params{ CASE_MVN_I8_6, 2, 3 },
        mvn_test_params{ CASE_MVN_U8_2, 2, 3 },
}));

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- LRN cases -------------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
struct lrn_test_params {
    tensor in_shape;
    data_types data_type;
    format input_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
    lrn_norm_region lrn_type;
    std::string kernel_name;
};

#define CASE_LRN_FP32_1 {2, 16, 4, 4}, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_LRN_FP32_2 {8, 16, 4, 4}, data_types::f32, format::yxfb, data_types::f32, format::yxfb
#define CASE_LRN_FP32_3 {2, 16, 4, 4}, data_types::f32, format::byxf, data_types::f32, format::byxf
#define CASE_LRN_FP32_4 {2, 16, 4, 4}, data_types::f32, format::b_fs_yx_fsv4, data_types::f32, format::bfyx
#define CASE_LRN_FP32_5 {2, 16, 4, 4}, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx

#define CASE_LRN_FP32_TO_FP16_1 {2, 16, 5, 5}, data_types::f32, format::bfyx, data_types::f16, format::bfyx
#define CASE_LRN_FP32_TO_FP16_2 {2, 16, 5, 5}, data_types::f32, format::byxf, data_types::f16, format::byxf
#define CASE_LRN_FP32_TO_FP16_3 {8, 16, 4, 4}, data_types::f32, format::yxfb, data_types::f16, format::byxf
#define CASE_LRN_FP32_TO_FP16_4 {2, 16, 4, 4}, data_types::f32, format::b_fs_yx_fsv4, data_types::f16, format::bfyx
#define CASE_LRN_FP32_TO_FP16_5 {2, 16, 4, 4}, data_types::f32, format::b_fs_yx_fsv16, data_types::f16, format::bfyx

#define CASE_LRN_FP16_1 {2, 16, 4, 4}, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_LRN_FP16_2 {8, 16, 4, 4}, data_types::f16, format::yxfb, data_types::f16, format::yxfb
#define CASE_LRN_FP16_3 {2, 16, 4, 4}, data_types::f16, format::byxf, data_types::f16, format::byxf
#define CASE_LRN_FP16_4 {2, 16, 4, 4}, data_types::f16, format::b_fs_yx_fsv4, data_types::f16, format::bfyx
#define CASE_LRN_FP16_5 {2, 16, 4, 4}, data_types::f16, format::b_fs_yx_fsv16, data_types::f16, format::bfyx

class LrnFusingTest : public ::BaseFusingTest<lrn_test_params> {
public:
    void execute(lrn_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));

        build_options options;
        implementation_desc lrn_impl = {p.input_format, p.kernel_name};
        options.set_option(build_option::optimize_data(true));
        options.set_option(build_option::force_implementations({{"lrn_norm", lrn_impl}}));
        network network_fused(this->engine, this->topology_fused, options);
        network network_not_fused(this->engine, this->topology_non_fused, this->bo_not_fused);

        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        ASSERT_FALSE(network_fused.get_primitives_info().empty());
        ASSERT_FALSE(network_not_fused.get_primitives_info().empty());

        auto find_lrn = [&](primitive_info& p) -> bool {
            if (p.original_id == "lrn_norm" || p.original_id == "reorder")
                return true;
            return false;
        };

        auto pi_fused = network_fused.get_primitives_info();
        auto pi_not_fused = network_not_fused.get_primitives_info();
        auto info_fused = std::find_if(pi_fused.begin(), pi_fused.end(), find_lrn);
        auto info_not_fused = std::find_if(pi_not_fused.begin(), pi_not_fused.end(), find_lrn);

        ASSERT_TRUE(info_fused != pi_fused.end());
        ASSERT_TRUE(info_not_fused != pi_not_fused.end());

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(lrn_test_params& p) { return layout{p.data_type, p.input_format, p.in_shape}; }

    layout get_per_channel_layout(lrn_test_params& p) {
        return layout{p.default_type, p.default_format, tensor{1, p.in_shape.feature[0], 1, 1}};
    }
};

class lrn_fp32_quantize_u8_scale_activation : public LrnFusingTest {};
TEST_P(lrn_fp32_quantize_u8_scale_activation, basic) {
    auto p = GetParam();

    uint32_t size = 5;
    float k = 1.0f;
    float alpha = (float)9.9e-05;
    float beta = 0.75;

    create_topologies(input_layout("input", get_input_layout(p)),
                      data("in_lo", get_mem(get_single_element_layout(p), min_random, 0)),
                      data("in_hi", get_mem(get_single_element_layout(p), 1, max_random)),
                      data("out_lo", get_mem(get_single_element_layout(p), 0)),
                      data("out_hi", get_mem(get_single_element_layout(p), 255)),
                      data("scale_data", get_mem(get_single_element_layout(p), 1.0f / 255)),
                      lrn("lrn_norm", "input", size, k, alpha, beta, p.lrn_type),
                      quantize("quantize", "lrn_norm", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::u8),
                      scale("scale", "quantize", "scale_data"),
                      activation("activation", "scale", activation_func::exp),
                      reorder("reorder", "activation", p.default_format, data_types::f32));

    tolerance = 1.0f;
    execute(p);
}

TEST_P(lrn_fp32_quantize_u8_scale_activation, per_channel) {
    auto p = GetParam();

    uint32_t size = 5;
    float k = 1.0f;
    float alpha = (float)9.9e-05;
    float beta = 0.75;

   create_topologies(input_layout("input", get_input_layout(p)),
                     data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                     data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                     data("out_lo", get_mem(get_single_element_layout(p), 0)),
                     data("out_hi", get_mem(get_single_element_layout(p), 255)),
                     data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / 255)),
                     lrn("lrn_norm", "input", size, k, alpha, beta, p.lrn_type),
                     quantize("quantize", "lrn_norm", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::u8),
                     scale("scale", "quantize", "scale_data"),
                     activation("activation", "scale", activation_func::exp),
                     reorder("reorder", "activation", p.default_format, data_types::f32));

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu,
                        lrn_fp32_quantize_u8_scale_activation,
                        ::testing::ValuesIn(std::vector<lrn_test_params>{
                            // InputDataType = FP32   OutputDataType = FP32
                            lrn_test_params{CASE_LRN_FP32_1, 2, 5, lrn_norm_region_across_channel, "lrn_ref"},
                            lrn_test_params{CASE_LRN_FP32_1, 2, 5, lrn_norm_region_within_channel, "lrn_gpu_within_channel_opt"},
                            lrn_test_params{CASE_LRN_FP32_1, 2, 5, lrn_norm_region_within_channel, "lrn_gpu_within_channel"},
                            lrn_test_params{CASE_LRN_FP32_1, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_ref"},
                            lrn_test_params{CASE_LRN_FP32_1, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features"},
                            lrn_test_params{CASE_LRN_FP32_2, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_yxfb_b8_opt"},
                            lrn_test_params{CASE_LRN_FP32_3, 2, 5, lrn_norm_region_within_channel, "lrn_within_channel_byxf_opt"},
                            lrn_test_params{CASE_LRN_FP32_4, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features"},
                            lrn_test_params{CASE_LRN_FP32_5, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features_fsv16"},

                            // InputDataType = FP32   OutputDataType = FP16
                            lrn_test_params{CASE_LRN_FP32_TO_FP16_1, 2, 5, lrn_norm_region_across_channel, "lrn_ref"},
                            lrn_test_params{CASE_LRN_FP32_TO_FP16_1, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features"},
                            lrn_test_params{CASE_LRN_FP32_TO_FP16_1, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_ref"},
                            lrn_test_params{CASE_LRN_FP32_TO_FP16_1, 2, 5, lrn_norm_region_within_channel, "lrn_gpu_within_channel_opt"},
                            lrn_test_params{CASE_LRN_FP32_TO_FP16_1, 2, 5, lrn_norm_region_within_channel, "lrn_gpu_within_channel"},
                            lrn_test_params{CASE_LRN_FP32_TO_FP16_3, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_yxfb_b8_opt"},
                            lrn_test_params{CASE_LRN_FP32_TO_FP16_4, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features"},
                            lrn_test_params{CASE_LRN_FP32_TO_FP16_5, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features_fsv16"},

                        }));

class lrn_fp32_quantize_i8_scale_activation : public LrnFusingTest {};
TEST_P(lrn_fp32_quantize_i8_scale_activation, basic) {
    auto p = GetParam();

    uint32_t size = 5;
    float k = 1.0f;
    float alpha = (float)9.9e-05;
    float beta = 0.75;

   create_topologies(input_layout("input", get_input_layout(p)),
                     data("in_lo", get_mem(get_single_element_layout(p), min_random, 0)),
                     data("in_hi", get_mem(get_single_element_layout(p), 1, max_random)),
                     data("out_lo", get_mem(get_single_element_layout(p), -127)),
                     data("out_hi", get_mem(get_single_element_layout(p),  127)),
                     data("scale_data", get_mem(get_single_element_layout(p), 1.0f / 255)),
                     lrn("lrn_norm", "input", size, k, alpha, beta, p.lrn_type),
                     scale("scale", "lrn_norm", "scale_data"),
                     activation("activation", "scale", activation_func::exp),
                     quantize("quantize", "activation", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::i8),
                     reorder("reorder", "quantize", p.default_format, data_types::f32));

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu,
                        lrn_fp32_quantize_i8_scale_activation,
                        ::testing::ValuesIn(std::vector<lrn_test_params>{
                            // InputDataType = FP32   OutputDataType = INT8
                            lrn_test_params{CASE_LRN_FP32_1, 2, 5, lrn_norm_region_within_channel, "lrn_gpu_within_channel_opt"},
                            lrn_test_params{CASE_LRN_FP32_1, 2, 5, lrn_norm_region_within_channel, "lrn_gpu_within_channel"},
                            lrn_test_params{CASE_LRN_FP32_1, 2, 5, lrn_norm_region_across_channel, "lrn_ref"},
                            lrn_test_params{CASE_LRN_FP32_1, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features"},
                            lrn_test_params{CASE_LRN_FP32_1, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_ref"},
                            lrn_test_params{CASE_LRN_FP32_2, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_yxfb_b8_opt"},
                            lrn_test_params{CASE_LRN_FP32_3, 2, 5, lrn_norm_region_within_channel, "lrn_within_channel_byxf_opt"},
                            lrn_test_params{CASE_LRN_FP32_4, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features"},
                            lrn_test_params{CASE_LRN_FP32_5, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features_fsv16"},

                            // InputDataType = FP16   OutputDataType = INT8/UINT8 can't be tested for now, because quantize
                            // primitive doesn't support input type FP16 while fusing (prepare_quantization.cpp :114 -> prepare_primitive_fusing.cpp :474)
                        }));

class lrn_fp32_scale_activation_quantize_u8 : public LrnFusingTest {};
TEST_P(lrn_fp32_scale_activation_quantize_u8, basic) {
    auto p = GetParam();

    uint32_t size = 5;
    float k = 1.0f;
    float alpha = (float)9.9e-05;
    float beta = 0.75;

   create_topologies(input_layout("input", get_input_layout(p)),
                     data("in_lo", get_mem(get_single_element_layout(p), min_random, 0)),
                     data("in_hi", get_mem(get_single_element_layout(p), 1, max_random)),
                     data("out_lo", get_mem(get_single_element_layout(p), 0)),
                     data("out_hi", get_mem(get_single_element_layout(p), 255)),
                     data("scale_data", get_mem(get_single_element_layout(p), 1.0f / 255)),
                     lrn("lrn_norm", "input", size, k, alpha, beta, p.lrn_type),
                     scale("scale", "lrn_norm", "scale_data"),
                     activation("activation", "scale", activation_func::exp),
                     quantize("quantize", "activation", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::u8),
                     reorder("reorder", "quantize", p.default_format, data_types::f32));

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu,
                        lrn_fp32_scale_activation_quantize_u8,
                        ::testing::ValuesIn(std::vector<lrn_test_params>{
                            // InputDataType = FP32   OutputDataType = UINT8
                            lrn_test_params{CASE_LRN_FP32_1, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_ref"},
                            lrn_test_params{CASE_LRN_FP32_1, 2, 5, lrn_norm_region_within_channel, "lrn_gpu_within_channel_opt"},
                            lrn_test_params{CASE_LRN_FP32_1, 2, 5, lrn_norm_region_within_channel, "lrn_gpu_within_channel"},
                            lrn_test_params{CASE_LRN_FP32_1, 2, 5, lrn_norm_region_across_channel, "lrn_ref"},
                            lrn_test_params{CASE_LRN_FP32_1, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features"},
                            lrn_test_params{CASE_LRN_FP32_2, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_yxfb_b8_opt"},
                            lrn_test_params{CASE_LRN_FP32_3, 2, 5, lrn_norm_region_within_channel, "lrn_within_channel_byxf_opt"},
                            lrn_test_params{CASE_LRN_FP32_4, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features"},
                            lrn_test_params{CASE_LRN_FP32_5, 2, 5, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features_fsv16"},
                        }));

class lrn_fp16_scale_activation : public LrnFusingTest {};
TEST_P(lrn_fp16_scale_activation, basic) {
    auto p = GetParam();

    uint32_t size = 5;
    float k = 1.0f;
    float alpha = (float)9.9e-05;
    float beta = 0.75;

    create_topologies(input_layout("input", get_input_layout(p)),
                      data("scale_data", get_mem(get_single_element_layout(p), 1.0f / 255)),
                      lrn("lrn_norm", "input", size, k, alpha, beta, p.lrn_type),
                      scale("scale", "lrn_norm", "scale_data"),
                      activation("activation", "scale", activation_func::exp),
                      reorder("reorder", "activation", p.default_format, data_types::f32));

    tolerance = 1e-05f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu,
                        lrn_fp16_scale_activation,
                        ::testing::ValuesIn(std::vector<lrn_test_params>{
                            // InputDataType = FP16   OutputDataType = FP16
                            lrn_test_params{CASE_LRN_FP16_1, 2, 4, lrn_norm_region_within_channel, "lrn_gpu_within_channel_opt"},
                            lrn_test_params{CASE_LRN_FP16_1, 2, 4, lrn_norm_region_within_channel, "lrn_gpu_within_channel"},
                            lrn_test_params{CASE_LRN_FP16_1, 2, 4, lrn_norm_region_across_channel, "lrn_ref"},
                            lrn_test_params{CASE_LRN_FP16_1, 2, 4, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features"},
                            lrn_test_params{CASE_LRN_FP16_1, 2, 4, lrn_norm_region_across_channel, "lrn_gpu_across_channel_ref"},
                            lrn_test_params{CASE_LRN_FP16_3, 2, 4, lrn_norm_region_within_channel, "lrn_within_channel_byxf_opt"},
                            lrn_test_params{CASE_LRN_FP16_4, 2, 4, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features"},
                            lrn_test_params{CASE_LRN_FP16_5, 2, 4, lrn_norm_region_across_channel, "lrn_gpu_across_channel_multiple_features_fsv16"},
                        }));

/* ----------------------------------------------------------------------------------------------------- */
/* -------------------------------- Activation cases --------------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
struct activation_test_params {
    tensor input_size;
    data_types input_type;
    format input_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
    std::string kernel_name;
};

#define CASE_ACTIVATION_F32_0 {7, 32, 3, 3}, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_ACTIVATION_F32_1 {1, 16, 8, 8}, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_ACTIVATION_F32_2 {7, 3, 7, 7}, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_ACTIVATION_F32_3 {1, 14, 8, 8}, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_ACTIVATION_F32_4 {1, 17, 31, 29}, data_types::f32, format::yxfb, data_types::f32, format::bfyx
#define CASE_ACTIVATION_F32_5 {1, 17, 31, 29}, data_types::f32, format::b_fs_yx_fsv4, data_types::f32, format::bfyx
#define CASE_ACTIVATION_F32_6 {1, 17, 31, 29}, data_types::f32, format::b_fs_yx_fsv32, data_types::f32, format::bfyx
#define CASE_ACTIVATION_F32_7 {1, 17, 31, 29}, data_types::f32, format::fyxb, data_types::f32, format::bfyx
#define CASE_ACTIVATION_3D_F32_0 {3, 16, 13, 13, 13}, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_ACTIVATION_3D_F32_1 {2, 16, 8, 8, 8}, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_ACTIVATION_3D_F32_2 {1, 16, 7, 7, 7}, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::bfzyx
#define CASE_ACTIVATION_3D_F32_3 {1, 17, 7, 7, 7}, data_types::f32, format::b_fs_zyx_fsv32, data_types::f32, format::bfzyx
#define CASE_ACTIVATION_3D_F32_4 {1, 17, 7, 7, 7}, data_types::f32, format::bs_fs_yx_bsv16_fsv16, data_types::f32, format::bfzyx
#define CASE_ACTIVATION_3D_F32_5 {1, 17, 7, 7, 7}, data_types::f32, format::fs_b_yx_fsv32, data_types::f32, format::bfzyx

#define CASE_ACTIVATION_F16_0 {7, 32, 5, 5}, data_types::f16, format::bfyx, data_types::f32, format::bfyx
#define CASE_ACTIVATION_F16_1 {1, 16, 8, 8}, data_types::f16, format::bfyx, data_types::f32, format::bfyx
#define CASE_ACTIVATION_F16_2 {7, 16, 7, 7}, data_types::f16, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_ACTIVATION_F16_3 {1, 14, 8, 8}, data_types::f16, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_ACTIVATION_F16_4 {1, 17, 31, 29}, data_types::f16, format::yxfb, data_types::f32, format::bfyx
#define CASE_ACTIVATION_F16_5 {1, 17, 31, 29}, data_types::f16, format::b_fs_yx_fsv4, data_types::f32, format::bfyx
#define CASE_ACTIVATION_F16_6 {1, 17, 31, 29}, data_types::f16, format::b_fs_yx_fsv32, data_types::f32, format::bfyx
#define CASE_ACTIVATION_F16_7 {1, 17, 31, 29}, data_types::f16, format::fyxb, data_types::f32, format::bfyx
#define CASE_ACTIVATION_3D_F16_0 {3, 16, 13, 13, 13}, data_types::f16, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_ACTIVATION_3D_F16_1 {2, 16, 8, 8, 8}, data_types::f16, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_ACTIVATION_3D_F16_2 {1, 16, 7, 7, 7}, data_types::f16, format::b_fs_zyx_fsv16, data_types::f32, format::bfzyx
#define CASE_ACTIVATION_3D_F16_3 {1, 17, 7, 7, 7}, data_types::f16, format::b_fs_zyx_fsv32, data_types::f32, format::bfzyx
#define CASE_ACTIVATION_3D_F16_4 {1, 17, 7, 7, 7}, data_types::f16, format::bs_fs_yx_bsv16_fsv16, data_types::f32, format::bfzyx
#define CASE_ACTIVATION_3D_F16_5 {1, 17, 7, 7, 7}, data_types::f16, format::fs_b_yx_fsv32, data_types::f32, format::bfzyx

#define CASE_ACTIVATION_U8_1 {1, 16, 8, 8}, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_ACTIVATION_U8_2 {1, 12, 8, 8}, data_types::u8, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_ACTIVATION_I8_1 {1, 16, 8, 8}, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_ACTIVATION_I8_2 {1, 14, 8, 8}, data_types::i8, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_ACTIVATION_3D_I8_1 {1, 17, 8, 8, 8}, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx

class ActivationFusingTest : public ::BaseFusingTest<activation_test_params> {
public:
    void execute(activation_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));

        build_options options;
        implementation_desc activation_impl = {p.input_format, p.kernel_name};
        options.set_option(build_option::optimize_data(true));
        options.set_option(build_option::force_implementations({{"act", activation_impl}}));
        network network_fused(this->engine, this->topology_fused, options);
        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);

        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(activation_test_params& p) { return layout{p.input_type, p.input_format, p.input_size}; }

    layout get_per_channel_layout(activation_test_params& p) {
        return layout{p.default_type, p.default_format, tensor{1, p.input_size.feature[0], 1, 1}};
    }

    format get_input_format(activation_test_params &p) { return p.input_format; }
};

class activation_quantize_i8 : public ActivationFusingTest {};
TEST_P(activation_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      activation("act", "input", activation_func::relu),
                      data("in_low", get_mem(get_single_element_layout(p), min_random, 0)),
                      data("in_high", get_mem(get_single_element_layout(p), 1, max_random)),
                      data("out_low", get_mem(get_single_element_layout(p), -127, 0)),
                      data("out_high", get_mem(get_single_element_layout(p), 0, 127)),
                      quantize("quant", "act", "in_low", "in_high", "out_low", "out_high", 255, data_types::i8),
                      reorder("reorder_bfyx", "quant", p.default_format, data_types::f32));

    tolerance = 1.0f;
    execute(p);
}

TEST_P(activation_quantize_i8, per_channel) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      activation("act", "input", activation_func::relu),
                      data("in_low", get_mem(get_per_channel_layout(p), min_random, 0)),
                      data("in_high", get_mem(get_per_channel_layout(p), 1, max_random)),
                      data("out_low", get_mem(get_single_element_layout(p), -127, 0)),
                      data("out_high", get_mem(get_single_element_layout(p), 0, 127)),
                      quantize("quant", "act", "in_low", "in_high", "out_low", "out_high", 255, data_types::i8),
                      reorder("reorder_bfyx", "quant", p.default_format, data_types::f32));

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(
    fusings_gpu,
    activation_quantize_i8,
    ::testing::ValuesIn(std::vector<activation_test_params>{
        // InputDataType = FP32
        activation_test_params{CASE_ACTIVATION_F32_0, 2, 3, "activation_opt"},
        activation_test_params{CASE_ACTIVATION_F32_1, 2, 3, "activation_opt"},
        activation_test_params{CASE_ACTIVATION_3D_F32_0, 2, 3, "activation_opt"},
        activation_test_params{CASE_ACTIVATION_3D_F32_1, 2, 3, "activation_opt"},

        activation_test_params{CASE_ACTIVATION_F32_0, 2, 3, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_F32_1, 2, 3, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_F32_2, 2, 3, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_F32_3, 2, 3, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_F32_4, 2, 3, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_3D_F32_0, 2, 3, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_3D_F32_1, 2, 3, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_3D_F32_2, 2, 3, "activation_ref"},
    }));

INSTANTIATE_TEST_SUITE_P(
    DISABLED_fusings_gpu,
    activation_quantize_i8,
    ::testing::ValuesIn(std::vector<activation_test_params>{
        activation_test_params{CASE_ACTIVATION_F32_5, 2, 3, "activation_ref"},     // FIXME - accuracy bug
        activation_test_params{CASE_ACTIVATION_F32_6, 2, 3, "activation_ref"},     // FIXME - accuracy bug
        activation_test_params{CASE_ACTIVATION_F32_7, 2, 3, "activation_ref"},     // FIXME - accuracy bug
        activation_test_params{CASE_ACTIVATION_3D_F32_3, 2, 3, "activation_ref"},  // FIXME - accuracy bug
        activation_test_params{CASE_ACTIVATION_3D_F32_5, 2, 3, "activation_ref"},  // FIXME - accuracy bug
    }));

class activation_scale_activation_quantize_u8 : public ActivationFusingTest {};
TEST_P(activation_scale_activation_quantize_u8, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      activation("act", "input", activation_func::relu),
                      data("scale_data", get_mem(get_single_element_layout(p), 1.0f / 255)),
                      data("in_low", get_mem(get_single_element_layout(p), 0)),
                      data("in_high", get_mem(get_single_element_layout(p), 1, max_random)),
                      data("out_low", get_mem(get_single_element_layout(p), -127)),
                      data("out_high", get_mem(get_single_element_layout(p), 127)),
                      scale("scale", "act", "scale_data"),
                      activation("act2", "scale", activation_func::softsign),
                      quantize("quant", "act2", "in_low", "in_high", "out_low", "out_high", 256, data_types::u8),
                      reorder("reorder_bfyx", "quant", p.default_format, data_types::f32));

    tolerance = 1.f;
    execute(p);
}

TEST_P(activation_scale_activation_quantize_u8, per_channel) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      activation("act", "input", activation_func::relu),
                      data("scale_data", get_mem(get_single_element_layout(p), 1.0f / 255)),
                      data("in_low", get_mem(get_per_channel_layout(p), 0)),
                      data("in_high", get_mem(get_per_channel_layout(p), 1, max_random)),
                      data("out_low", get_mem(get_single_element_layout(p), -127)),
                      data("out_high", get_mem(get_single_element_layout(p), 127)),
                      scale("scale", "act", "scale_data"),
                      activation("act2", "scale", activation_func::softsign),
                      quantize("quant", "act2", "in_low", "in_high", "out_low", "out_high", 256, data_types::u8),
                      reorder("reorder_bfyx", "quant", p.default_format, data_types::f32));

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(
    fusings_gpu,
    activation_scale_activation_quantize_u8,
    ::testing::ValuesIn(std::vector<activation_test_params>{
        // InputDataType = FP32
        activation_test_params{CASE_ACTIVATION_F32_0, 2, 5, "activation_opt"},
        activation_test_params{CASE_ACTIVATION_F32_1, 2, 5, "activation_opt"},
        activation_test_params{CASE_ACTIVATION_3D_F32_0, 2, 5, "activation_opt"},
        activation_test_params{CASE_ACTIVATION_3D_F32_1, 2, 5, "activation_opt"},

        activation_test_params{CASE_ACTIVATION_F32_0, 2, 5, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_F32_1, 2, 5, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_F32_2, 2, 5, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_F32_3, 2, 5, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_F32_4, 2, 5, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_F32_5, 2, 5, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_F32_6, 2, 5, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_F32_7, 2, 5, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_3D_F32_0, 2, 5, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_3D_F32_1, 2, 5, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_3D_F32_2, 2, 5, "activation_ref"},
    }));

INSTANTIATE_TEST_SUITE_P(
    DISABLED_fusings_gpu,
    activation_scale_activation_quantize_u8,
    ::testing::ValuesIn(std::vector<activation_test_params>{
        activation_test_params{CASE_ACTIVATION_3D_F32_5, 2, 5, "activation_ref"},  // FIXME - accuracy bug
    }));

class activation_scale_activation : public ActivationFusingTest {};
TEST_P(activation_scale_activation, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      activation("act", "input", activation_func::relu),
                      data("scale_data", get_mem(get_single_element_layout(p), 1.0f / 255)),
                      scale("scale", "act", "scale_data"),
                      activation("act2", "scale", activation_func::exp),
                      reorder("reorder_bfyx", "act2", p.default_format, data_types::f32));

    tolerance = 1e-05f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(
    fusings_gpu,
    activation_scale_activation,
    ::testing::ValuesIn(std::vector<activation_test_params>{
        // InputDataType = FP32
        activation_test_params{CASE_ACTIVATION_F32_0, 2, 4, "activation_opt"},
        activation_test_params{CASE_ACTIVATION_F32_1, 2, 4, "activation_opt"},
        activation_test_params{CASE_ACTIVATION_3D_F32_0, 2, 4, "activation_opt"},
        activation_test_params{CASE_ACTIVATION_3D_F32_1, 2, 4, "activation_opt"},

        activation_test_params{CASE_ACTIVATION_F32_0, 2, 4, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_F32_1, 2, 4, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_F32_2, 2, 4, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_F32_3, 2, 4, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_F32_4, 2, 4, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_F32_5, 2, 4, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_F32_6, 2, 4, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_F32_7, 2, 4, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_3D_F32_0, 2, 4, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_3D_F32_1, 2, 4, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_3D_F32_2, 2, 4, "activation_ref"},

        // InputDataType = FP16
        activation_test_params{CASE_ACTIVATION_F16_0, 2, 4, "activation_opt"},
        activation_test_params{CASE_ACTIVATION_F16_1, 2, 4, "activation_opt"},
        activation_test_params{CASE_ACTIVATION_3D_F16_0, 2, 4, "activation_opt"},
        activation_test_params{CASE_ACTIVATION_3D_F16_1, 2, 4, "activation_opt"},

        activation_test_params{CASE_ACTIVATION_F16_0, 2, 4, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_F16_1, 2, 4, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_F16_2, 2, 4, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_F16_3, 2, 4, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_F16_4, 2, 4, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_F16_5, 2, 4, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_F16_6, 2, 4, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_F16_7, 2, 4, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_3D_F16_0, 2, 4, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_3D_F16_1, 2, 4, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_3D_F16_2, 2, 4, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_3D_F16_3, 2, 4, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_3D_F16_4, 2, 4, "activation_ref"},

        // InputDataType = UINT8
        activation_test_params{CASE_ACTIVATION_U8_1, 2, 4, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_U8_2, 2, 4, "activation_ref"},

        // InputDataType = INT8
        activation_test_params{CASE_ACTIVATION_I8_1, 2, 4, "activation_opt"},
        activation_test_params{CASE_ACTIVATION_3D_I8_1, 2, 4, "activation_opt"},

        activation_test_params{CASE_ACTIVATION_I8_1, 2, 4, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_I8_2, 2, 4, "activation_ref"},
        activation_test_params{CASE_ACTIVATION_3D_I8_1, 2, 4, "activation_ref"}
    }));

INSTANTIATE_TEST_SUITE_P(
    DISABLED_fusings_gpu,
    activation_scale_activation,
    ::testing::ValuesIn(std::vector<activation_test_params>{
        activation_test_params{CASE_ACTIVATION_3D_F32_4, 2, 4, "activation_ref"},  // FIXME - accuracy bug
        activation_test_params{CASE_ACTIVATION_3D_F32_5, 2, 4, "activation_ref"},  // FIXME - accuracy bug
    }));

/* ----------------------------------------------------------------------------------------------------- */
/* --------------------------------------- Deconvolution cases ----------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
using deconv_test_params = bc_test_params;

// in_shape; out_shape; kernel; stride; pad; dilation; groups; data_type; input_format; weights_type; weights_format; default_type; default_format;
#define CASE_DECONV_FP32_1 {1, 15, 4, 5}, {1, 30, 6, 7}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::bfyx, data_types::f32, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_FP32_2 {1, 16, 4, 5}, {1, 32, 6, 7}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::is_os_yx_isv16_osv16, data_types::f32, format::bfyx
#define CASE_DECONV_FP32_3 {1, 16, 4, 5}, {1, 32, 4, 5}, {1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::is_os_yx_isv16_osv16, data_types::f32, format::bfyx
#define CASE_DECONV_FP32_4 {1, 32, 4, 5}, {1, 32, 4, 5}, {1, 1, 3, 3}, tensor{1}, tensor{0, 0, -1, -1, 0, 0}, tensor{1}, 32, data_types::f32, format::b_fs_yx_fsv16, data_types::f32,  format::gs_oiyx_gsv16, data_types::f32, format::bfyx
#define CASE_DECONV_FP32_5 {1, 15, 4, 5}, {1, 30, 9, 11}, {1, 1, 3, 3}, tensor{1, 1, 2, 2}, tensor{0}, tensor{1}, 1, data_types::f32, format::bfyx, data_types::f32, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_FP32_6 {1, 16, 4, 5}, {1, 32, 9, 11}, {1, 1, 3, 3}, tensor{1, 1, 2, 2}, tensor{0}, tensor{1}, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::is_os_yx_isv16_osv16, data_types::f32, format::bfyx
#define CASE_DECONV_FP32_7 {1, 16, 4, 5}, {1, 32, 7, 9}, {1, 1, 1, 1}, tensor{1, 1, 2, 2}, tensor{0}, tensor{1}, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::is_os_yx_isv16_osv16, data_types::f32, format::bfyx
#define CASE_DECONV_FP32_8 {1, 32, 4, 5}, {1, 32, 7, 9}, {1, 1, 3, 3}, tensor{1, 1, 2, 2}, tensor{0, 0, -1, -1, 0, 0}, tensor{1}, 32, data_types::f32, format::b_fs_yx_fsv16, data_types::f32,  format::gs_oiyx_gsv16, data_types::f32, format::bfyx

#define CASE_DECONV_FP16_1 {1, 15, 4, 5}, {1, 30, 6, 7}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f16, format::bfyx, data_types::f16, format::oiyx, data_types::f16, format::bfyx
#define CASE_DECONV_FP16_2 {1, 16, 4, 5}, {1, 32, 6, 7}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f16, format::b_fs_yx_fsv16, data_types::f16, format::is_os_yx_isv16_osv16, data_types::f16, format::bfyx
#define CASE_DECONV_FP16_3 {1, 16, 4, 5}, {1, 32, 4, 5}, {1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f16, format::b_fs_yx_fsv16, data_types::f16, format::is_os_yx_isv16_osv16, data_types::f16, format::bfyx
#define CASE_DECONV_FP16_4 {1, 32, 4, 5}, {1, 32, 4, 5}, {1, 1, 3, 3}, tensor{1}, tensor{0, 0, -1, -1, 0, 0}, tensor{1}, 32, data_types::f16, format::b_fs_yx_fsv16, data_types::f16,  format::gs_oiyx_gsv16, data_types::f16, format::bfyx
#define CASE_DECONV_FP16_5 {1, 15, 4, 5}, {1, 30, 9, 11}, {1, 1, 3, 3}, tensor{1, 1, 2, 2}, tensor{0}, tensor{1}, 1, data_types::f16, format::bfyx, data_types::f16, format::oiyx, data_types::f16, format::bfyx
#define CASE_DECONV_FP16_6 {1, 16, 4, 5}, {1, 32, 9, 11}, {1, 1, 3, 3}, tensor{1, 1, 2, 2}, tensor{0}, tensor{1}, 1, data_types::f16, format::b_fs_yx_fsv16, data_types::f16, format::is_os_yx_isv16_osv16, data_types::f16, format::bfyx
#define CASE_DECONV_FP16_7 {1, 16, 4, 5}, {1, 32, 7, 9}, {1, 1, 1, 1}, tensor{1, 1, 2, 2}, tensor{0}, tensor{1}, 1, data_types::f16, format::b_fs_yx_fsv16, data_types::f16, format::is_os_yx_isv16_osv16, data_types::f16, format::bfyx
#define CASE_DECONV_FP16_8 {1, 32, 4, 5}, {1, 32, 7, 9}, {1, 1, 3, 3}, tensor{1, 1, 2, 2}, tensor{0, 0, -1, -1, 0, 0}, tensor{1}, 32, data_types::f16, format::b_fs_yx_fsv16, data_types::f16,  format::gs_oiyx_gsv16, data_types::f16, format::bfyx

#define CASE_DECONV_S8S8_1 {1, 15, 4, 5}, {1, 30, 6, 7}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_S8S8_2 {1, 16, 4, 5}, {1, 32, 6, 7}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::b_fs_yx_fsv16, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_S8S8_3 {1, 16, 4, 5}, {1, 32, 4, 5}, {1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::b_fs_yx_fsv16, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_S8S8_4 {1, 32, 4, 5}, {1, 32, 4, 5}, {1, 1, 3, 3}, tensor{1}, tensor{0, 0, -1, -1}, tensor{1}, 32, data_types::i8, format::b_fs_yx_fsv16, data_types::i8,  format::goiyx, data_types::f32, format::bfyx
#define CASE_DECONV_S8S8_5 {1, 15, 4, 5}, {1, 30, 9, 11}, {1, 1, 3, 3}, tensor{1, 1, 2, 2}, tensor{0}, tensor{1}, 1, data_types::i8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_S8S8_6 {1, 16, 4, 5}, {1, 32, 9, 11}, {1, 1, 3, 3}, tensor{1, 1, 2, 2}, tensor{0}, tensor{1}, 1, data_types::i8, format::b_fs_yx_fsv16, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_S8S8_7 {1, 16, 4, 5}, {1, 32, 7, 9}, {1, 1, 1, 1}, tensor{1, 1, 2, 2}, tensor{0}, tensor{1}, 1, data_types::i8, format::b_fs_yx_fsv16, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_S8S8_8 {1, 32, 4, 5}, {1, 32, 7, 9}, {1, 1, 3, 3}, tensor{1, 1, 2, 2}, tensor{0, 0, -1, -1, 0, 0}, tensor{1}, 32, data_types::i8, format::b_fs_yx_fsv16, data_types::i8,  format::goiyx, data_types::f32, format::bfyx

#define CASE_DECONV_U8S8_1 {1, 15, 4, 5}, {1, 30, 6, 7}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_U8S8_2 {1, 16, 4, 5}, {1, 32, 6, 7}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::b_fs_yx_fsv16, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_U8S8_3 {1, 16, 4, 5}, {1, 32, 4, 5}, {1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::b_fs_yx_fsv16, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_U8S8_4 {1, 32, 4, 5}, {1, 32, 4, 5}, {1, 1, 3, 3}, tensor{1}, tensor{0, 0, -1, -1}, tensor{1}, 32, data_types::u8, format::b_fs_yx_fsv16, data_types::i8,  format::goiyx, data_types::f32, format::bfyx
#define CASE_DECONV_U8S8_5 {1, 15, 4, 5}, {1, 30, 9, 11}, {1, 1, 3, 3}, tensor{1, 1, 2, 2}, tensor{0}, tensor{1}, 1, data_types::u8, format::bfyx, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_U8S8_6 {1, 16, 4, 5}, {1, 32, 9, 11}, {1, 1, 3, 3}, tensor{1, 1, 2, 2}, tensor{0}, tensor{1}, 1, data_types::u8, format::b_fs_yx_fsv16, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_U8S8_7 {1, 16, 4, 5}, {1, 32, 7, 9}, {1, 1, 1, 1}, tensor{1, 1, 2, 2}, tensor{0}, tensor{1}, 1, data_types::u8, format::b_fs_yx_fsv16, data_types::i8, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_U8S8_8 {1, 32, 4, 5}, {1, 32, 7, 9}, {1, 1, 3, 3}, tensor{1, 1, 2, 2}, tensor{0, 0, -1, -1, 0, 0}, tensor{1}, 32, data_types::u8, format::b_fs_yx_fsv16, data_types::i8,  format::goiyx, data_types::f32, format::bfyx

// 3D
// in_shape; out_shape; kernel; stride; pad; dilation; groups; data_type; input_format; weights_type; weights_format; default_type; default_format;
#define CASE_DECONV_FP32_3D_1 {1, 15, 4, 5, 3}, {1, 30, 6, 7, 5}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::bfzyx, data_types::f32, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_FP32_3D_2 {1, 16, 4, 5, 3}, {1, 32, 6, 7, 5}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::is_os_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_DECONV_FP32_3D_3 {1, 16, 4, 5, 3}, {1, 32, 4, 5, 3}, {1, 1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::is_os_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_DECONV_FP32_3D_4 {1, 32, 4, 5, 3}, {1, 32, 4, 5, 3}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0, 0, -1, -1, -1}, tensor{1}, 32, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32,  format::gs_oizyx_gsv16, data_types::f32, format::bfzyx
#define CASE_DECONV_FP32_3D_5 {1, 15, 4, 5, 3}, {1, 30, 9, 11, 7}, {1, 1, 3, 3, 3}, tensor{1, 1, 2, 2, 2}, tensor{0}, tensor{1}, 1, data_types::f32, format::bfzyx, data_types::f32, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_FP32_3D_6 {1, 16, 4, 5, 3}, {1, 32, 9, 11, 7}, {1, 1, 3, 3, 3}, tensor{1, 1, 2, 2, 2}, tensor{0}, tensor{1}, 1, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::is_os_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_DECONV_FP32_3D_7 {1, 16, 4, 5, 3}, {1, 32, 7, 9, 5}, {1, 1, 1, 1, 1}, tensor{1, 1, 2, 2, 2}, tensor{0}, tensor{1}, 1, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::is_os_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_DECONV_FP32_3D_8 {1, 32, 4, 5, 3}, {1, 32, 7, 9, 5}, {1, 1, 3, 3, 3}, tensor{1, 1, 2, 2, 2}, tensor{0, 0, -1, -1, -1}, tensor{1}, 32, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32,  format::gs_oizyx_gsv16, data_types::f32, format::bfzyx
#define CASE_DECONV_FP32_3D_9 {16, 16, 4, 5, 3}, {16, 32, 7, 9, 5}, {1, 1, 1, 1, 1}, tensor{1, 1, 2, 2, 2}, tensor{0}, tensor{1}, 1, data_types::f32, format::bs_fs_zyx_bsv16_fsv16, data_types::f32, format::is_os_zyx_isv16_osv16, data_types::f32, format::bfzyx

#define CASE_DECONV_FP16_3D_1 {1, 15, 4, 5, 3}, {1, 30, 6, 7, 5}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f16, format::bfzyx, data_types::f16, format::oizyx, data_types::f16, format::bfzyx
#define CASE_DECONV_FP16_3D_2 {1, 16, 4, 5, 3}, {1, 32, 6, 7, 5}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::is_os_zyx_isv16_osv16, data_types::f16, format::bfzyx
#define CASE_DECONV_FP16_3D_3 {1, 16, 4, 5, 3}, {1, 32, 4, 5, 3}, {1, 1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::is_os_zyx_isv16_osv16, data_types::f16, format::bfzyx
#define CASE_DECONV_FP16_3D_4 {1, 32, 4, 5, 3}, {1, 32, 4, 5, 3}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0, 0, -1, -1, -1}, tensor{1}, 32, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16,  format::gs_oizyx_gsv16, data_types::f16, format::bfzyx
#define CASE_DECONV_FP16_3D_5 {1, 15, 4, 5, 3}, {1, 30, 9, 11, 7}, {1, 1, 3, 3, 3}, tensor{1, 1, 2, 2, 2}, tensor{0}, tensor{1}, 1, data_types::f16, format::bfzyx, data_types::f16, format::oizyx, data_types::f16, format::bfzyx
#define CASE_DECONV_FP16_3D_6 {1, 16, 4, 5, 3}, {1, 32, 9, 11, 7}, {1, 1, 3, 3, 3}, tensor{1, 1, 2, 2, 2}, tensor{0}, tensor{1}, 1, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::is_os_zyx_isv16_osv16, data_types::f16, format::bfzyx
#define CASE_DECONV_FP16_3D_7 {1, 16, 4, 5, 3}, {1, 32, 7, 9, 5}, {1, 1, 1, 1, 1}, tensor{1, 1, 2, 2, 2}, tensor{0}, tensor{1}, 1, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::is_os_zyx_isv16_osv16, data_types::f16, format::bfzyx
#define CASE_DECONV_FP16_3D_8 {1, 32, 4, 5, 3}, {1, 32, 7, 9, 5}, {1, 1, 3, 3, 3}, tensor{1, 1, 2, 2, 2}, tensor{0, 0, -1, -1, -1}, tensor{1}, 32, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16,  format::gs_oizyx_gsv16, data_types::f16, format::bfzyx
#define CASE_DECONV_FP16_3D_9 {16, 16, 4, 5, 3}, {16, 32, 7, 9, 5}, {1, 1, 1, 1, 1}, tensor{1, 1, 2, 2, 2}, tensor{0}, tensor{1}, 1, data_types::f16, format::bs_fs_zyx_bsv16_fsv16, data_types::f16, format::is_os_zyx_isv16_osv16, data_types::f16, format::bfzyx

#define CASE_DECONV_S8S8_3D_1 {1, 15, 4, 5, 3}, {1, 30, 6, 7, 5}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::bfzyx, data_types::i8, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_S8S8_3D_2 {1, 16, 4, 5, 3}, {1, 32, 6, 7, 5}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::b_fs_zyx_fsv16, data_types::i8, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_S8S8_3D_3 {1, 16, 4, 5, 3}, {1, 32, 4, 5, 3}, {1, 1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::b_fs_zyx_fsv16, data_types::i8, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_S8S8_3D_4 {1, 32, 4, 5, 3}, {1, 32, 4, 5, 3}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0, 0, -1, -1, -1}, tensor{1}, 32, data_types::i8, format::b_fs_zyx_fsv16, data_types::i8,  format::goizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_S8S8_3D_5 {1, 15, 4, 5, 3}, {1, 30, 9, 11, 7}, {1, 1, 3, 3, 3}, tensor{1, 1, 2, 2, 2}, tensor{0}, tensor{1}, 1, data_types::i8, format::bfzyx, data_types::i8, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_S8S8_3D_6 {1, 16, 4, 5, 3}, {1, 32, 9, 11, 7}, {1, 1, 3, 3, 3}, tensor{1, 1, 2, 2, 2}, tensor{0}, tensor{1}, 1, data_types::i8, format::b_fs_zyx_fsv16, data_types::i8, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_S8S8_3D_7 {1, 16, 4, 5, 3}, {1, 32, 7, 9, 5}, {1, 1, 1, 1, 1}, tensor{1, 1, 2, 2, 2}, tensor{0}, tensor{1}, 1, data_types::i8, format::b_fs_zyx_fsv16, data_types::i8, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_S8S8_3D_8 {1, 32, 4, 5, 3}, {1, 32, 7, 9, 5}, {1, 1, 3, 3, 3}, tensor{1, 1, 2, 2, 2}, tensor{0, 0, -1, -1, -1}, tensor{1}, 32, data_types::i8, format::b_fs_zyx_fsv16, data_types::i8,  format::goizyx, data_types::f32, format::bfzyx

#define CASE_DECONV_U8S8_3D_1 {1, 15, 4, 5, 3}, {1, 30, 6, 7, 5}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::bfzyx, data_types::i8, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_U8S8_3D_2 {1, 16, 4, 5, 3}, {1, 32, 6, 7, 5}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::b_fs_zyx_fsv16, data_types::i8, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_U8S8_3D_3 {1, 16, 4, 5, 3}, {1, 32, 4, 5, 3}, {1, 1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::u8, format::b_fs_zyx_fsv16, data_types::i8, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_U8S8_3D_4 {1, 32, 4, 5, 3}, {1, 32, 4, 5, 3}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0, 0, -1, -1, -1}, tensor{1}, 32, data_types::u8, format::b_fs_zyx_fsv16, data_types::i8,  format::goizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_U8S8_3D_5 {1, 15, 4, 5, 3}, {1, 30, 9, 11, 7}, {1, 1, 3, 3, 3}, tensor{1, 1, 2, 2, 2}, tensor{0}, tensor{1}, 1, data_types::u8, format::bfzyx, data_types::i8, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_U8S8_3D_6 {1, 16, 4, 5, 3}, {1, 32, 9, 11, 7}, {1, 1, 3, 3, 3}, tensor{1, 1, 2, 2, 2}, tensor{0}, tensor{1}, 1, data_types::u8, format::b_fs_zyx_fsv16, data_types::i8, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_U8S8_3D_7 {1, 16, 4, 5, 3}, {1, 32, 7, 9, 5}, {1, 1, 1, 1, 1}, tensor{1, 1, 2, 2, 2}, tensor{0}, tensor{1}, 1, data_types::u8, format::b_fs_zyx_fsv16, data_types::i8, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_U8S8_3D_8 {1, 32, 4, 5, 3}, {1, 32, 7, 9, 5}, {1, 1, 3, 3, 3}, tensor{1, 1, 2, 2, 2}, tensor{0, 0, -1, -1, -1}, tensor{1}, 32, data_types::u8, format::b_fs_zyx_fsv16, data_types::i8,  format::goizyx, data_types::f32, format::bfzyx

#define CASE_DECONV_ELTW_FP32_1 {1, 16, 4, 5}, {1, 32, 6, 7}, {1, 32, 1, 1}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::oiyx, data_types::f32, format::bfyx
#define CASE_DECONV_ELTW_FP32_2 {1, 16, 4, 5}, {1, 32, 6, 7}, {1, 1, 1, 1}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::os_is_yx_isv16_osv16, data_types::f32, format::bfyx
#define CASE_DECONV_ELTW_FP32_3 {1, 16, 4, 5}, {1, 32, 4, 5}, {1, 1, 1, 1}, {1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::is_os_yx_isv16_osv16, data_types::f32, format::bfyx
#define CASE_DECONV_ELTW_FP32_4 {1, 15, 4, 5, 3}, {1, 30, 6, 7, 5}, {1, 1, 6, 7, 5}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::bfzyx, data_types::f32, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_ELTW_FP32_5 {1, 15, 4, 5, 4}, {1, 30, 6, 7, 6}, {1, 30, 6, 1, 6}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::bfzyx, data_types::f32, format::oizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_ELTW_FP32_6 {1, 32, 2, 2, 2}, {1, 16, 4, 4, 4}, {1, 16, 1, 4, 1}, {1, 1, 3, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::os_is_zyx_isv16_osv16, data_types::f32, format::bfzyx
#define CASE_DECONV_ELTW_FP32_7 {1, 16, 3, 5}, {1, 32, 5, 7}, {1, 32, 1, 7}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::os_is_yx_isv16_osv16, data_types::f32, format::bfyx
#define CASE_DECONV_ELTW_FP32_8 {1, 32, 4, 5}, {1, 32, 7, 9}, {1, 32, 1, 1}, {1, 1, 3, 3}, tensor{1, 1, 2, 2}, tensor{0, 0, -1, -1, 0, 0}, tensor{1}, 32, data_types::f32, format::b_fs_yx_fsv16, data_types::f32,  format::gs_oiyx_gsv16, data_types::f32, format::bfyx

#define CASE_DECONV_ELTW_i8_1 {1, 16, 3, 5}, {1, 32, 5, 7}, {1, 32, 5, 1}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::b_fs_yx_fsv16, data_types::i8, format::os_is_yx_osv16_isv16, data_types::f32, format::bfyx
#define CASE_DECONV_ELTW_i8_2 {1, 32, 4, 5, 3}, {1, 32, 6, 7, 5}, {1, 32, 1, 1, 1}, {1, 1, 3, 3, 3}, tensor{1, 1, 2, 2, 2}, tensor{0, 0, -1, -1, -1}, tensor{1}, 32, data_types::u8, format::b_fs_zyx_fsv16, data_types::i8,  format::goizyx, data_types::f32, format::bfzyx
#define CASE_DECONV_ELTW_i8_3 {1, 5, 5, 5, 5}, {1, 5, 5, 5, 5}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::bfzyx, data_types::i8, format::oiyx, data_types::f32, format::bfzyx
#define CASE_DECONV_ELTW_i8_4 {1, 16, 1, 4}, {1, 16, 1, 6}, {1, 16, 1, 1}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::b_fs_yx_fsv16, data_types::i8, format::os_is_yx_osv16_isv16, data_types::f32, format::bfyx
#define CASE_DECONV_ELTW_i8_5 {1, 16, 2, 4}, {1, 16, 4, 6}, {1, 16, 4, 1}, {1, 1, 3, 3}, tensor{1}, tensor{0}, tensor{1}, 1, data_types::i8, format::b_fs_yx_fsv16, data_types::i8, format::os_is_yx_osv16_isv16, data_types::f32, format::bfyx


class DeconvolutionFusingTest : public ::WeightsPrimitiveFusingTest<bc_test_params> {
public:
    void execute(deconv_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
        auto find_conv = [](primitive_info& p) -> bool {
            if (p.original_id == "deconv")
                return true;
            return false;
        };

        auto pi_fused = network_fused.get_primitives_info();
        auto info_fused = std::find_if(pi_fused.begin(), pi_fused.end(), find_conv);
        if (info_fused != pi_fused.end())
            std::cout << "kernel: " << info_fused->kernel_id << std::endl;
    }
};

class deconv_actv : public DeconvolutionFusingTest {};
TEST_P(deconv_actv, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        deconvolution("deconv", "input", { "weights" }, p.groups, p.stride, p.pad),
        activation("act", "deconv", activation_func::relu),
        reorder("out", "act", p.default_format, data_types::f32)
    );
    // Need much higher tolerance because of deconvolution -> convolution optimization
    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, deconv_actv,
    ::testing::ValuesIn(std::vector<deconv_test_params>{
        deconv_test_params{ CASE_DECONV_FP32_1, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP32_2, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP32_3, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP32_4, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP32_5, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP32_6, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP32_7, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP32_8, 2, 3 },

        deconv_test_params{ CASE_DECONV_FP16_1, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP16_2, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP16_3, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP16_4, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP16_5, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP16_6, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP16_7, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP16_8, 2, 3 },

        deconv_test_params{ CASE_DECONV_U8S8_1, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_2, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_3, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_4, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_5, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_6, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_7, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_8, 2, 3 },

        deconv_test_params{ CASE_DECONV_S8S8_1, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_2, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_3, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_4, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_5, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_6, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_7, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_8, 2, 3 },

        deconv_test_params{ CASE_DECONV_FP32_3D_1, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP32_3D_2, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP32_3D_3, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP32_3D_4, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP32_3D_5, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP32_3D_6, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP32_3D_7, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP32_3D_8, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP32_3D_9, 2, 3 },

        deconv_test_params{ CASE_DECONV_FP16_3D_1, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP16_3D_2, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP16_3D_3, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP16_3D_4, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP16_3D_5, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP16_3D_6, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP16_3D_7, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP16_3D_8, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP16_3D_9, 2, 3 },

        deconv_test_params{ CASE_DECONV_U8S8_3D_1, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_2, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_3, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_4, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_5, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_6, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_7, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_8, 2, 3 },

        deconv_test_params{ CASE_DECONV_S8S8_3D_1, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_2, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_3, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_4, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_5, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_6, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_7, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_8, 2, 3 },
}));


class deconv_bias : public DeconvolutionFusingTest {};
TEST_P(deconv_bias, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("bias", get_mem(get_bias_layout(p))),
        deconvolution("deconv", "input", { "weights" }, p.groups, p.stride, p.pad),
        eltwise("bias_add", {"deconv", "bias"}, eltwise_mode::sum),
        reorder("out", "bias_add", p.default_format, data_types::f32)
    );
    // Need much higher tolerance because of deconvolution -> convolution optimization
    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, deconv_bias,
    ::testing::ValuesIn(std::vector<deconv_test_params>{
        deconv_test_params{ CASE_DECONV_FP32_1, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP32_2, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP32_3, 2, 3 },

        deconv_test_params{ CASE_DECONV_FP16_1, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP16_2, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP16_3, 2, 3 },

        deconv_test_params{ CASE_DECONV_U8S8_1, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_2, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_3, 2, 3 },

        deconv_test_params{ CASE_DECONV_S8S8_1, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_2, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_3, 2, 3 },

        deconv_test_params{ CASE_DECONV_FP32_3D_1, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP32_3D_2, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP32_3D_3, 2, 3 },

        deconv_test_params{ CASE_DECONV_FP16_3D_1, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP16_3D_2, 2, 3 },
        deconv_test_params{ CASE_DECONV_FP16_3D_3, 2, 3 },

        deconv_test_params{ CASE_DECONV_U8S8_3D_1, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_2, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_3, 2, 3 },

        deconv_test_params{ CASE_DECONV_S8S8_3D_1, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_2, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_3, 2, 3 },
}));

class deconv_scale : public DeconvolutionFusingTest {};
TEST_P(deconv_scale, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count())),
        deconvolution("deconv", "input", { "weights" }, p.groups, p.stride, p.pad),
        scale("scale", "deconv", "scale_data"),
        reorder("out", "scale", p.default_format, data_types::f32)
    );
    tolerance = 1e-5f;
    execute(p);
}

TEST_P(deconv_scale, fp16_scale_out) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/p.kernel.count())),
        deconvolution("deconv", "input", { "weights" }, p.groups, p.stride, p.pad),
        scale("scale", "deconv", "scale_data", optional_data_type{data_types::f16}),
        reorder("out", "scale", p.default_format, data_types::f32)
    );
    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, deconv_scale,
    ::testing::ValuesIn(std::vector<deconv_test_params>{
        deconv_test_params{ CASE_DECONV_U8S8_1, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_2, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_3, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_4, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_5, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_6, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_7, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_8, 2, 3 },

        deconv_test_params{ CASE_DECONV_S8S8_1, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_2, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_3, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_4, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_5, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_6, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_7, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_8, 2, 3 },

        deconv_test_params{ CASE_DECONV_U8S8_3D_1, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_2, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_3, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_4, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_5, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_6, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_7, 2, 3 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_8, 2, 3 },

        deconv_test_params{ CASE_DECONV_S8S8_3D_1, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_2, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_3, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_4, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_5, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_6, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_7, 2, 3 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_8, 2, 3 },
}));

class deconv_actv_eltw_actv : public DeconvolutionFusingTest {};
TEST_P(deconv_actv_eltw_actv, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("eltw_data", get_mem(get_output_layout(p))),
        deconvolution("deconv", "input", { "weights" }, p.groups, p.stride, p.pad),
        activation("act1", "deconv", activation_func::relu),
        eltwise("eltw", {"act1", "eltw_data"}, eltwise_mode::sum),
        activation("act2", "eltw", activation_func::relu),
        reorder("out", "act2", p.default_format, data_types::f32)
    );
    // Need much higher tolerance because of deconvolution -> convolution optimization
    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, deconv_actv_eltw_actv,
    ::testing::ValuesIn(std::vector<deconv_test_params>{
        // Some fusings disabled under deconvolution -> convolution optimization
        deconv_test_params{ CASE_DECONV_FP32_1, 3, 5 },
        deconv_test_params{ CASE_DECONV_FP32_2, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP32_3, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP32_4, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP32_5, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP32_6, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP32_7, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP32_8, 2, 5 },

        deconv_test_params{ CASE_DECONV_FP16_1, 3, 5 },
        deconv_test_params{ CASE_DECONV_FP16_2, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP16_3, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP16_4, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP16_5, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP16_6, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP16_7, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP16_8, 2, 5 },

        deconv_test_params{ CASE_DECONV_U8S8_1, 2, 5 },
        deconv_test_params{ CASE_DECONV_U8S8_2, 2, 5 },
        deconv_test_params{ CASE_DECONV_U8S8_3, 2, 5 },
        deconv_test_params{ CASE_DECONV_U8S8_4, 2, 5 },
        deconv_test_params{ CASE_DECONV_U8S8_5, 2, 5 },
        deconv_test_params{ CASE_DECONV_U8S8_6, 2, 5 },
        deconv_test_params{ CASE_DECONV_U8S8_7, 2, 5 },
        deconv_test_params{ CASE_DECONV_U8S8_8, 2, 5 },

        deconv_test_params{ CASE_DECONV_S8S8_1, 2, 5 },
        deconv_test_params{ CASE_DECONV_S8S8_2, 2, 5 },
        deconv_test_params{ CASE_DECONV_S8S8_3, 2, 5 },
        deconv_test_params{ CASE_DECONV_S8S8_4, 2, 5 },
        deconv_test_params{ CASE_DECONV_S8S8_5, 2, 5 },
        deconv_test_params{ CASE_DECONV_S8S8_6, 2, 5 },
        deconv_test_params{ CASE_DECONV_S8S8_7, 2, 5 },
        deconv_test_params{ CASE_DECONV_S8S8_8, 2, 5 },

        deconv_test_params{ CASE_DECONV_FP32_3D_1, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP32_3D_2, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP32_3D_3, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP32_3D_4, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP32_3D_5, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP32_3D_6, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP32_3D_7, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP32_3D_8, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP32_3D_9, 2, 5 },

        deconv_test_params{ CASE_DECONV_FP16_3D_1, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP16_3D_2, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP16_3D_3, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP16_3D_4, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP16_3D_5, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP16_3D_6, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP16_3D_7, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP16_3D_8, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP16_3D_9, 2, 5 },

        deconv_test_params{ CASE_DECONV_U8S8_3D_1, 2, 5 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_2, 2, 5 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_3, 2, 5 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_4, 2, 5 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_5, 2, 5 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_6, 2, 5 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_7, 2, 5 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_8, 2, 5 },

        deconv_test_params{ CASE_DECONV_S8S8_3D_1, 2, 5 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_2, 2, 5 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_3, 2, 5 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_4, 2, 5 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_5, 2, 5 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_6, 2, 5 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_7, 2, 5 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_8, 2, 5 },
}));

class deconv_scale_actv_quant_i8 : public DeconvolutionFusingTest {};
TEST_P(deconv_scale_actv_quant_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.f/p.kernel.count())),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        deconvolution("deconv", "input", { "weights" }, p.groups, p.stride, p.pad),
        scale("scale", "deconv", "scale_data"),
        activation("actv", "scale", activation_func::softsign),
        quantize("quant", "actv", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("out", "quant", p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, deconv_scale_actv_quant_i8,
    ::testing::ValuesIn(std::vector<deconv_test_params>{
        deconv_test_params{ CASE_DECONV_FP32_1, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP32_2, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP32_3, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP32_4, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP32_5, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP32_6, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP32_7, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP32_8, 2, 5 },

        deconv_test_params{ CASE_DECONV_FP16_1, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP16_2, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP16_3, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP16_4, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP16_5, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP16_6, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP16_7, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP16_8, 2, 5 },

        deconv_test_params{ CASE_DECONV_U8S8_1, 2, 5 },
        deconv_test_params{ CASE_DECONV_U8S8_2, 2, 5 },
        deconv_test_params{ CASE_DECONV_U8S8_3, 2, 5 },
        deconv_test_params{ CASE_DECONV_U8S8_4, 2, 5 },
        deconv_test_params{ CASE_DECONV_U8S8_5, 2, 5 },
        deconv_test_params{ CASE_DECONV_U8S8_6, 2, 5 },
        deconv_test_params{ CASE_DECONV_U8S8_7, 2, 5 },
        deconv_test_params{ CASE_DECONV_U8S8_8, 2, 5 },

        deconv_test_params{ CASE_DECONV_S8S8_1, 2, 5 },
        deconv_test_params{ CASE_DECONV_S8S8_2, 2, 5 },
        deconv_test_params{ CASE_DECONV_S8S8_3, 2, 5 },
        deconv_test_params{ CASE_DECONV_S8S8_4, 2, 5 },
        deconv_test_params{ CASE_DECONV_S8S8_5, 2, 5 },
        deconv_test_params{ CASE_DECONV_S8S8_6, 2, 5 },
        deconv_test_params{ CASE_DECONV_S8S8_7, 2, 5 },
        deconv_test_params{ CASE_DECONV_S8S8_8, 2, 5 },

        deconv_test_params{ CASE_DECONV_FP32_3D_1, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP32_3D_2, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP32_3D_3, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP32_3D_4, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP32_3D_5, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP32_3D_6, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP32_3D_7, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP32_3D_8, 2, 5 },
        // FIXME no quantize implementation for bs_fs_yx_bsv16_fsv16 format AND add_required_reorders pass completely ruins data types
        // add_required_reorders pass tries to reorder everything to output type if no format exists, this ruins fp32 -> int8 quantize
        //deconv_test_params{ CASE_DECONV_FP32_3D_9, 3, 5 },

        deconv_test_params{ CASE_DECONV_FP16_3D_1, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP16_3D_2, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP16_3D_3, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP16_3D_4, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP16_3D_5, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP16_3D_6, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP16_3D_7, 2, 5 },
        deconv_test_params{ CASE_DECONV_FP16_3D_8, 2, 5 },
        //deconv_test_params{ CASE_DECONV_FP16_3D_9, 3, 5 },

        deconv_test_params{ CASE_DECONV_U8S8_3D_1, 2, 5 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_2, 2, 5 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_3, 2, 5 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_4, 2, 5 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_5, 2, 5 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_6, 2, 5 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_7, 2, 5 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_8, 2, 5 },

        deconv_test_params{ CASE_DECONV_S8S8_3D_1, 2, 5 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_2, 2, 5 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_3, 2, 5 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_4, 2, 5 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_5, 2, 5 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_6, 2, 5 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_7, 2, 5 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_8, 2, 5 },
}));

class deconv_scale_actv_quant_u8_eltw_scale_actv_quant_i8 : public DeconvolutionFusingTest {};
TEST_P(deconv_scale_actv_quant_u8_eltw_scale_actv_quant_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("scale1_data", get_mem(get_per_channel_layout(p), 1.f / p.kernel.count())),
        data("in1_lo", get_mem(get_per_channel_layout(p), 0)),
        data("in1_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out1_lo", get_mem(get_single_element_layout(p), 0)),
        data("out1_hi", get_mem(get_single_element_layout(p), 255)),
        data("eltw_data", get_mem(layout(p.default_type, p.input_format, p.out_shape))),
        data("scale2_data", get_mem(get_per_channel_layout(p), 1.f / p.kernel.count())),
        data("in2_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in2_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out2_lo", get_mem(get_single_element_layout(p), -127)),
        data("out2_hi", get_mem(get_single_element_layout(p), 127)),
        deconvolution("deconv", "input", { "weights" }, p.groups, p.stride, p.pad),
        scale("scale1", "deconv", "scale1_data"),
        activation("actv1", "scale1", activation_func::relu),
        quantize("quant1", "actv1", "in1_lo", "in1_hi", "out1_lo", "out1_hi", 256, data_types::u8),
        eltwise("eltw", {"quant1", "eltw_data"}, eltwise_mode::sum, p.default_type),
        scale("scale2", "eltw", "scale2_data"),
        activation("actv2", "scale2", activation_func::relu),
        quantize("quant2", "actv2", "in2_lo", "in2_hi", "out2_lo", "out2_hi", 255, data_types::i8),
        reorder("out", "quant2", p.default_format, data_types::f32)
    );

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, deconv_scale_actv_quant_u8_eltw_scale_actv_quant_i8,
    ::testing::ValuesIn(std::vector<deconv_test_params>{
        deconv_test_params{ CASE_DECONV_FP32_1, 2, 9 },
        deconv_test_params{ CASE_DECONV_FP32_2, 2, 9 },
        deconv_test_params{ CASE_DECONV_FP32_3, 2, 9 },
        deconv_test_params{ CASE_DECONV_FP32_4, 2, 9 },
        deconv_test_params{ CASE_DECONV_FP32_5, 2, 9 },
        deconv_test_params{ CASE_DECONV_FP32_6, 2, 9 },
        deconv_test_params{ CASE_DECONV_FP32_7, 2, 9 },
        deconv_test_params{ CASE_DECONV_FP32_8, 2, 9 },

        deconv_test_params{ CASE_DECONV_FP16_1, 2, 9 },
        deconv_test_params{ CASE_DECONV_FP16_2, 2, 9 },
        deconv_test_params{ CASE_DECONV_FP16_3, 2, 9 },
        deconv_test_params{ CASE_DECONV_FP16_4, 2, 9 },
        deconv_test_params{ CASE_DECONV_FP16_5, 2, 9 },
        deconv_test_params{ CASE_DECONV_FP16_6, 2, 9 },
        deconv_test_params{ CASE_DECONV_FP16_7, 2, 9 },
        deconv_test_params{ CASE_DECONV_FP16_8, 2, 9 },

        deconv_test_params{ CASE_DECONV_U8S8_1, 2, 9 },
        deconv_test_params{ CASE_DECONV_U8S8_2, 2, 9 },
        deconv_test_params{ CASE_DECONV_U8S8_3, 2, 9 },
        deconv_test_params{ CASE_DECONV_U8S8_4, 2, 9 },
        deconv_test_params{ CASE_DECONV_U8S8_5, 2, 9 },
        deconv_test_params{ CASE_DECONV_U8S8_6, 2, 9 },
        deconv_test_params{ CASE_DECONV_U8S8_7, 2, 9 },
        deconv_test_params{ CASE_DECONV_U8S8_8, 2, 9 },

        deconv_test_params{ CASE_DECONV_S8S8_1, 2, 9 },
        deconv_test_params{ CASE_DECONV_S8S8_2, 2, 9 },
        deconv_test_params{ CASE_DECONV_S8S8_3, 2, 9 },
        deconv_test_params{ CASE_DECONV_S8S8_4, 2, 9 },
        deconv_test_params{ CASE_DECONV_S8S8_5, 2, 9 },
        deconv_test_params{ CASE_DECONV_S8S8_6, 2, 9 },
        deconv_test_params{ CASE_DECONV_S8S8_7, 2, 9 },
        deconv_test_params{ CASE_DECONV_S8S8_8, 2, 9 },

        deconv_test_params{ CASE_DECONV_FP32_3D_1, 2, 9 },
        deconv_test_params{ CASE_DECONV_FP32_3D_2, 2, 9 },
        deconv_test_params{ CASE_DECONV_FP32_3D_3, 2, 9 },
        deconv_test_params{ CASE_DECONV_FP32_3D_4, 2, 9 },
        deconv_test_params{ CASE_DECONV_FP32_3D_5, 2, 9 },
        deconv_test_params{ CASE_DECONV_FP32_3D_6, 2, 9 },
        // deconv_test_params{ CASE_DECONV_FP32_3D_7, 2, 9 },
        deconv_test_params{ CASE_DECONV_FP32_3D_8, 2, 9 },
        // deconv_test_params{ CASE_DECONV_FP32_3D_9, 6, 9 },

        deconv_test_params{ CASE_DECONV_FP16_3D_1, 2, 9 },
        deconv_test_params{ CASE_DECONV_FP16_3D_2, 2, 9 },
        deconv_test_params{ CASE_DECONV_FP16_3D_3, 2, 9 },
        deconv_test_params{ CASE_DECONV_FP16_3D_4, 2, 9 },
        deconv_test_params{ CASE_DECONV_FP16_3D_5, 2, 9 },
        deconv_test_params{ CASE_DECONV_FP16_3D_6, 2, 9 },
        deconv_test_params{ CASE_DECONV_FP16_3D_7, 2, 9 },
        deconv_test_params{ CASE_DECONV_FP16_3D_8, 2, 9 },
        // deconv_test_params{ CASE_DECONV_FP16_3D_9, 6, 9 },

        deconv_test_params{ CASE_DECONV_U8S8_3D_1, 2, 9 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_2, 2, 9 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_3, 2, 9 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_4, 2, 9 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_5, 2, 9 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_6, 2, 9 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_7, 2, 9 },
        deconv_test_params{ CASE_DECONV_U8S8_3D_8, 2, 9 },

        deconv_test_params{ CASE_DECONV_S8S8_3D_1, 2, 9 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_2, 2, 9 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_3, 2, 9 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_4, 2, 9 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_5, 2, 9 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_6, 2, 9 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_7, 2, 9 },
        deconv_test_params{ CASE_DECONV_S8S8_3D_8, 2, 9 },
}));

class deconv_scale_activation_quantize_i8_eltwise_quantize_u8 : public ConvEltwTest {};
TEST_P(deconv_scale_activation_quantize_i8_eltwise_quantize_u8, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 deconvolution("deconv_prim", "input", { "weights" }, p.groups, p.stride, p.pad),
                 data("scale_data", get_mem(get_per_channel_layout(p), 1.f / p.kernel.count())),
                 scale("scale", "deconv_prim", "scale_data"),
                 activation("activation", "scale", activation_func::relu),
                 data("in_low", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_high", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_low", get_mem(get_single_element_layout(p), -127)),
                 data("out_high", get_mem(get_single_element_layout(p), 127)),
                 quantize("quant", "activation", "in_low", "in_high", "out_low", "out_high", 255, data_types::i8),
                 data("eltwise_data", get_mem(layout{ p.data_type, p.input_format, p.eltw_shape })),
                 eltwise("eltw", { "quant", "eltwise_data" }, eltwise_mode::sum, p.default_type),
                 data("in_low2", get_mem(get_per_channel_layout(p), min_random, 0)),
                 data("in_high2", get_mem(get_per_channel_layout(p), 1, max_random)),
                 data("out_low2", get_mem(get_single_element_layout(p), 0)),
                 data("out_high2", get_mem(get_single_element_layout(p), 255)),
                 quantize("quant2", "eltw", "in_low2", "in_high2", "out_low2", "out_high2", 256, data_types::u8),
                 reorder("reorder_bfyx", "quant2", p.default_format, data_types::f32)
    );
    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, deconv_scale_activation_quantize_i8_eltwise_quantize_u8,
                        ::testing::ValuesIn(std::vector<conv_eltw_test_params>{
                                conv_eltw_test_params{CASE_DECONV_ELTW_FP32_1, 2, 7},
                                conv_eltw_test_params{CASE_DECONV_ELTW_FP32_2, 2, 7},
                                conv_eltw_test_params{CASE_DECONV_ELTW_FP32_3, 2, 7},
                                conv_eltw_test_params{CASE_DECONV_ELTW_FP32_4, 2, 7},
                                conv_eltw_test_params{CASE_DECONV_ELTW_FP32_5, 2, 7},
                                conv_eltw_test_params{CASE_DECONV_ELTW_FP32_6, 2, 7},
                                conv_eltw_test_params{CASE_DECONV_ELTW_FP32_7, 2, 7},
                                conv_eltw_test_params{CASE_DECONV_ELTW_FP32_8, 2, 7},

                                conv_eltw_test_params{CASE_DECONV_ELTW_i8_1, 2, 7},
                                conv_eltw_test_params{CASE_DECONV_ELTW_i8_2, 2, 7},
                                conv_eltw_test_params{CASE_DECONV_ELTW_i8_3, 2, 7},
                                conv_eltw_test_params{CASE_DECONV_ELTW_i8_4, 2, 7},
                                conv_eltw_test_params{CASE_DECONV_ELTW_i8_5, 2, 7},

                        }));


class deconv_activation_eltwise_diff_sizes : public ConvEltwTest {};
TEST_P(deconv_activation_eltwise_diff_sizes, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                 data("weights", get_mem(get_weights_layout(p))),
                 data("eltwise_data", get_mem(layout{ p.data_type, p.input_format, p.eltw_shape })),
                 deconvolution("deconv_prim", "input", { "weights" }, p.groups, p.stride, p.pad),
                 activation("activation", "deconv_prim", activation_func::relu),
                 eltwise("sum", { "activation", "eltwise_data" }, eltwise_mode::sum, p.default_type),
                 reorder("reorder_bfyx", "sum", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, deconv_activation_eltwise_diff_sizes,
                        ::testing::ValuesIn(std::vector<conv_eltw_test_params>{
                                conv_eltw_test_params{CASE_DECONV_ELTW_FP32_1, 2, 4},
                                conv_eltw_test_params{CASE_DECONV_ELTW_FP32_2, 2, 4},
                                conv_eltw_test_params{CASE_DECONV_ELTW_FP32_3, 2, 4},
                                conv_eltw_test_params{CASE_DECONV_ELTW_FP32_4, 2, 4},
                                conv_eltw_test_params{CASE_DECONV_ELTW_FP32_5, 2, 4},
                                conv_eltw_test_params{CASE_DECONV_ELTW_FP32_6, 2, 4},
                                conv_eltw_test_params{CASE_DECONV_ELTW_FP32_7, 2, 4},
                                conv_eltw_test_params{CASE_DECONV_ELTW_FP32_8, 2, 4},

                                conv_eltw_test_params{CASE_DECONV_ELTW_i8_1, 2, 4},
                                conv_eltw_test_params{CASE_DECONV_ELTW_i8_2, 2, 4},
                                conv_eltw_test_params{CASE_DECONV_ELTW_i8_3, 2, 4},
                                conv_eltw_test_params{CASE_DECONV_ELTW_i8_4, 2, 4},
                                conv_eltw_test_params{CASE_DECONV_ELTW_i8_5, 2, 4},
                        }));

/* ----------------------------------------------------------------------------------------------------- */
/* --------------------------------------- Pooling cases ----------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
struct pooling_test_params {
    tensor in_shape;
    data_types data_type;
    format input_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
    pooling_mode pool_mode;
    std::string kernel_name;
};

#define CASE_POOLING_F32_1 {1, 16, 8, 8}, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_POOLING_F32_2 {2, 16, 8, 8}, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_POOLING_F32_3 {1, 32, 10, 10}, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_POOLING_F32_4 {1, 32, 10, 10}, data_types::f32, format::fs_b_yx_fsv32, data_types::f32, format::bfyx
#define CASE_POOLING_F32_5 {1, 32, 10, 10}, data_types::f32, format::byxf, data_types::f32, format::bfyx
#define CASE_POOLING_F32_6 {1, 32, 40, 40}, data_types::f32, format::byxf, data_types::f32, format::bfyx
#define CASE_POOLING_F32_7 {16, 32, 10, 10}, data_types::f32, format::bs_fs_yx_bsv16_fsv16, data_types::f32, format::bfyx
#define CASE_POOLING_F32_8 {16, 32, 10, 10}, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_POOLING_F32_9 {16, 32, 10, 10}, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::bfyx
#define CASE_POOLING_F32_10 {16, 32, 10, 10, 10}, data_types::f32, format::bs_fs_zyx_bsv16_fsv16, data_types::f32, format::bfyx

#define CASE_POOLING_F32_F16_1 {1, 16, 8, 8}, data_types::f32, format::bfyx, data_types::f16, format::bfyx
#define CASE_POOLING_F32_F16_2 {2, 16, 8, 8}, data_types::f32, format::bfyx, data_types::f16, format::bfyx
#define CASE_POOLING_F32_F16_3 {1, 32, 10, 10}, data_types::f32, format::bfyx, data_types::f16, format::bfyx
#define CASE_POOLING_F32_F16_4 {1, 32, 10, 10}, data_types::f32, format::fs_b_yx_fsv32, data_types::f16, format::bfyx
#define CASE_POOLING_F32_F16_5 {1, 32, 10, 10}, data_types::f32, format::byxf, data_types::f16, format::bfyx
#define CASE_POOLING_F32_F16_6 {1, 32, 40, 40}, data_types::f32, format::byxf, data_types::f16, format::bfyx
#define CASE_POOLING_F32_F16_7 {16, 32, 10, 10}, data_types::f32, format::bs_fs_yx_bsv16_fsv16, data_types::f16, format::bfyx
#define CASE_POOLING_F32_F16_8 {16, 32, 10, 10}, data_types::f32, format::b_fs_yx_fsv16, data_types::f16, format::bfyx
#define CASE_POOLING_F32_F16_9 {16, 32, 10, 10}, data_types::f32, format::b_fs_zyx_fsv16, data_types::f16, format::bfyx
#define CASE_POOLING_F32_F16_10 {16, 32, 10, 10, 10}, data_types::f32, format::bs_fs_zyx_bsv16_fsv16, data_types::f16, format::bfyx

#define CASE_POOLING_F16_1 {1, 16, 8, 8}, data_types::f16, format::bfyx, data_types::f32, format::bfyx
#define CASE_POOLING_F16_3 {1, 32, 10, 10}, data_types::f16, format::bfyx, data_types::f32, format::bfyx
#define CASE_POOLING_F16_4 {1, 32, 10, 10}, data_types::f16, format::fs_b_yx_fsv32, data_types::f32, format::bfyx
#define CASE_POOLING_F16_5 {1, 32, 10, 10}, data_types::f16, format::byxf, data_types::f32, format::bfyx
#define CASE_POOLING_F16_6 {1, 32, 40, 40}, data_types::f16, format::byxf, data_types::f32, format::bfyx
#define CASE_POOLING_F16_7 {16, 32, 10, 10}, data_types::f16, format::bs_fs_yx_bsv16_fsv16, data_types::f32, format::bfyx
#define CASE_POOLING_F16_8 {16, 32, 10, 10}, data_types::f16, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_POOLING_F16_9 {16, 32, 10, 10, 10}, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::bfyx
#define CASE_POOLING_F16_10 {16, 32, 10, 10, 10}, data_types::f32, format::bs_fs_zyx_bsv16_fsv16, data_types::f32, format::bfyx

#define CASE_POOLING_F16_FP16_1 {1, 32, 10, 10}, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_POOLING_F16_FP16_2 {1, 32, 10, 10}, data_types::f16, format::fs_b_yx_fsv32, data_types::f16, format::bfyx
#define CASE_POOLING_F16_FP16_3 {1, 32, 10, 10}, data_types::f16, format::byxf, data_types::f16, format::bfyx
#define CASE_POOLING_F16_FP16_4 {1, 32, 40, 40}, data_types::f16, format::byxf, data_types::f16, format::bfyx
#define CASE_POOLING_F16_FP16_5 {16, 32, 10, 10}, data_types::f16, format::bs_fs_yx_bsv16_fsv16, data_types::f16, format::bfyx
#define CASE_POOLING_F16_FP16_6 {16, 32, 10, 10}, data_types::f16, format::b_fs_yx_fsv16, data_types::f16, format::bfyx
#define CASE_POOLING_F16_FP16_7 {16, 32, 10, 10, 10}, data_types::f16, format::b_fs_zyx_fsv16, data_types::f16, format::bfyx
#define CASE_POOLING_F16_FP16_8 {16, 32, 10, 10, 10}, data_types::f16, format::bs_fs_zyx_bsv16_fsv16, data_types::f16, format::bfyx

#define CASE_POOLING_U8_1 {1, 16, 8, 8}, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_POOLING_U8_2 {2, 16, 8, 8}, data_types::u8, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_POOLING_U8_3 {1, 32, 10, 10}, data_types::u8, format::b_fs_yx_fsv4, data_types::f32, format::b_fs_yx_fsv4
#define CASE_POOLING_U8_5 {16, 32, 10, 10, 10}, data_types::u8, format::b_fs_zyx_fsv32, data_types::f32, format::bfyx
#define CASE_POOLING_U8_6 {16, 32, 10, 10, 10}, data_types::u8, format::b_fs_zyx_fsv32, data_types::f32, format::bfyx

#define CASE_POOLING_U8_FP16_3 {1, 32, 10, 10}, data_types::u8, format::b_fs_yx_fsv4, data_types::f16, format::b_fs_yx_fsv4
#define CASE_POOLING_U8_FP16_5 {16, 32, 10, 10, 10}, data_types::u8, format::b_fs_zyx_fsv32, data_types::f16, format::bfyx
#define CASE_POOLING_U8_FP16_6 {16, 32, 10, 10, 10}, data_types::u8, format::b_fs_zyx_fsv32, data_types::f16, format::bfyx

#define CASE_POOLING_I8_1 {1, 16, 8, 8}, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_POOLING_I8_2 {2, 16, 8, 8}, data_types::i8, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_POOLING_I8_5 {1, 32, 10, 10}, data_types::i8, format::b_fs_yx_fsv4, data_types::f32, format::b_fs_yx_fsv4
#define CASE_POOLING_I8_6 {16, 32, 10, 10, 10}, data_types::i8, format::b_fs_zyx_fsv32, data_types::f32, format::bfyx

#define CASE_POOLING_I8_FP16_5 {1, 32, 10, 10}, data_types::i8, format::b_fs_yx_fsv4, data_types::f16, format::b_fs_yx_fsv4
#define CASE_POOLING_I8_FP16_6 {16, 32, 10, 10, 10}, data_types::i8, format::b_fs_zyx_fsv32, data_types::f16, format::bfyx

class PoolingFusingTest : public ::BaseFusingTest<pooling_test_params> {
public:
    void execute(pooling_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        build_options options;
        options.set_option(build_option::optimize_data(true));
        if (!p.kernel_name.empty()) {
            implementation_desc impl = {p.input_format, p.kernel_name};
            options.set_option(build_option::force_implementations({{"pooling", impl}}));
        }
        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, options);

        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        ASSERT_FALSE(network_fused.get_primitives_info().empty());
        ASSERT_FALSE(network_not_fused.get_primitives_info().empty());

        auto find_and_check = [&](primitive_info& p) -> bool {
            if (p.original_id == "pooling" || p.original_id == "output_reorder")
                return true;
            return false;
        };

        auto pi_fused = network_fused.get_primitives_info();
        auto pi_not_fused = network_not_fused.get_primitives_info();
        auto info_fused = std::find_if(pi_fused.begin(), pi_fused.end(), find_and_check);
        auto info_not_fused = std::find_if(pi_not_fused.begin(), pi_not_fused.end(), find_and_check);

        ASSERT_TRUE(info_fused != pi_fused.end());
        ASSERT_TRUE(info_not_fused != pi_not_fused.end());

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(pooling_test_params& p) { return layout{p.data_type, p.input_format, p.in_shape}; }
    layout get_per_channel_layout(pooling_test_params& p) {
        return layout{p.default_type, p.default_format, tensor{1, p.in_shape.feature[0], 1, 1}};
    }
};

class pooling_f32_activation : public PoolingFusingTest {};
TEST_P(pooling_f32_activation, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        pooling("pooling", "input", p.pool_mode, tensor{1, 1, 3, 3}, tensor{1}, tensor{0, 0, -1, -1, 0, 0}),
        activation("act", "pooling", activation_func::relu),
        reorder("output_reorder", "act", format::bfyx, data_types::f32));

    tolerance = 1e-05f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu,
                        pooling_f32_activation,
                        ::testing::ValuesIn(std::vector<pooling_test_params>{
                            pooling_test_params{CASE_POOLING_F32_1, 2, 3, pooling_mode::max, ""},
                            pooling_test_params{CASE_POOLING_F32_1, 2, 3, pooling_mode::average, ""},
                            pooling_test_params{CASE_POOLING_F16_1, 2, 3, pooling_mode::max, ""},
                            pooling_test_params{CASE_POOLING_F16_1, 2, 3, pooling_mode::average, ""},
                            pooling_test_params{CASE_POOLING_I8_1, 2, 3, pooling_mode::max, ""},
                            pooling_test_params{CASE_POOLING_I8_1, 2, 3, pooling_mode::average, ""},
                            pooling_test_params{CASE_POOLING_U8_1, 2, 3, pooling_mode::max, ""},
                            pooling_test_params{CASE_POOLING_U8_1, 2, 3, pooling_mode::average, ""},
                            pooling_test_params{CASE_POOLING_U8_2, 2, 3, pooling_mode::max, ""},
                            pooling_test_params{CASE_POOLING_U8_2, 2, 3, pooling_mode::average, ""},
                            pooling_test_params{CASE_POOLING_I8_1, 2, 3, pooling_mode::max, ""},
                            pooling_test_params{CASE_POOLING_I8_1, 2, 3, pooling_mode::average, ""},
                            pooling_test_params{CASE_POOLING_I8_2, 2, 3, pooling_mode::max, ""},
                            pooling_test_params{CASE_POOLING_I8_2, 2, 3, pooling_mode::average, ""},
                        }));

class pooling_f32_scale : public PoolingFusingTest {};
TEST_P(pooling_f32_scale, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / tensor{1, 1, 3, 3}.count())),
        pooling("pooling", "input", p.pool_mode, tensor{1, 1, 3, 3}, tensor{1}, tensor{0, 0, -1, -1, 0, 0}),
        scale("scale", "pooling", "scale_data"),
        reorder("output_reorder", "scale", format::bfyx, data_types::f32));

    tolerance = 1e-05f;
    execute(p);
}

TEST_P(pooling_f32_scale, fp16_scale_out) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / tensor{1, 1, 3, 3}.count())),
        pooling("pooling", "input", p.pool_mode, tensor{1, 1, 3, 3}, tensor{1}, tensor{0, 0, -1, -1, 0, 0}),
        scale("scale", "pooling", "scale_data", optional_data_type{data_types::f16}),
        reorder("output_reorder", "scale", format::bfyx, data_types::f32));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu,
                        pooling_f32_scale,
                        ::testing::ValuesIn(std::vector<pooling_test_params>{
                            pooling_test_params{CASE_POOLING_F32_1, 2, 3, pooling_mode::max, ""},
                            pooling_test_params{CASE_POOLING_F32_1, 2, 3, pooling_mode::average, ""},
                            pooling_test_params{CASE_POOLING_F16_1, 2, 3, pooling_mode::max, ""},
                            pooling_test_params{CASE_POOLING_F16_1, 2, 3, pooling_mode::average, ""},
                            pooling_test_params{CASE_POOLING_U8_1, 2, 3, pooling_mode::max, ""},
                            pooling_test_params{CASE_POOLING_U8_1, 2, 3, pooling_mode::average, ""},
                            pooling_test_params{CASE_POOLING_U8_2, 2, 3, pooling_mode::max, ""},
                            pooling_test_params{CASE_POOLING_U8_2, 2, 3, pooling_mode::average, ""},
                            pooling_test_params{CASE_POOLING_I8_1, 2, 3, pooling_mode::max, ""},
                            pooling_test_params{CASE_POOLING_I8_1, 2, 3, pooling_mode::average, ""},
                            pooling_test_params{CASE_POOLING_I8_2, 2, 3, pooling_mode::max, ""},
                            pooling_test_params{CASE_POOLING_I8_2, 2, 3, pooling_mode::average, ""},
                        }));

class pooling_scale_activation_quantize : public PoolingFusingTest {};
TEST_P(pooling_scale_activation_quantize, basic) {
    auto p = GetParam();

    create_topologies(input_layout("input", get_input_layout(p)),
                      data("in_lo", get_mem(get_single_element_layout(p), min_random, 0)),
                      data("in_hi", get_mem(get_single_element_layout(p), 1, max_random)),
                      data("out_lo", get_mem(get_single_element_layout(p), 0)),
                      data("out_hi", get_mem(get_single_element_layout(p), 255)),
                      data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / tensor{1, 1, 4, 4}.count())),
                      pooling("pooling", "input", "", p.pool_mode, tensor(1, 1, 4, 4), tensor(1, 1, 2, 2)),
                      scale("scale", "pooling", "scale_data"),
                      activation("activation", "scale", activation_func::relu),
                      quantize("quantize", "activation", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::u8),
                      reorder("output_reorder", "quantize", p.default_format, data_types::f32));

    tolerance = 1.0f;
    execute(p);
}

TEST_P(pooling_scale_activation_quantize, i8_output_data_type) {
    auto p = GetParam();

    create_topologies(input_layout("input", get_input_layout(p)),
                      data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                      data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                      data("out_lo", get_mem(get_single_element_layout(p), -127, 127)),
                      data("out_hi", get_mem(get_single_element_layout(p), -127, 127)),
                      data("scale_data",  get_mem(get_per_channel_layout(p), 1.0f / tensor{1, 1, 4, 4}.count())),
                      pooling("pooling", "input", "", p.pool_mode, tensor(1, 1, 4, 4), tensor(1, 1, 2, 2)),
                      scale("scale", "pooling", "scale_data"),
                      activation("activation", "scale", activation_func::relu),
                      quantize("quantize", "activation", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
                      reorder("output_reorder", "quantize", p.default_format, data_types::f32));

    tolerance = 1.0f;
    execute(p);
}

TEST_P(pooling_scale_activation_quantize, per_channel) {
    auto p = GetParam();

    create_topologies(input_layout("input", get_input_layout(p)),
                      data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                      data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                      data("out_lo", get_mem(get_single_element_layout(p), 0)),
                      data("out_hi", get_mem(get_single_element_layout(p), 255)),
                      data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / tensor{1, 1, 4, 4}.count())),
                      pooling("pooling", "input", "", p.pool_mode, tensor(1, 1, 4, 4), tensor(1, 1, 2, 2)),
                      scale("scale", "pooling", "scale_data"),
                      activation("activation", "scale", activation_func::atan),
                      quantize("quantize", "activation", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::u8),
                      reorder("output_reorder", "quantize", p.default_format, data_types::f32));

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu,
                         pooling_scale_activation_quantize,
                         ::testing::ValuesIn(std::vector<pooling_test_params>{
                            // Input type: FP32
                            pooling_test_params{CASE_POOLING_F32_3, 2, 5, pooling_mode::average, "pooling_gpu_bfyx_block_opt"},
                            pooling_test_params{CASE_POOLING_F32_3, 2, 5, pooling_mode::max, "pooling_gpu_bfyx_block_opt"},
                            pooling_test_params{CASE_POOLING_F32_3, 2, 5, pooling_mode::average, "pooling_gpu_ref"},
                            pooling_test_params{CASE_POOLING_F32_3, 2, 5, pooling_mode::max, "pooling_gpu_ref"},
                            pooling_test_params{CASE_POOLING_F32_4, 2, 5, pooling_mode::average, "pooling_gpu_fs_b_yx_fsv32"},
                            pooling_test_params{CASE_POOLING_F32_4, 2, 5, pooling_mode::max, "pooling_gpu_fs_b_yx_fsv32"},
                            pooling_test_params{CASE_POOLING_F32_5, 2, 5, pooling_mode::average, "pooling_gpu_byxf_padding_opt"},
                            pooling_test_params{CASE_POOLING_F32_5, 2, 5, pooling_mode::max, "pooling_gpu_byxf_padding_opt"},
                            pooling_test_params{CASE_POOLING_F32_6, 2, 5, pooling_mode::average, "pooling_gpu_byxf_opt"},
                            pooling_test_params{CASE_POOLING_F32_6, 2, 5, pooling_mode::max, "pooling_gpu_byxf_opt"},
                            pooling_test_params{CASE_POOLING_F32_7, 2, 5, pooling_mode::average, "pooling_gpu_bsv16_fsv16"},
                            pooling_test_params{CASE_POOLING_F32_7, 2, 5, pooling_mode::max, "pooling_gpu_bsv16_fsv16"},
                            pooling_test_params{CASE_POOLING_F32_8, 2, 5, pooling_mode::average, "pooling_gpu_blocked"},
                            pooling_test_params{CASE_POOLING_F32_8, 2, 5, pooling_mode::max, "pooling_gpu_blocked"},
                            pooling_test_params{CASE_POOLING_F32_9, 2, 5, pooling_mode::average, "pooling_gpu_ref"},
                            pooling_test_params{CASE_POOLING_F32_9, 2, 5, pooling_mode::max, "pooling_gpu_ref"},
                            pooling_test_params{CASE_POOLING_F32_10, 2, 5, pooling_mode::average, "pooling_gpu_bsv16_fsv16"},
                            pooling_test_params{CASE_POOLING_F32_10, 2, 5, pooling_mode::max, "pooling_gpu_bsv16_fsv16"},

                            // Input type: INT8
                            pooling_test_params{CASE_POOLING_I8_5, 2, 5, pooling_mode::average, "pooling_gpu_b_fs_yx_fsv4"},
                            pooling_test_params{CASE_POOLING_I8_5, 2, 5, pooling_mode::max, "pooling_gpu_b_fs_yx_fsv4"},
                            pooling_test_params{CASE_POOLING_I8_6, 2, 5, pooling_mode::average, "pooling_gpu_int8_ref"},
                            pooling_test_params{CASE_POOLING_I8_6, 2, 5, pooling_mode::max, "pooling_gpu_int8_ref"},

                            // Input type: UINT8
                            pooling_test_params{CASE_POOLING_U8_3, 2, 5, pooling_mode::average, "pooling_gpu_int8_ref"},
                            pooling_test_params{CASE_POOLING_U8_3, 2, 5, pooling_mode::max, "pooling_gpu_int8_ref"},
                            pooling_test_params{CASE_POOLING_U8_3, 2, 5, pooling_mode::average, "pooling_gpu_b_fs_yx_fsv4"},
                            pooling_test_params{CASE_POOLING_U8_3, 2, 5, pooling_mode::max, "pooling_gpu_b_fs_yx_fsv4"},
                            pooling_test_params{CASE_POOLING_U8_5, 2, 5, pooling_mode::average, "pooling_gpu_int8_ref"},
                            pooling_test_params{CASE_POOLING_U8_5, 2, 5, pooling_mode::max, "pooling_gpu_int8_ref"},
                            pooling_test_params{CASE_POOLING_U8_6, 2, 5, pooling_mode::average, "pooling_gpu_int8_ref"},
                            pooling_test_params{CASE_POOLING_U8_6, 2, 5, pooling_mode::max, "pooling_gpu_int8_ref"},
                        }));

INSTANTIATE_TEST_SUITE_P(DISABLED_fusings_gpu,
                         pooling_scale_activation_quantize,
                         ::testing::ValuesIn(std::vector<pooling_test_params>{
                            pooling_test_params{CASE_POOLING_F32_3, 2, 5, pooling_mode::average, "pooling_gpu_average_opt"},  //currently not enabled, fusing not upported
                        }));

class pooling_scale_activation : public PoolingFusingTest {};
TEST_P(pooling_scale_activation, basic) {
    auto p = GetParam();

    create_topologies(input_layout("input", get_input_layout(p)),
                      data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / tensor{1, 1, 4, 4}.count())),
                      pooling("pooling", "input", "", p.pool_mode, tensor(1, 1, 4, 4), tensor(1, 1, 2, 2)),
                      scale("scale", "pooling", "scale_data"),
                      activation("activation", "scale", activation_func::relu),
                      reorder("output_reorder", "activation", p.default_format, data_types::f32));

    tolerance = 1e-05f;
    execute(p);
}

TEST_P(pooling_scale_activation, eltwise_mul) {
    auto p = GetParam();

    create_topologies(input_layout("input", get_input_layout(p)),
                      data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / tensor{1, 1, 4, 4}.count())),
                      pooling("pooling", "input", "", p.pool_mode, tensor(1, 1, 4, 4), tensor(1, 1, 2, 2)),
                      eltwise("scale", {"pooling", "scale_data"}, eltwise_mode::prod, p.default_type),
                      activation("activation", "scale", activation_func::relu),
                      reorder("output_reorder", "activation", p.default_format, data_types::f32));

    tolerance = 1e-05f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu,
                        pooling_scale_activation,
                        ::testing::ValuesIn(std::vector<pooling_test_params>{
                            // Input type: F32
                            pooling_test_params{CASE_POOLING_F32_3, 2, 4, pooling_mode::average, "pooling_gpu_bfyx_block_opt"},
                            pooling_test_params{CASE_POOLING_F32_3, 2, 4, pooling_mode::max, "pooling_gpu_bfyx_block_opt"},
                            pooling_test_params{CASE_POOLING_F32_3, 2, 4, pooling_mode::average, "pooling_gpu_ref"},
                            pooling_test_params{CASE_POOLING_F32_3, 2, 4, pooling_mode::max, "pooling_gpu_ref"},
                            pooling_test_params{CASE_POOLING_F32_4, 2, 4, pooling_mode::average, "pooling_gpu_fs_b_yx_fsv32"},
                            pooling_test_params{CASE_POOLING_F32_4, 2, 4, pooling_mode::max, "pooling_gpu_fs_b_yx_fsv32"},
                            pooling_test_params{CASE_POOLING_F32_5, 2, 4, pooling_mode::average, "pooling_gpu_byxf_padding_opt"},
                            pooling_test_params{CASE_POOLING_F32_5, 2, 4, pooling_mode::max, "pooling_gpu_byxf_padding_opt"},
                            pooling_test_params{CASE_POOLING_F32_6, 2, 4, pooling_mode::average, "pooling_gpu_byxf_opt"},
                            pooling_test_params{CASE_POOLING_F32_6, 2, 4, pooling_mode::max, "pooling_gpu_byxf_opt"},
                            pooling_test_params{CASE_POOLING_F32_7, 2, 4, pooling_mode::average, "pooling_gpu_bsv16_fsv16"},
                            pooling_test_params{CASE_POOLING_F32_7, 2, 4, pooling_mode::max, "pooling_gpu_bsv16_fsv16"},
                            pooling_test_params{CASE_POOLING_F32_8, 2, 4, pooling_mode::average, "pooling_gpu_blocked"},
                            pooling_test_params{CASE_POOLING_F32_8, 2, 4, pooling_mode::max, "pooling_gpu_blocked"},
                            pooling_test_params{CASE_POOLING_F32_9, 2, 4, pooling_mode::average, "pooling_gpu_ref"},
                            pooling_test_params{CASE_POOLING_F32_9, 2, 4, pooling_mode::max, "pooling_gpu_ref"},
                            pooling_test_params{CASE_POOLING_F32_10, 2, 4, pooling_mode::average, "pooling_gpu_bsv16_fsv16"},
                            pooling_test_params{CASE_POOLING_F32_10, 2, 4, pooling_mode::max, "pooling_gpu_bsv16_fsv16"},

                            // Input type: INT8
                            pooling_test_params{CASE_POOLING_I8_5, 2, 4, pooling_mode::average, "pooling_gpu_b_fs_yx_fsv4"},
                            pooling_test_params{CASE_POOLING_I8_5, 2, 4, pooling_mode::max, "pooling_gpu_b_fs_yx_fsv4"},
                            pooling_test_params{CASE_POOLING_I8_6, 2, 4, pooling_mode::average, "pooling_gpu_int8_ref"},
                            pooling_test_params{CASE_POOLING_I8_6, 2, 4, pooling_mode::max, "pooling_gpu_int8_ref"},

                            // Input type: UINT8
                            pooling_test_params{CASE_POOLING_U8_3, 2, 4, pooling_mode::average, "pooling_gpu_int8_ref"},
                            pooling_test_params{CASE_POOLING_U8_3, 2, 4, pooling_mode::max, "pooling_gpu_int8_ref"},
                            pooling_test_params{CASE_POOLING_U8_3, 2, 4, pooling_mode::average, "pooling_gpu_b_fs_yx_fsv4"},
                            pooling_test_params{CASE_POOLING_U8_3, 2, 4, pooling_mode::max, "pooling_gpu_b_fs_yx_fsv4"},
                            pooling_test_params{CASE_POOLING_U8_5, 2, 4, pooling_mode::average, "pooling_gpu_int8_ref"},
                            pooling_test_params{CASE_POOLING_U8_5, 2, 4, pooling_mode::max, "pooling_gpu_int8_ref"},
                            pooling_test_params{CASE_POOLING_U8_6, 2, 4, pooling_mode::average, "pooling_gpu_int8_ref"},
                            pooling_test_params{CASE_POOLING_U8_6, 2, 4, pooling_mode::max, "pooling_gpu_int8_ref"},

                            // Input type: FP16  Output type: F32
                            pooling_test_params{CASE_POOLING_F16_3, 2, 4, pooling_mode::average, "pooling_gpu_bfyx_block_opt"},
                            pooling_test_params{CASE_POOLING_F16_3, 2, 4, pooling_mode::max, "pooling_gpu_bfyx_block_opt"},
                            pooling_test_params{CASE_POOLING_F16_3, 2, 4, pooling_mode::average, "pooling_gpu_ref"},
                            pooling_test_params{CASE_POOLING_F16_3, 2, 4, pooling_mode::max, "pooling_gpu_ref"},
                            pooling_test_params{CASE_POOLING_F16_4, 2, 4, pooling_mode::average, "pooling_gpu_fs_b_yx_fsv32"},
                            pooling_test_params{CASE_POOLING_F16_4, 2, 4, pooling_mode::max, "pooling_gpu_fs_b_yx_fsv32"},
                            pooling_test_params{CASE_POOLING_F16_5, 2, 4, pooling_mode::average, "pooling_gpu_byxf_padding_opt"},
                            pooling_test_params{CASE_POOLING_F16_5, 2, 4, pooling_mode::max, "pooling_gpu_byxf_padding_opt"},
                            pooling_test_params{CASE_POOLING_F16_6, 2, 4, pooling_mode::average, "pooling_gpu_byxf_opt"},
                            pooling_test_params{CASE_POOLING_F16_6, 2, 4, pooling_mode::max, "pooling_gpu_byxf_opt"},
                            pooling_test_params{CASE_POOLING_F16_7, 2, 4, pooling_mode::average, "pooling_gpu_bsv16_fsv16"},
                            pooling_test_params{CASE_POOLING_F16_7, 2, 4, pooling_mode::max, "pooling_gpu_bsv16_fsv16"},
                            pooling_test_params{CASE_POOLING_F16_8, 2, 4, pooling_mode::average, "pooling_gpu_blocked"},
                            pooling_test_params{CASE_POOLING_F16_8, 2, 4, pooling_mode::max, "pooling_gpu_blocked"},
                            pooling_test_params{CASE_POOLING_F16_9, 2, 4, pooling_mode::average, "pooling_gpu_ref"},
                            pooling_test_params{CASE_POOLING_F16_9, 2, 4, pooling_mode::max, "pooling_gpu_ref"},
                            pooling_test_params{CASE_POOLING_F16_10, 2, 4, pooling_mode::average, "pooling_gpu_bsv16_fsv16"},
                            pooling_test_params{CASE_POOLING_F16_10, 2, 4, pooling_mode::max, "pooling_gpu_bsv16_fsv16"},

                            // Input type: FP16
                            pooling_test_params{CASE_POOLING_F16_FP16_1, 2, 4, pooling_mode::average, "pooling_gpu_bfyx_block_opt"},
                            pooling_test_params{CASE_POOLING_F16_FP16_1, 2, 4, pooling_mode::max, "pooling_gpu_bfyx_block_opt"},
                            pooling_test_params{CASE_POOLING_F16_FP16_1, 2, 4, pooling_mode::average, "pooling_gpu_ref"},
                            pooling_test_params{CASE_POOLING_F16_FP16_1, 2, 4, pooling_mode::max, "pooling_gpu_ref"},
                            pooling_test_params{CASE_POOLING_F16_FP16_2, 2, 4, pooling_mode::average, "pooling_gpu_fs_b_yx_fsv32"},
                            pooling_test_params{CASE_POOLING_F16_FP16_2, 2, 4, pooling_mode::max, "pooling_gpu_fs_b_yx_fsv32"},
                            pooling_test_params{CASE_POOLING_F16_FP16_3, 2, 4, pooling_mode::average, "pooling_gpu_byxf_padding_opt"},
                            pooling_test_params{CASE_POOLING_F16_FP16_3, 2, 4, pooling_mode::max, "pooling_gpu_byxf_padding_opt"},
                            pooling_test_params{CASE_POOLING_F16_FP16_4, 2, 4, pooling_mode::average, "pooling_gpu_byxf_opt"},
                            pooling_test_params{CASE_POOLING_F16_FP16_4, 2, 4, pooling_mode::max, "pooling_gpu_byxf_opt"},
                            pooling_test_params{CASE_POOLING_F16_FP16_5, 2, 4, pooling_mode::average, "pooling_gpu_bsv16_fsv16"},
                            pooling_test_params{CASE_POOLING_F16_FP16_5, 2, 4, pooling_mode::max, "pooling_gpu_bsv16_fsv16"},
                            pooling_test_params{CASE_POOLING_F16_FP16_6, 2, 4, pooling_mode::average, "pooling_gpu_blocked"},
                            pooling_test_params{CASE_POOLING_F16_FP16_6, 2, 4, pooling_mode::max, "pooling_gpu_blocked"},
                            pooling_test_params{CASE_POOLING_F16_FP16_7, 2, 4, pooling_mode::average, "pooling_gpu_ref"},
                            pooling_test_params{CASE_POOLING_F16_FP16_7, 2, 4, pooling_mode::max, "pooling_gpu_ref"},
                            pooling_test_params{CASE_POOLING_F16_FP16_8, 2, 4, pooling_mode::average, "pooling_gpu_bsv16_fsv16"},
                            pooling_test_params{CASE_POOLING_F16_FP16_8, 2, 4, pooling_mode::max, "pooling_gpu_bsv16_fsv16"},

                            // Input type: FP32
                            pooling_test_params{CASE_POOLING_F32_F16_3, 2, 4, pooling_mode::average, "pooling_gpu_bfyx_block_opt"},
                            pooling_test_params{CASE_POOLING_F32_F16_3, 2, 4, pooling_mode::max, "pooling_gpu_bfyx_block_opt"},
                            pooling_test_params{CASE_POOLING_F32_F16_3, 2, 4, pooling_mode::average, "pooling_gpu_ref"},
                            pooling_test_params{CASE_POOLING_F32_F16_3, 2, 4, pooling_mode::max, "pooling_gpu_ref"},
                            pooling_test_params{CASE_POOLING_F32_F16_4, 2, 4, pooling_mode::average, "pooling_gpu_fs_b_yx_fsv32"},
                            pooling_test_params{CASE_POOLING_F32_F16_4, 2, 4, pooling_mode::max, "pooling_gpu_fs_b_yx_fsv32"},
                            pooling_test_params{CASE_POOLING_F32_F16_5, 2, 4, pooling_mode::average, "pooling_gpu_byxf_padding_opt"},
                            pooling_test_params{CASE_POOLING_F32_F16_5, 2, 4, pooling_mode::max, "pooling_gpu_byxf_padding_opt"},
                            pooling_test_params{CASE_POOLING_F32_F16_6, 2, 4, pooling_mode::average, "pooling_gpu_byxf_opt"},
                            pooling_test_params{CASE_POOLING_F32_F16_6, 2, 4, pooling_mode::max, "pooling_gpu_byxf_opt"},
                            pooling_test_params{CASE_POOLING_F32_F16_7, 2, 4, pooling_mode::average, "pooling_gpu_bsv16_fsv16"},
                            pooling_test_params{CASE_POOLING_F32_F16_7, 2, 4, pooling_mode::max, "pooling_gpu_bsv16_fsv16"},
                            pooling_test_params{CASE_POOLING_F32_F16_8, 2, 4, pooling_mode::average, "pooling_gpu_blocked"},
                            pooling_test_params{CASE_POOLING_F32_F16_8, 2, 4, pooling_mode::max, "pooling_gpu_blocked"},
                            pooling_test_params{CASE_POOLING_F32_F16_9, 2, 4, pooling_mode::average, "pooling_gpu_ref"},
                            pooling_test_params{CASE_POOLING_F32_F16_9, 2, 4, pooling_mode::max, "pooling_gpu_ref"},
                            pooling_test_params{CASE_POOLING_F32_F16_10, 2, 4, pooling_mode::average, "pooling_gpu_bsv16_fsv16"},
                            pooling_test_params{CASE_POOLING_F32_F16_10, 2, 4, pooling_mode::max, "pooling_gpu_bsv16_fsv16"},

                            // Input type: INT8
                            pooling_test_params{CASE_POOLING_I8_FP16_5, 2, 4, pooling_mode::average, "pooling_gpu_b_fs_yx_fsv4"},
                            pooling_test_params{CASE_POOLING_I8_FP16_5, 2, 4, pooling_mode::max, "pooling_gpu_b_fs_yx_fsv4"},
                            pooling_test_params{CASE_POOLING_I8_FP16_6, 2, 4, pooling_mode::average, "pooling_gpu_int8_ref"},
                            pooling_test_params{CASE_POOLING_I8_FP16_6, 2, 4, pooling_mode::max, "pooling_gpu_int8_ref"},

                            // Input type: UINT8
                            pooling_test_params{CASE_POOLING_U8_FP16_3, 2, 4, pooling_mode::max, "pooling_gpu_int8_ref"},
                            pooling_test_params{CASE_POOLING_U8_FP16_3, 2, 4, pooling_mode::average, "pooling_gpu_int8_ref"},
                            pooling_test_params{CASE_POOLING_U8_FP16_3, 2, 4, pooling_mode::average, "pooling_gpu_b_fs_yx_fsv4"},
                            pooling_test_params{CASE_POOLING_U8_FP16_3, 2, 4, pooling_mode::max, "pooling_gpu_b_fs_yx_fsv4"},
                            pooling_test_params{CASE_POOLING_U8_FP16_5, 2, 4, pooling_mode::average, "pooling_gpu_int8_ref"},
                            pooling_test_params{CASE_POOLING_U8_FP16_5, 2, 4, pooling_mode::max, "pooling_gpu_int8_ref"},
                            pooling_test_params{CASE_POOLING_U8_FP16_6, 2, 4, pooling_mode::average, "pooling_gpu_int8_ref"},
                            pooling_test_params{CASE_POOLING_U8_FP16_6, 2, 4, pooling_mode::max, "pooling_gpu_int8_ref"},
                     }));

/* ----------------------------------------------------------------------------------------------------- */
/* -------------------------------- DepthToSpace cases ------------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
struct depth_to_space_test_params {
    tensor input_size;
    tensor output_size;
    depth_to_space_mode mode;
    data_types input_type;
    format input_format;
    size_t block_size;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

#define CASE_DEPTH_TO_SPACE_F32_1 {1, 16, 8, 10}, {1, 4, 16, 20}, depth_to_space_mode::blocks_first, data_types::f32, format::bfyx, 2, data_types::f32, format::bfyx
#define CASE_DEPTH_TO_SPACE_F32_2 {1, 32, 8, 8},  {1, 2, 32, 32}, depth_to_space_mode::blocks_first, data_types::f32, format::b_fs_yx_fsv16, 4, data_types::f32, format::bfyx
#define CASE_DEPTH_TO_SPACE_F16_1 {1, 12, 8, 8},  {1, 3, 16, 16}, depth_to_space_mode::blocks_first, data_types::f16, format::bfyx, 2, data_types::f32, format::bfyx
#define CASE_DEPTH_TO_SPACE_F16_2 {1, 16, 9, 8},  {1, 1, 36, 32}, depth_to_space_mode::blocks_first, data_types::f16, format::b_fs_yx_fsv16, 4, data_types::f32, format::bfyx
#define CASE_DEPTH_TO_SPACE_U8_1  {1, 128, 8, 8}, {1, 2, 64, 64}, depth_to_space_mode::blocks_first, data_types::u8, format::bfyx, 8, data_types::f32, format::bfyx
#define CASE_DEPTH_TO_SPACE_U8_2  {1, 128, 4, 8}, {1, 8, 16, 32}, depth_to_space_mode::blocks_first, data_types::u8, format::b_fs_yx_fsv16, 4, data_types::f32, format::bfyx
#define CASE_DEPTH_TO_SPACE_I8_1  {1, 16, 8, 8},  {1, 4, 16, 16}, depth_to_space_mode::blocks_first, data_types::i8, format::bfyx, 2, data_types::f32, format::bfyx
#define CASE_DEPTH_TO_SPACE_I8_2  {1, 256, 8, 8}, {1, 4, 64, 64}, depth_to_space_mode::blocks_first, data_types::i8, format::b_fs_yx_fsv16, 8, data_types::f32, format::bfyx

class DepthToSpaceFusingsTest : public ::BaseFusingTest<depth_to_space_test_params> {
public:
    void execute(depth_to_space_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));

        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);

        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(depth_to_space_test_params& p) { return layout{p.input_type, p.input_format, p.input_size}; }

    layout get_per_channel_layout(depth_to_space_test_params& p) {
        return layout{p.default_type, p.default_format, tensor{1, p.output_size.feature[0], 1, 1}};
    }
    format get_input_format(depth_to_space_test_params &p) { return p.input_format; }
};

class depth_to_space_quantize_i8 : public DepthToSpaceFusingsTest {};
TEST_P(depth_to_space_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      depth_to_space("depth_to_space", "input", p.block_size, p.mode),
                      data("in_low", get_mem(get_per_channel_layout(p), min_random, 0)),
                      data("in_high", get_mem(get_per_channel_layout(p), 1, max_random)),
                      data("out_low", get_mem(get_single_element_layout(p), -128)),
                      data("out_high", get_mem(get_single_element_layout(p), 127)),
                      quantize("quant", "depth_to_space", "in_low", "in_high", "out_low", "out_high", 256, data_types::i8),
                      reorder("reorder_bfyx", "quant", format::bfyx, data_types::f32));

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(
    fusings_gpu,
    depth_to_space_quantize_i8,
    ::testing::ValuesIn(std::vector<depth_to_space_test_params>{
        depth_to_space_test_params{CASE_DEPTH_TO_SPACE_F32_1, 2, 3},
        depth_to_space_test_params{CASE_DEPTH_TO_SPACE_F32_2, 2, 3},
        depth_to_space_test_params{CASE_DEPTH_TO_SPACE_F16_1, 2, 3},
        depth_to_space_test_params{CASE_DEPTH_TO_SPACE_F16_2, 2, 3},
    }));

class depth_to_space_scale_act_eltwise_quantize_u8 : public DepthToSpaceFusingsTest {};
TEST_P(depth_to_space_scale_act_eltwise_quantize_u8, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      depth_to_space("depth_to_space", "input", p.block_size, p.mode),
                      data("scale1_data", get_mem(get_per_channel_layout(p), -0.125f)),
                      scale("scale1", "depth_to_space", "scale1_data"),
                      activation("actv1", "scale1", activation_func::relu),
                      data("eltw_data", get_mem(layout(p.default_type, p.input_format, p.output_size))),
                      eltwise("eltw", {"actv1", "eltw_data"}, eltwise_mode::sum, p.default_type),
                      data("in_low", get_mem(get_per_channel_layout(p), min_random, 0)),
                      data("in_high", get_mem(get_per_channel_layout(p), 1, max_random)),
                      data("out_low", get_mem(get_single_element_layout(p), 0)),
                      data("out_high", get_mem(get_single_element_layout(p), 255)),
                      quantize("quant", "eltw", "in_low", "in_high", "out_low", "out_high", 256, data_types::u8),
                      reorder("reorder_bfyx", "quant", format::bfyx, data_types::f32));

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(
    fusings_gpu,
    depth_to_space_scale_act_eltwise_quantize_u8,
    ::testing::ValuesIn(std::vector<depth_to_space_test_params>{
        depth_to_space_test_params{CASE_DEPTH_TO_SPACE_F32_1, 2, 6},
        depth_to_space_test_params{CASE_DEPTH_TO_SPACE_F32_2, 2, 6},
        depth_to_space_test_params{CASE_DEPTH_TO_SPACE_F16_1, 2, 6},
        depth_to_space_test_params{CASE_DEPTH_TO_SPACE_F16_2, 2, 6},
        depth_to_space_test_params{CASE_DEPTH_TO_SPACE_U8_1, 2, 6},
        depth_to_space_test_params{CASE_DEPTH_TO_SPACE_U8_2, 2, 6},
        depth_to_space_test_params{CASE_DEPTH_TO_SPACE_I8_1, 2, 6},
        depth_to_space_test_params{CASE_DEPTH_TO_SPACE_I8_2, 2, 6},
    }));


class depth_to_space_scale_act_eltw : public DepthToSpaceFusingsTest {};
TEST_P(depth_to_space_scale_act_eltw, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      depth_to_space("depth_to_space", "input", p.block_size, p.mode),
                      data("scale1_data", get_mem(get_per_channel_layout(p), -0.125f)),
                      scale("scale1", "depth_to_space", "scale1_data"),
                      activation("actv1", "scale1", activation_func::relu),
                      data("eltw_data", get_mem(layout(p.default_type, p.input_format, p.output_size))),
                      eltwise("eltw", {"actv1", "eltw_data"}, eltwise_mode::sum, p.default_type),
                      reorder("reorder_bfyx", "eltw", format::bfyx, data_types::f32));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(
    fusings_gpu,
    depth_to_space_scale_act_eltw,
    ::testing::ValuesIn(std::vector<depth_to_space_test_params>{
        depth_to_space_test_params{CASE_DEPTH_TO_SPACE_F32_1, 2, 5},
        depth_to_space_test_params{CASE_DEPTH_TO_SPACE_F32_2, 2, 5},
        depth_to_space_test_params{CASE_DEPTH_TO_SPACE_F16_1, 2, 5},
        depth_to_space_test_params{CASE_DEPTH_TO_SPACE_F16_2, 2, 5},
        depth_to_space_test_params{CASE_DEPTH_TO_SPACE_U8_1, 2, 5},
        depth_to_space_test_params{CASE_DEPTH_TO_SPACE_U8_2, 2, 5},
        depth_to_space_test_params{CASE_DEPTH_TO_SPACE_I8_1, 2, 5},
        depth_to_space_test_params{CASE_DEPTH_TO_SPACE_I8_2, 2, 5},
    }));

/* ----------------------------------------------------------------------------------------------------- */
/* -------------------------------- SpaceToDepth cases ------------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
struct space_to_depth_params {
    tensor input_size;
    tensor output_size;
    space_to_depth::depth_mode mode;
    data_types input_type;
    format input_format;
    size_t block_size;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

#define CASE_SPACE_TO_DEPTH_F32_1 {2, 2, 8, 10}, {2, 8, 4, 5}, space_to_depth::depth_mode::blocks_first, data_types::f32, format::bfyx, 2, data_types::f32, format::bfyx
#define CASE_SPACE_TO_DEPTH_F32_2 {1, 2, 6, 6, 6},  {1, 54, 2, 2, 2}, space_to_depth::depth_mode::depth_first, data_types::f32, format::bfzyx, 3, data_types::f32, format::bfyx
#define CASE_SPACE_TO_DEPTH_F16_1 {1, 3, 6, 6},  {1, 12, 3, 3}, space_to_depth::depth_mode::blocks_first, data_types::f16, format::bfyx, 2, data_types::f32, format::bfyx
#define CASE_SPACE_TO_DEPTH_F16_2 {2, 1, 3, 3},  {2, 9, 1, 1}, space_to_depth::depth_mode::blocks_first, data_types::f16, format::b_fs_yx_fsv16, 3, data_types::f32, format::bfyx
#define CASE_SPACE_TO_DEPTH_U8_1  {2, 2, 8, 10}, {2, 8, 4, 5}, space_to_depth::depth_mode::blocks_first, data_types::u8, format::bfyx, 2, data_types::f32, format::bfyx
#define CASE_SPACE_TO_DEPTH_U8_2  {1, 2, 6, 6, 6},  {1, 54, 2, 2, 2}, space_to_depth::depth_mode::depth_first, data_types::u8, format::bfzyx, 3, data_types::f32, format::bfyx
#define CASE_SPACE_TO_DEPTH_I8_1  {1, 3, 6, 6},  {1, 12, 3, 3}, space_to_depth::depth_mode::blocks_first, data_types::i8, format::bfyx, 2, data_types::f32, format::bfyx
#define CASE_SPACE_TO_DEPTH_I8_2  {2, 1, 3, 3},  {2, 9, 1, 1}, space_to_depth::depth_mode::blocks_first, data_types::i8, format::b_fs_yx_fsv16, 3, data_types::f32, format::bfyx

class SpaceToDepthFusingsTest : public ::BaseFusingTest<space_to_depth_params> {
public:
    void execute(space_to_depth_params& p) {
        auto input_prim = get_mem(get_input_layout(p));

        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);

        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(space_to_depth_params& p) { return layout{p.input_type, p.input_format, p.input_size}; }

    layout get_per_channel_layout(space_to_depth_params& p) {
        return layout{p.default_type, p.default_format, tensor{1, p.output_size.feature[0], 1, 1}};
    }
    format get_input_format(space_to_depth_params &p) { return p.input_format; }
};

class space_to_depth_quantize_i8 : public SpaceToDepthFusingsTest {};
TEST_P(space_to_depth_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      space_to_depth("space_to_depth", "input", p.mode, p.block_size),
                      data("in_low", get_mem(get_per_channel_layout(p), min_random, 0)),
                      data("in_high", get_mem(get_per_channel_layout(p), 1, max_random)),
                      data("out_low", get_mem(get_single_element_layout(p), -128)),
                      data("out_high", get_mem(get_single_element_layout(p), 127)),
                      quantize("quant", "space_to_depth", "in_low", "in_high", "out_low", "out_high", 256, data_types::i8),
                      reorder("reorder_bfyx", "quant", format::bfyx, data_types::f32));

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(
    fusings_gpu,
    space_to_depth_quantize_i8,
    ::testing::ValuesIn(std::vector<space_to_depth_params>{
        space_to_depth_params{CASE_SPACE_TO_DEPTH_F32_1, 2, 3},
        space_to_depth_params{CASE_SPACE_TO_DEPTH_F32_2, 2, 3},
        space_to_depth_params{CASE_SPACE_TO_DEPTH_F16_1, 2, 3},
        space_to_depth_params{CASE_SPACE_TO_DEPTH_F16_2, 2, 3},
    }));

class space_to_depth_scale_act_eltwise_quantize_u8 : public SpaceToDepthFusingsTest {};
TEST_P(space_to_depth_scale_act_eltwise_quantize_u8, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      space_to_depth("space_to_depth", "input", p.mode, p.block_size),
                      data("scale1_data", get_mem(get_per_channel_layout(p), -0.125f)),
                      scale("scale1", "space_to_depth", "scale1_data"),
                      activation("actv1", "scale1", activation_func::relu),
                      data("eltw_data", get_mem(layout(p.default_type, p.input_format, p.output_size))),
                      eltwise("eltw", {"actv1", "eltw_data"}, eltwise_mode::sum, p.default_type),
                      data("in_low", get_mem(get_per_channel_layout(p), min_random, 0)),
                      data("in_high", get_mem(get_per_channel_layout(p), 1, max_random)),
                      data("out_low", get_mem(get_single_element_layout(p), 0)),
                      data("out_high", get_mem(get_single_element_layout(p), 255)),
                      quantize("quant", "eltw", "in_low", "in_high", "out_low", "out_high", 256, data_types::u8),
                      reorder("reorder_bfyx", "quant", format::bfyx, data_types::f32));

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(
    fusings_gpu,
    space_to_depth_scale_act_eltwise_quantize_u8,
    ::testing::ValuesIn(std::vector<space_to_depth_params>{
        space_to_depth_params{CASE_SPACE_TO_DEPTH_F32_1, 2, 6},
        space_to_depth_params{CASE_SPACE_TO_DEPTH_F32_2, 2, 6},
        space_to_depth_params{CASE_SPACE_TO_DEPTH_F16_1, 2, 6},
        space_to_depth_params{CASE_SPACE_TO_DEPTH_F16_2, 2, 6},
        space_to_depth_params{CASE_SPACE_TO_DEPTH_U8_1, 2, 6},
        space_to_depth_params{CASE_SPACE_TO_DEPTH_U8_2, 2, 6},
        space_to_depth_params{CASE_SPACE_TO_DEPTH_I8_1, 2, 6},
        space_to_depth_params{CASE_SPACE_TO_DEPTH_I8_2, 2, 6},
    }));


class space_to_depth_scale_act_eltw : public SpaceToDepthFusingsTest {};
TEST_P(space_to_depth_scale_act_eltw, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      space_to_depth("space_to_depth", "input", p.mode, p.block_size),
                      data("scale1_data", get_mem(get_per_channel_layout(p), -0.125f)),
                      scale("scale1", "space_to_depth", "scale1_data"),
                      activation("actv1", "scale1", activation_func::relu),
                      data("eltw_data", get_mem(layout(p.default_type, p.input_format, p.output_size))),
                      eltwise("eltw", {"actv1", "eltw_data"}, eltwise_mode::sum, p.default_type),
                      reorder("reorder_bfyx", "eltw", format::bfyx, data_types::f32));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(
    fusings_gpu,
    space_to_depth_scale_act_eltw,
    ::testing::ValuesIn(std::vector<space_to_depth_params>{
        space_to_depth_params{CASE_SPACE_TO_DEPTH_F32_1, 2, 5},
        space_to_depth_params{CASE_SPACE_TO_DEPTH_F32_2, 2, 5},
        space_to_depth_params{CASE_SPACE_TO_DEPTH_F16_1, 2, 5},
        space_to_depth_params{CASE_SPACE_TO_DEPTH_F16_2, 2, 5},
        space_to_depth_params{CASE_SPACE_TO_DEPTH_U8_1, 2, 5},
        space_to_depth_params{CASE_SPACE_TO_DEPTH_U8_2, 2, 5},
        space_to_depth_params{CASE_SPACE_TO_DEPTH_I8_1, 2, 5},
        space_to_depth_params{CASE_SPACE_TO_DEPTH_I8_2, 2, 5},
    }));

/* ----------------------------------------------------------------------------------------------------- */
/* ------------------------------------------ Gather cases --------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
struct gather_test_params {
    tensor dictionary_shape;
    tensor indices_shape;
    tensor out_shape;
    format out_format;
    cldnn::gather::gather_axis axis;
    data_types data_type;
    format input_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

#define CASE_GATHER_FP32_1 {2, 3, 1, 4}, {4, 1, 1, 1}, {4, 3, 1, 4}, format::bfyx, cldnn::gather::gather_axis::along_b, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GATHER_FP32_2 {3, 2, 1, 2}, {2, 3, 1, 1}, {2, 3, 2, 2}, format::bfyx, cldnn::gather::gather_axis::along_b, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GATHER_FP32_3 {3, 1, 1, 2}, {2, 1, 1, 1}, {3, 2, 1, 2}, format::bfyx, cldnn::gather::gather_axis::along_f, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GATHER_FP32_4 {5, 3, 2, 2}, {3, 1, 1, 1}, {5, 2, 2, 3}, format::bfyx, cldnn::gather::gather_axis::along_y, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GATHER_FP32_5 {2, 3, 1, 2}, {1, 3, 1, 1}, {2, 3, 3, 1}, format::bfyx, cldnn::gather::gather_axis::along_y, data_types::f32, format::bfyx, data_types::f32, format::bfyx

#define CASE_GATHER_FP16_1 {2, 3, 1, 4}, {4, 1, 1, 1}, {4, 3, 1, 4}, format::bfyx, cldnn::gather::gather_axis::along_b, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GATHER_FP16_2 {3, 2, 1, 2}, {2, 3, 1, 1}, {2, 3, 2, 2}, format::bfyx, cldnn::gather::gather_axis::along_b, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GATHER_FP16_3 {3, 1, 1, 2}, {2, 1, 1, 1}, {3, 2, 1, 2}, format::bfyx, cldnn::gather::gather_axis::along_f, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GATHER_FP16_4 {5, 3, 2, 2}, {3, 1, 1, 1}, {5, 2, 2, 3}, format::bfyx, cldnn::gather::gather_axis::along_y, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GATHER_FP16_5 {2, 3, 1, 2}, {1, 3, 1, 1}, {2, 3, 3, 1}, format::bfyx, cldnn::gather::gather_axis::along_y, data_types::f16, format::bfyx, data_types::f16, format::bfyx

#define CASE_GATHER_5D_FP32_1 {2, 3, 1, 4, 1}, {4, 1, 1, 1}, {4, 3, 1, 4, 1}, format::bfzyx, cldnn::gather::gather_axis::along_b, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_GATHER_5D_FP32_2 {2, 3, 2, 2, 2}, {2, 1, 1, 1}, {2, 2, 2, 2, 2}, format::bfzyx, cldnn::gather::gather_axis::along_f, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_GATHER_5D_FP32_3 {5, 3, 2, 2, 2}, {3, 1, 1, 1}, {5, 3, 2, 3, 2}, format::bfzyx, cldnn::gather::gather_axis::along_y, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_GATHER_5D_FP32_4 {2, 3, 1, 4, 4}, {2, 1, 1, 1}, {2, 3, 1, 4, 2}, format::bfzyx, cldnn::gather::gather_axis::along_z, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_GATHER_5D_FP32_5 {3, 1, 5, 2, 1}, {2, 1, 1, 1}, {3, 1, 2, 2, 1}, format::bfzyx, cldnn::gather::gather_axis::along_x, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx

#define CASE_GATHER_5D_FP16_1 {3, 2, 1, 2, 1}, {2, 1, 1, 1}, {2, 2, 2, 2, 1}, format::bfzyx, cldnn::gather::gather_axis::along_b, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_GATHER_5D_FP16_2 {1, 3, 1, 2, 1}, {2, 1, 1, 1}, {1, 2, 1, 2, 1}, format::bfzyx, cldnn::gather::gather_axis::along_f, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_GATHER_5D_FP16_3 {2, 3, 1, 3, 3}, {1, 2, 1, 1}, {2, 3, 1, 2, 3}, format::bfzyx, cldnn::gather::gather_axis::along_y, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_GATHER_5D_FP16_4 {3, 2, 2, 2, 2}, {2, 1, 1, 1}, {3, 2, 2, 2, 2}, format::bfzyx, cldnn::gather::gather_axis::along_z, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_GATHER_5D_FP16_5 {1, 1, 2, 1, 1}, {3, 1, 1, 1}, {1, 1, 3, 1, 1}, format::bfzyx, cldnn::gather::gather_axis::along_x, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx

class GatherPrimitiveFusingTest : public ::BaseFusingTest<gather_test_params> {
public:
    void execute(gather_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(gather_test_params& p) {
        return layout{ p.data_type, p.input_format, p.dictionary_shape };
    }

    layout get_indices_layout(gather_test_params& p) {
        return layout{ p.data_type, format::bfyx, p.indices_shape };
    }

    size_t get_axis_dim(gather_test_params& p) {
        switch (p.axis) {
            case cldnn::gather::gather_axis::along_x:
                return p.dictionary_shape.spatial[0];
            case cldnn::gather::gather_axis::along_y:
                return p.dictionary_shape.spatial[1];
            case cldnn::gather::gather_axis::along_z:
                return p.dictionary_shape.spatial[2];
            case cldnn::gather::gather_axis::along_w:
                return p.dictionary_shape.spatial[3];
            case cldnn::gather::gather_axis::along_f:
                return p.dictionary_shape.feature[0];
            case cldnn::gather::gather_axis::along_b:
                return p.dictionary_shape.batch[0];
            default:
                return 1;
        }
    }

    layout get_per_channel_layout(gather_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{1, p.out_shape.feature[0], 1, 1} };
    }
};

class gather_quantize : public GatherPrimitiveFusingTest {};
TEST_P(gather_quantize, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
        data("gather_indices", get_mem(get_indices_layout(p), 0, static_cast<int>(get_axis_dim(p)))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        gather("gather_prim", "input", "gather_indices", p.axis, p.out_format, p.out_shape),
        quantize("quantize", "gather_prim", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gather_quantize,
    ::testing::ValuesIn(std::vector<gather_test_params>{
                        gather_test_params{ CASE_GATHER_FP32_1, 2, 3 },
                        gather_test_params{ CASE_GATHER_FP32_2, 2, 3 },
                        gather_test_params{ CASE_GATHER_FP32_3, 2, 3 },
                        gather_test_params{ CASE_GATHER_FP32_4, 2, 3 },
                        gather_test_params{ CASE_GATHER_FP32_5, 2, 3 },

                        gather_test_params{ CASE_GATHER_FP16_1, 2, 3 },
                        gather_test_params{ CASE_GATHER_FP16_2, 2, 3 },
                        gather_test_params{ CASE_GATHER_FP16_3, 2, 3 },
                        gather_test_params{ CASE_GATHER_FP16_4, 2, 3 },
                        gather_test_params{ CASE_GATHER_FP16_5, 2, 3 },

                        gather_test_params{ CASE_GATHER_5D_FP32_1, 2, 3 },
                        gather_test_params{ CASE_GATHER_5D_FP32_2, 2, 3 },
                        gather_test_params{ CASE_GATHER_5D_FP32_3, 2, 3 },
                        gather_test_params{ CASE_GATHER_5D_FP32_4, 2, 3 },
                        gather_test_params{ CASE_GATHER_5D_FP32_5, 2, 3 },

                        gather_test_params{ CASE_GATHER_5D_FP16_1, 2, 3 },
                        gather_test_params{ CASE_GATHER_5D_FP16_2, 2, 3 },
                        gather_test_params{ CASE_GATHER_5D_FP16_3, 2, 3 },
                        gather_test_params{ CASE_GATHER_5D_FP16_4, 2, 3 },
                        gather_test_params{ CASE_GATHER_5D_FP16_5, 2, 3 },
}));

class gather_scale_activation : public GatherPrimitiveFusingTest {};
TEST_P(gather_scale_activation, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
        data("gather_indices", get_mem(get_indices_layout(p), 0, static_cast<int>(get_axis_dim(p)))),
        data("scale_data", get_mem(get_per_channel_layout(p), -10, 10)),
        gather("gather_prim", "input", "gather_indices", p.axis, p.out_format, p.out_shape),
        activation("activation", "gather_prim", activation_func::abs),
        scale("scale", "activation", "scale_data"),
        reorder("reorder_bfyx", "scale", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gather_scale_activation,
    ::testing::ValuesIn(std::vector<gather_test_params>{
                        gather_test_params{ CASE_GATHER_FP32_1, 2, 4 },
                        gather_test_params{ CASE_GATHER_FP32_2, 2, 4 },
                        gather_test_params{ CASE_GATHER_FP32_3, 2, 4 },
                        gather_test_params{ CASE_GATHER_FP32_4, 2, 4 },
                        gather_test_params{ CASE_GATHER_FP32_5, 2, 4 },

                        gather_test_params{ CASE_GATHER_FP16_1, 2, 4 },
                        gather_test_params{ CASE_GATHER_FP16_2, 2, 4 },
                        gather_test_params{ CASE_GATHER_FP16_3, 2, 4 },
                        gather_test_params{ CASE_GATHER_FP16_4, 2, 4 },
                        gather_test_params{ CASE_GATHER_FP16_5, 2, 4 },

                        gather_test_params{ CASE_GATHER_5D_FP32_1, 2, 4 },
                        gather_test_params{ CASE_GATHER_5D_FP32_2, 2, 4 },
                        gather_test_params{ CASE_GATHER_5D_FP32_3, 2, 4 },
                        gather_test_params{ CASE_GATHER_5D_FP32_4, 2, 4 },
                        gather_test_params{ CASE_GATHER_5D_FP32_5, 2, 4 },

                        gather_test_params{ CASE_GATHER_5D_FP16_1, 2, 4 },
                        gather_test_params{ CASE_GATHER_5D_FP16_2, 2, 4 },
                        gather_test_params{ CASE_GATHER_5D_FP16_3, 2, 4 },
                        gather_test_params{ CASE_GATHER_5D_FP16_4, 2, 4 },
                        gather_test_params{ CASE_GATHER_5D_FP16_5, 2, 4 },
}));


/* ----------------------------------------------------------------------------------------------------- */
/* ------------------------------------------ ScatterUpdate cases --------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

struct scatter_update_test_params {
    tensor dictionary_shape;
    tensor indices_shape;
    tensor updates_shape;
    cldnn::scatter_update::scatter_update_axis axis;
    data_types data_type;
    format input_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

#define CASE_SCATTER_UPDATE_FP32_1 {2, 4, 1, 1}, {2, 1, 1, 1}, {2, 4, 1, 1}, cldnn::scatter_update::scatter_update_axis::along_b, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_SCATTER_UPDATE_FP32_2 {8, 1, 1, 1}, {4, 1, 1, 1}, {4, 1, 1, 1}, cldnn::scatter_update::scatter_update_axis::along_b, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_SCATTER_UPDATE_FP32_3 {4, 3, 1, 1}, {2, 2, 1, 1}, {2, 2, 1, 3}, cldnn::scatter_update::scatter_update_axis::along_b, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_SCATTER_UPDATE_FP32_4 {2, 5, 1, 2}, {2, 2, 1, 1}, {2, 2, 2, 2}, cldnn::scatter_update::scatter_update_axis::along_f, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_SCATTER_UPDATE_FP32_5 {2, 2, 1, 4}, {2, 2, 1, 1}, {2, 2, 2, 2}, cldnn::scatter_update::scatter_update_axis::along_y, data_types::f32, format::bfyx, data_types::f32, format::bfyx

#define CASE_SCATTER_UPDATE_FP16_1 {2, 4, 1, 1}, {1, 1, 1, 2}, {2, 1, 2, 1}, cldnn::scatter_update::scatter_update_axis::along_f, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_SCATTER_UPDATE_FP16_2 {8, 2, 1, 20}, {2, 3, 1, 1}, {2, 3, 20, 2}, cldnn::scatter_update::scatter_update_axis::along_b, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_SCATTER_UPDATE_FP16_3 {2, 2, 4, 1}, {3, 1, 1, 1}, {2, 2, 3, 1}, cldnn::scatter_update::scatter_update_axis::along_x, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_SCATTER_UPDATE_FP16_4 {6, 2, 1, 1}, {1, 2, 1, 2}, {1, 2, 2, 2}, cldnn::scatter_update::scatter_update_axis::along_b, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_SCATTER_UPDATE_FP16_5 {3, 1, 1, 5}, {2, 2, 1, 1}, {3, 1, 2, 2}, cldnn::scatter_update::scatter_update_axis::along_y, data_types::f16, format::bfyx, data_types::f16, format::bfyx

#define CASE_SCATTER_UPDATE_5D_FP32_1 {4, 3, 1, 4, 1}, {4, 1, 1, 1}, {4, 3, 1, 4, 1}, cldnn::scatter_update::scatter_update_axis::along_b, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_SCATTER_UPDATE_5D_FP32_2 {2, 3, 2, 2, 2}, {2, 1, 1, 1}, {2, 2, 2, 2, 2}, cldnn::scatter_update::scatter_update_axis::along_f, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_SCATTER_UPDATE_5D_FP32_3 {5, 3, 2, 4, 2}, {3, 1, 1, 1}, {5, 3, 2, 3, 2}, cldnn::scatter_update::scatter_update_axis::along_y, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_SCATTER_UPDATE_5D_FP32_4 {2, 3, 1, 4, 4}, {2, 1, 1, 1}, {2, 3, 1, 4, 2}, cldnn::scatter_update::scatter_update_axis::along_z, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_SCATTER_UPDATE_5D_FP32_5 {3, 1, 5, 2, 1}, {2, 1, 1, 1}, {3, 1, 2, 2, 1}, cldnn::scatter_update::scatter_update_axis::along_x, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx

#define CASE_SCATTER_UPDATE_5D_FP16_1 {3, 2, 1, 2, 1}, {2, 1, 1, 1}, {2, 2, 2, 2, 1}, cldnn::scatter_update::scatter_update_axis::along_b, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_SCATTER_UPDATE_5D_FP16_2 {1, 3, 1, 2, 1}, {2, 1, 1, 1}, {1, 2, 1, 2, 1}, cldnn::scatter_update::scatter_update_axis::along_f, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_SCATTER_UPDATE_5D_FP16_3 {2, 3, 1, 3, 3}, {1, 2, 1, 1}, {2, 3, 1, 2, 3}, cldnn::scatter_update::scatter_update_axis::along_y, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_SCATTER_UPDATE_5D_FP16_4 {3, 2, 2, 2, 2}, {2, 1, 1, 1}, {3, 2, 2, 2, 2}, cldnn::scatter_update::scatter_update_axis::along_z, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_SCATTER_UPDATE_5D_FP16_5 {1, 1, 4, 1, 1}, {3, 1, 1, 1}, {1, 1, 3, 1, 1}, cldnn::scatter_update::scatter_update_axis::along_x, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx

class ScatterUpdatePrimitiveFusingTest : public ::BaseFusingTest<scatter_update_test_params> {
public:
    void execute(scatter_update_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);
        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(scatter_update_test_params& p) {
        return layout{ p.data_type, p.input_format, p.dictionary_shape };
    }

    layout get_indices_layout(scatter_update_test_params& p) {
        return layout{ p.data_type, format::bfyx, p.indices_shape };
    }

    layout get_updates_layout(scatter_update_test_params& p) {
        return layout{ p.data_type, p.input_format, p.updates_shape };
    }

    size_t get_axis_dim(scatter_update_test_params& p) {
        switch (p.axis) {
            case cldnn::scatter_update::scatter_update_axis::along_x:
                return p.dictionary_shape.spatial[0];
            case cldnn::scatter_update::scatter_update_axis::along_y:
                return p.dictionary_shape.spatial[1];
            case cldnn::scatter_update::scatter_update_axis::along_z:
                return p.dictionary_shape.spatial[2];
            case cldnn::scatter_update::scatter_update_axis::along_w:
                return p.dictionary_shape.spatial[3];
            case cldnn::scatter_update::scatter_update_axis::along_f:
                return p.dictionary_shape.feature[0];
            case cldnn::scatter_update::scatter_update_axis::along_b:
                return p.dictionary_shape.batch[0];
            default:
                return 1;
        }
    }

    layout get_per_channel_layout(scatter_update_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{1, p.dictionary_shape.feature[0], 1, 1} };
    }
};

class scatter_update_quantize : public ScatterUpdatePrimitiveFusingTest {};
TEST_P(scatter_update_quantize, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
        data("scatter_update_indices", get_repeatless_mem(get_indices_layout(p), 0, static_cast<int>(get_axis_dim(p)) - 1)),
        data("scatter_update_updates", get_mem(get_updates_layout(p), 0, 1000)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        scatter_update("scatter_update_prim", "input", "scatter_update_indices", "scatter_update_updates", p.axis),
        quantize("quantize", "scatter_update_prim", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );
    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, scatter_update_quantize,
    ::testing::ValuesIn(std::vector<scatter_update_test_params>{
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_1, 2, 3 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_2, 2, 3 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_3, 2, 3 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_4, 2, 3 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_5, 2, 3 },

                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_1, 2, 3 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_2, 2, 3 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_3, 2, 3 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_4, 2, 3 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_5, 2, 3 },


                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_1, 2, 3 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_2, 2, 3 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_3, 2, 3 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_4, 2, 3 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_5, 2, 3 },

                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_1, 2, 3 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_2, 2, 3 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_3, 2, 3 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_4, 2, 3 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_5, 2, 3 },
}));

class scatter_update_scale_activation : public ScatterUpdatePrimitiveFusingTest {};
TEST_P(scatter_update_scale_activation, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
        data("scatter_update_indices", get_repeatless_mem(get_indices_layout(p), 0, static_cast<int>(get_axis_dim(p)) - 1)),
        data("scatter_update_updates", get_mem(get_updates_layout(p), 0, 1000)),
        data("scale_data", get_mem(get_per_channel_layout(p), -10, 10)),
        scatter_update("scatter_update_prim", "input", "scatter_update_indices", "scatter_update_updates", p.axis),
        activation("activation", "scatter_update_prim", activation_func::abs),
        scale("scale", "activation", "scale_data"),
        reorder("reorder_bfyx", "scale", p.default_format, data_types::f32)
    );
    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, scatter_update_scale_activation,
    ::testing::ValuesIn(std::vector<scatter_update_test_params>{
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_1, 2, 4 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_2, 2, 4 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_3, 2, 4 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_4, 2, 4 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_5, 2, 4 },

                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_1, 2, 4 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_2, 2, 4 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_3, 2, 4 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_4, 2, 4 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_5, 2, 4 },


                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_1, 2, 4 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_2, 2, 4 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_3, 2, 4 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_4, 2, 4 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_5, 2, 4 },

                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_1, 2, 4 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_2, 2, 4 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_3, 2, 4 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_4, 2, 4 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_5, 2, 4 },
}));

class scatter_update_scale_activation_eltwise : public ScatterUpdatePrimitiveFusingTest {};
TEST_P(scatter_update_scale_activation_eltwise, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
        data("scatter_update_indices", get_repeatless_mem(get_indices_layout(p), 0, static_cast<int>(get_axis_dim(p)) - 1)),
        data("scatter_update_updates", get_mem(get_updates_layout(p), 0, 1000)),
        data("scale_data", get_mem(get_per_channel_layout(p), -10, 10)),
        data("eltw_data", get_mem(layout(p.default_type, p.default_format, p.dictionary_shape))),
        scatter_update("scatter_update_prim", "input", "scatter_update_indices", "scatter_update_updates", p.axis),
        activation("activation", "scatter_update_prim", activation_func::abs),
        eltwise("eltw", {"activation", "eltw_data"}, eltwise_mode::sum, p.default_type),
        scale("scale", "eltw", "scale_data"),
        reorder("reorder_bfyx", "scale", p.default_format, data_types::f32)
    );
    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, scatter_update_scale_activation_eltwise,
                        ::testing::ValuesIn(std::vector<scatter_update_test_params> {
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_1, 3, 5 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_2, 3, 5 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_3, 3, 5 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_4, 3, 5 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP32_5, 3, 5 },

                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_1, 3, 5 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_2, 3, 5 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_3, 3, 5 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_4, 3, 5 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_FP16_5, 3, 5 },


                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_1, 3, 5 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_2, 3, 5 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_3, 3, 5 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_4, 3, 5 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP32_5, 3, 5 },

                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_1, 3, 5 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_2, 3, 5 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_3, 3, 5 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_4, 3, 5 },
                        scatter_update_test_params{ CASE_SCATTER_UPDATE_5D_FP16_5, 3, 5 },

}));

/* ----------------------------------------------------------------------------------------------------- */
/* ------------------------------------------ ScatterElementsUpdate cases --------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

struct scatter_elements_update_test_params {
    tensor input_shape;
    tensor indices_shape;
    cldnn::scatter_elements_update::scatter_elements_update_axis axis;
    data_types data_type;
    format input_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

// input shape along the update axis should be larger than the total number of elements in the update tensor.
// This is not a limitation of operation itself, but a limitation of test implementation.
#define CASE_SCATTER_ELEMENTS_UPDATE_FP32_1 {8, 4, 1, 1}, {2, 4, 1, 1}, cldnn::scatter_elements_update::scatter_elements_update_axis::along_b, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ELEMENTS_UPDATE_FP32_2 {2, 8, 1, 2}, {2, 2, 1, 2}, cldnn::scatter_elements_update::scatter_elements_update_axis::along_f, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ELEMENTS_UPDATE_FP32_3 {2, 3, 10, 10}, {2, 2, 1, 2}, cldnn::scatter_elements_update::scatter_elements_update_axis::along_y, data_types::f32, format::bfyx, data_types::f32, format::bfyx

#define CASE_SCATTER_ELEMENTS_UPDATE_FP16_1 {2, 2, 14, 12}, {2, 2, 3, 1}, cldnn::scatter_elements_update::scatter_elements_update_axis::along_x, data_types::f16, format::bfyx, data_types::f16, format::bfyx

#define CASE_SCATTER_ELEMENTS_UPDATE_5D_FP32_1 {24, 3, 1, 4, 1}, {4, 3, 1, 2, 1}, cldnn::scatter_elements_update::scatter_elements_update_axis::along_b, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_SCATTER_ELEMENTS_UPDATE_5D_FP32_2 {2, 17, 2, 2, 2}, {1, 2, 2, 2, 2}, cldnn::scatter_elements_update::scatter_elements_update_axis::along_f, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_SCATTER_ELEMENTS_UPDATE_5D_FP32_3 {5, 3, 2, 20, 22}, {5, 1, 1, 2, 2}, cldnn::scatter_elements_update::scatter_elements_update_axis::along_y, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx

#define CASE_SCATTER_ELEMENTS_UPDATE_5D_FP16_1 {13, 2, 1, 2, 1}, {2, 2, 1, 2, 1}, cldnn::scatter_elements_update::scatter_elements_update_axis::along_b, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_SCATTER_ELEMENTS_UPDATE_5D_FP16_2 {1, 13, 1, 2, 1}, {1, 2, 1, 2, 1}, cldnn::scatter_elements_update::scatter_elements_update_axis::along_f, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_SCATTER_ELEMENTS_UPDATE_5D_FP16_3 {2, 3, 1, 13, 13}, {2, 3, 1, 2, 1}, cldnn::scatter_elements_update::scatter_elements_update_axis::along_y, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx

class ScatterElementsUpdatePrimitiveFusingTest : public ::BaseFusingTest<scatter_elements_update_test_params> {
public:
    void execute(scatter_elements_update_test_params& p) {

        auto input_prim = get_mem(get_input_layout(p), -5, 5);
        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);
        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(scatter_elements_update_test_params& p) {
        return layout{ p.data_type, p.input_format, p.input_shape };
    }

    layout get_indices_layout(scatter_elements_update_test_params& p) {
        return layout{ p.data_type, p.input_format, p.indices_shape };
    }

    layout get_updates_layout(scatter_elements_update_test_params& p) {
        return layout{ p.data_type, p.input_format, p.indices_shape };
    }

    size_t get_axis_dim(scatter_elements_update_test_params& p) {
        switch (p.axis) {
            case cldnn::scatter_elements_update::scatter_elements_update_axis::along_x:
                return p.input_shape.spatial[0];
            case cldnn::scatter_elements_update::scatter_elements_update_axis::along_y:
                return p.input_shape.spatial[1];
            case cldnn::scatter_elements_update::scatter_elements_update_axis::along_z:
                return p.input_shape.spatial[2];
            case cldnn::scatter_elements_update::scatter_elements_update_axis::along_w:
                return p.input_shape.spatial[3];
            case cldnn::scatter_elements_update::scatter_elements_update_axis::along_f:
                return p.input_shape.feature[0];
            case cldnn::scatter_elements_update::scatter_elements_update_axis::along_b:
                return p.input_shape.batch[0];
            default:
                return 1;
        }
    }

    layout get_per_channel_layout(scatter_elements_update_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{1, p.input_shape.feature[0], 1, 1} };
    }
};

class scatter_elements_update_quantize : public ScatterElementsUpdatePrimitiveFusingTest {};
TEST_P(scatter_elements_update_quantize, basic) {
    auto p = GetParam();
    const auto &seu = scatter_elements_update("scatter_elements_update_prim", "input", "scatter_elements_update_indices", "scatter_elements_update_updates", p.axis);
    const auto &q = quantize("quantize", "scatter_elements_update_prim", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8);
    const auto &r = reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32);
    create_topologies(input_layout("input", get_input_layout(p)),
        data("scatter_elements_update_indices", get_repeatless_mem(get_indices_layout(p), 0, static_cast<int>(get_axis_dim(p)) - 1)),
        data("scatter_elements_update_updates", get_mem(get_updates_layout(p), 0, 100)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        seu,
        q,
        r
    );
    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, scatter_elements_update_quantize,
    ::testing::ValuesIn(std::vector<scatter_elements_update_test_params>{
                        scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_FP32_1, 2, 3 },
                        scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_FP32_2, 2, 3 },
                        scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_FP32_3, 2, 3 },

                        scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_FP16_1, 2, 3 },

                        scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_5D_FP32_1, 2, 3 },
                        scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_5D_FP32_2, 2, 3 },
                        scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_5D_FP32_3, 2, 3 },

                        scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_5D_FP16_1, 2, 3 },
                        scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_5D_FP16_2, 2, 3 },
                        scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_5D_FP16_3, 2, 3 },
}));

class scatter_elements_update_scale_activation_eltwise : public ScatterElementsUpdatePrimitiveFusingTest {};
TEST_P(scatter_elements_update_scale_activation_eltwise, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
        data("scatter_elements_update_indices", get_repeatless_mem(get_indices_layout(p), 0, static_cast<int>(get_axis_dim(p)) - 1)),
        data("scatter_elements_update_updates", get_mem(get_updates_layout(p), 0, 5)),
        data("scale_data", get_mem(get_per_channel_layout(p), -1, 1)),
        data("eltwise_data", get_mem(layout{ p.data_type, p.input_format, p.input_shape})),
        scatter_elements_update("scatter_elements_update_prim", "input", "scatter_elements_update_indices", "scatter_elements_update_updates", p.axis),
        activation("activation", "scatter_elements_update_prim", activation_func::abs),
        scale("scale", "activation", "scale_data"),
        eltwise("eltwise", {"scale", "eltwise_data"}, eltwise_mode::sum, p.data_type),
        reorder("reorder_bfyx", "eltwise", p.default_format, data_types::f32)
    );
    tolerance = 1e-2f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, scatter_elements_update_scale_activation_eltwise,
    ::testing::ValuesIn(std::vector<scatter_elements_update_test_params>{
                        scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_FP32_1, 2, 5 },
                        scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_FP32_2, 2, 5 },
                        scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_FP32_3, 2, 5 },

                        scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_FP16_1, 2, 5 },

                        scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_5D_FP32_1, 2, 5 },
                        scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_5D_FP32_2, 2, 5 },
                        scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_5D_FP32_3, 2, 5 },

                        scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_5D_FP16_1, 2, 5 },
                        scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_5D_FP16_2, 2, 5 },
                        scatter_elements_update_test_params{ CASE_SCATTER_ELEMENTS_UPDATE_5D_FP16_3, 2, 5 },
}));

/* ------------------------------------------------------------------------------------------------------------ */
/* ---------------------------------------- PERMUTE FUSE cases -------------------------------------------------- */
/* ------------------------------------------------------------------------------------------------------------ */
struct permute_params {
    tensor in_shape;
    tensor out_shape;
    std::vector<uint16_t> permute_order;
    tensor eltw_in_shape;
    data_types data_type;
    format input_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

#define CASE_PERMUTE_F32_0 {1, 16, 2, 2}, {1, 16, 2, 2}, {0, 1, 2, 3}, tensor{0}, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_F32_1 {1, 15, 16, 16}, {1, 15, 16, 16}, {0, 1, 2, 3}, tensor{0}, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_F32_2 {1, 8, 16, 16}, {16, 16, 8, 1}, {3, 2, 1, 0}, tensor{0}, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_F32_3 {1, 1, 3, 4}, {1, 3, 4, 1}, {1, 2, 3, 0}, tensor{0}, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_F32_4 {2, 16, 16, 16}, {2, 16, 16, 16}, {0, 1, 2, 3}, tensor{0}, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_F32_5 {1, 32, 4, 5}, {32, 4, 5, 1}, {1, 2, 3, 0}, tensor{0}, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_F32_6 {1, 16, 4, 5}, {5, 16, 4, 1}, {3, 1, 2, 0}, tensor{0}, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_F32_7 {1, 16, 1, 1}, {1, 1, 1, 16}, {2, 3, 0, 1}, tensor{0}, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx

#define CASE_PERMUTE_F16_0 {1, 16, 4, 5}, {1, 16, 4, 5}, {0, 1, 2, 3}, tensor{0}, data_types::f16, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_F16_1 {2, 16, 4, 5}, {16, 4, 5, 2}, {1, 2, 3, 0}, tensor{0}, data_types::f16, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_F16_2 {1, 32, 2, 3}, {2, 3, 32, 1}, {2, 3, 1, 0}, tensor{0}, data_types::f16, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_F16_3 {3, 16, 1, 1}, {1, 1, 16, 3}, {3, 2, 1, 0}, tensor{0}, data_types::f16, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_F16_4 {2, 15, 4, 5}, {4, 2, 5, 15}, {2, 0, 3, 1}, tensor{0}, data_types::f16, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_F16_5 {1, 15, 1, 2}, {15, 2, 1, 1}, {1, 3, 2, 0}, tensor{0}, data_types::f16, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_F16_6 {1, 15, 4, 4}, {4, 4, 1, 15}, {2, 3, 0, 1}, tensor{0}, data_types::f16, format::bfyx, data_types::f32, format::bfyx

#define CASE_PERMUTE_S8_0 {1, 15, 4, 5}, {1, 15, 4, 5}, {0, 1, 2, 3}, tensor{0}, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_S8_1 {1, 15, 4, 5}, {5, 4, 15, 1}, {3, 2, 1, 0}, tensor{0}, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_S8_2 {1, 16, 1, 2}, {1, 1, 16, 2}, {2, 0, 1, 3}, tensor{0}, data_types::i8, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_S8_3 {1, 16, 2, 2}, {2, 2, 16, 1}, {2, 3, 1, 0}, tensor{0}, data_types::i8, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_U8_0 {1, 15, 4, 5}, {15, 5, 1, 4}, {1, 3, 0, 2}, tensor{0}, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_U8_1 {1, 15, 16, 16}, {15, 16, 1, 16}, {1, 2, 0, 3}, tensor{0}, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_U8_2 {1, 32, 5, 4}, {1, 32, 5, 4}, {0, 1, 2, 3}, tensor{0}, data_types::u8, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_PERMUTE_U8_3 {1, 16, 4, 5}, {5, 4, 16, 1}, {3, 2, 1, 0}, tensor{0}, data_types::u8, format::b_fs_yx_fsv16, data_types::f32, format::bfyx

// 3d
#define CASE_PERMUTE_F32_3D_0 {1, 15, 4, 4, 5}, {1, 15, 4, 4, 5}, {0, 1, 2, 3, 4}, tensor{0}, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_F32_3D_1 {2, 15, 2, 3, 4}, {15, 2, 3, 4, 2}, {1, 2, 3, 4, 0}, tensor{0}, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_F32_3D_2 {2, 16, 4, 4, 5}, {4, 2, 4, 5, 16}, {3, 0, 2, 4, 1}, tensor{0}, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_F32_3D_3 {1, 32, 4, 2, 2}, {2, 2, 32, 1, 4}, {4, 3, 1, 0, 2}, tensor{0}, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_F32_3D_4 {1, 16, 1, 1, 1}, {1, 1, 1, 16, 1}, {2, 4, 0, 1, 3}, tensor{0}, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx

#define CASE_PERMUTE_F16_3D_0 {1, 15, 4, 4, 5}, {1, 15, 4, 4, 5}, {0, 1, 2, 3, 4}, tensor{0}, data_types::f16, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_F16_3D_1 {2, 15, 4, 3, 4}, {4, 4, 2, 15, 3}, {2, 4, 0, 1, 3}, tensor{0}, data_types::f16, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_F16_3D_2 {2, 16, 4, 4, 3}, {2, 4, 3, 16, 4}, {0, 3, 4, 1, 2}, tensor{0}, data_types::f16, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_F16_3D_3 {1, 32, 4, 2, 1}, {2, 32, 4, 1, 1}, {3, 1, 2, 4, 0}, tensor{0}, data_types::f16, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_F16_3D_4 {16, 16, 1, 1, 1},{1, 16, 1, 1, 16},{4, 0, 3, 2, 1}, tensor{0}, data_types::f16, format::bfzyx, data_types::f32, format::bfzyx

#define CASE_PERMUTE_S8_3D_0 {1, 15, 4, 4, 5}, {1, 15, 4, 4, 5}, {0, 1, 2, 3, 4}, tensor{0}, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_S8_3D_1 {2, 15, 4, 3, 4}, {4, 4, 15, 2, 3}, {4, 2, 1, 0, 3}, tensor{0}, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_S8_3D_2 {2, 16, 4, 4, 3}, {2, 4, 3, 16, 4}, {0, 3, 4, 1, 2}, tensor{0}, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_S8_3D_3 {1, 32, 4, 2, 1}, {2, 32, 4, 1, 1}, {3, 1, 2, 4, 0}, tensor{0}, data_types::i8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_U8_3D_0 {16, 16, 1, 1, 1}, {1, 1, 16, 16, 1}, {2, 4, 0, 1, 3}, tensor{0}, data_types::u8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_U8_3D_1 {16, 16, 1, 1, 1}, {1, 1, 1, 16, 16}, {4, 3, 2, 1, 0}, tensor{0}, data_types::u8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_U8_3D_2 {2, 16, 4, 4, 3}, {4, 2, 4, 3, 16}, {3, 0, 2, 4, 1}, tensor{0}, data_types::u8, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_U8_3D_3 {1, 32, 4, 2, 1}, {1, 2, 32, 1, 4}, {4, 3, 1, 0, 2}, tensor{0}, data_types::u8, format::bfzyx, data_types::f32, format::bfzyx

// permute_tile_8x8_4x4
#define CASE_PERMUTE_TILE_8x8_4x4_4D_0 {1, 8, 8, 2}, {1, 2, 8, 8}, {0, 3, 1, 2}, tensor{0}, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_TILE_8x8_4x4_4D_1 {1, 5, 8, 2}, {1, 2, 5, 8}, {0, 3, 1, 2}, tensor{0}, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_TILE_8x8_4x4_4D_2 {1, 8, 5, 2}, {1, 2, 8, 5}, {0, 3, 1, 2}, tensor{0}, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_TILE_8x8_4x4_4D_3 {1, 5, 5, 2}, {1, 2, 5, 5}, {0, 3, 1, 2}, tensor{0}, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_PERMUTE_TILE_8x8_4x4_5D_0 {1, 8, 8, 2, 2}, {1, 2, 8, 8, 2}, {0, 4, 1, 2, 3}, tensor{0}, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_TILE_8x8_4x4_5D_1 {1, 5, 8, 2, 2}, {1, 2, 5, 8, 2}, {0, 4, 1, 2, 3}, tensor{0}, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_TILE_8x8_4x4_5D_2 {1, 8, 5, 2, 2}, {1, 2, 8, 5, 2}, {0, 4, 1, 2, 3}, tensor{0}, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_TILE_8x8_4x4_5D_3 {1, 5, 5, 2, 2}, {1, 2, 5, 5, 2}, {0, 4, 1, 2, 3}, tensor{0}, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_PERMUTE_TILE_8x8_4x4_6D_0 {1, 8, 8, 2, 2, 2}, {1, 2, 8, 8, 2, 2}, {0, 5, 1, 2, 3, 4}, tensor{0}, data_types::f32, format::bfwzyx, data_types::f32, format::bfwzyx
#define CASE_PERMUTE_TILE_8x8_4x4_6D_1 {1, 5, 8, 2, 2, 2}, {1, 2, 5, 8, 2, 2}, {0, 5, 1, 2, 3, 4}, tensor{0}, data_types::f32, format::bfwzyx, data_types::f32, format::bfwzyx
#define CASE_PERMUTE_TILE_8x8_4x4_6D_2 {1, 8, 5, 2, 2, 2}, {1, 2, 8, 5, 2, 2}, {0, 5, 1, 2, 3, 4}, tensor{0}, data_types::f32, format::bfwzyx, data_types::f32, format::bfwzyx
#define CASE_PERMUTE_TILE_8x8_4x4_6D_3 {1, 5, 5, 2, 2, 2}, {1, 2, 5, 5, 2, 2}, {0, 5, 1, 2, 3, 4}, tensor{0}, data_types::f32, format::bfwzyx, data_types::f32, format::bfwzyx

// permute_tile_8x8_4x4_fsv16
#define CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_0 {1, 16, 16, 2}, {1, 2, 16, 16}, {0, 3, 1, 2}, tensor{0}, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::b_fs_yx_fsv16
#define CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_1 {1, 15, 16, 2}, {1, 2, 15, 16}, {0, 3, 1, 2}, tensor{0}, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::b_fs_yx_fsv16
#define CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_2 {1, 16,  3, 2}, {1, 2, 16,  3}, {0, 3, 1, 2}, tensor{0}, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::b_fs_yx_fsv16
#define CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_3 {1,  5,  7, 2}, {1, 2,  5,  7}, {0, 3, 1, 2}, tensor{0}, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::b_fs_yx_fsv16
#define CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_0 {1, 16, 16, 2, 2}, {1, 2, 16, 16, 2}, {0, 4, 1, 2, 3}, tensor{0}, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::b_fs_zyx_fsv16
#define CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_1 {1, 15, 16, 2, 2}, {1, 2, 15, 16, 2}, {0, 4, 1, 2, 3}, tensor{0}, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::b_fs_zyx_fsv16
#define CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_2 {1, 16,  3, 2, 2}, {1, 2, 16,  3, 2}, {0, 4, 1, 2, 3}, tensor{0}, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::b_fs_zyx_fsv16
#define CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_3 {1,  5,  7, 2, 2}, {1, 2,  5,  7, 2}, {0, 4, 1, 2, 3}, tensor{0}, data_types::f32, format::b_fs_zyx_fsv16, data_types::f32, format::b_fs_zyx_fsv16

class PermuteFusingTest : public ::BaseFusingTest<permute_params> {
public:

    void execute(permute_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(permute_params& p) {
        return layout{ p.data_type, p.input_format, p.in_shape, padding{} };
    }

    layout get_per_channel_layout(permute_params& p) {
        return layout{ p.default_type, p.default_format, tensor{1, p.out_shape.feature[0], 1, 1} };
    }
};

class permute_activation_scale_eltwise: public PermuteFusingTest {};
TEST_P(permute_activation_scale_eltwise, basic) {
        auto p = GetParam();

        create_topologies(
            input_layout("input", get_input_layout(p)),
            data("eltwise_data", get_mem(layout{ p.data_type, p.input_format, p.out_shape})),
            data("scale_data", get_mem(get_per_channel_layout(p), 5e-1f)),
            permute("permute", "input", p.permute_order),
            scale("scale", "permute", "scale_data"),
            activation("actv", "scale", activation_func::relu),
            eltwise("eltwise", {"actv", "eltwise_data"}, eltwise_mode::sum, p.data_type),
            reorder("reorder_bfyx", "eltwise", p.default_format, p.default_type)
        );

        tolerance = 1e-5f;
        execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, permute_activation_scale_eltwise,
                        ::testing::ValuesIn(std::vector<permute_params> {
                            permute_params{CASE_PERMUTE_F32_0, 2, 5},
                            permute_params{CASE_PERMUTE_F32_1, 2, 5},
                            permute_params{CASE_PERMUTE_F32_2, 2, 5},
                            permute_params{CASE_PERMUTE_F32_3, 2, 5},
                            permute_params{CASE_PERMUTE_F32_4, 2, 5},
                            permute_params{CASE_PERMUTE_F32_5, 2, 5},
                            permute_params{CASE_PERMUTE_F32_6, 2, 5},
                            permute_params{CASE_PERMUTE_F32_7, 2, 5},

                            permute_params{CASE_PERMUTE_F16_0, 2, 5},
                            permute_params{CASE_PERMUTE_F16_1, 2, 5},
                            permute_params{CASE_PERMUTE_F16_2, 2, 5},
                            permute_params{CASE_PERMUTE_F16_3, 2, 5},
                            permute_params{CASE_PERMUTE_F16_4, 2, 5},
                            permute_params{CASE_PERMUTE_F16_5, 2, 5},
                            permute_params{CASE_PERMUTE_F16_6, 2, 5},

                            permute_params{CASE_PERMUTE_S8_0, 2, 5},
                            permute_params{CASE_PERMUTE_S8_1, 2, 5},
                            permute_params{CASE_PERMUTE_S8_2, 2, 5},
                            permute_params{CASE_PERMUTE_S8_3, 2, 5},

                            permute_params{CASE_PERMUTE_U8_0, 2, 5},
                            permute_params{CASE_PERMUTE_U8_1, 2, 5},
                            permute_params{CASE_PERMUTE_U8_2, 2, 5},
                            permute_params{CASE_PERMUTE_U8_3, 2, 5},

                            permute_params{CASE_PERMUTE_F32_3D_0, 2, 5},
                            permute_params{CASE_PERMUTE_F32_3D_1, 2, 5},
                            permute_params{CASE_PERMUTE_F32_3D_2, 2, 5},
                            permute_params{CASE_PERMUTE_F32_3D_3, 2, 5},
                            permute_params{CASE_PERMUTE_F32_3D_4, 2, 5},

                            permute_params{CASE_PERMUTE_F16_3D_0, 2, 5},
                            permute_params{CASE_PERMUTE_F16_3D_1, 2, 5},
                            permute_params{CASE_PERMUTE_F16_3D_2, 2, 5},
                            permute_params{CASE_PERMUTE_F16_3D_3, 2, 5},
                            permute_params{CASE_PERMUTE_F16_3D_4, 2, 5},

                            permute_params{CASE_PERMUTE_S8_3D_0, 2, 5},
                            permute_params{CASE_PERMUTE_S8_3D_1, 2, 5},
                            permute_params{CASE_PERMUTE_S8_3D_2, 2, 5},
                            permute_params{CASE_PERMUTE_S8_3D_3, 2, 5},

                            permute_params{CASE_PERMUTE_U8_3D_0, 2, 5},
                            permute_params{CASE_PERMUTE_U8_3D_1, 2, 5},
                            permute_params{CASE_PERMUTE_U8_3D_2, 2, 5},
                            permute_params{CASE_PERMUTE_U8_3D_3, 2, 5},

                            // Fusing tests for permute_tile_8x8_4x4
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_4D_0, 2, 5},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_4D_1, 2, 5},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_4D_2, 2, 5},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_4D_3, 2, 5},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_5D_0, 2, 5},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_5D_1, 2, 5},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_5D_2, 2, 5},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_5D_3, 2, 5},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_6D_0, 2, 5},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_6D_1, 2, 5},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_6D_2, 2, 5},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_6D_3, 2, 5},

                            // Fusing tests for permute_tile_8x8_4x4_fsv16
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_0, 2, 5},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_1, 2, 5},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_2, 2, 5},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_3, 2, 5},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_0, 2, 5},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_1, 2, 5},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_2, 2, 5},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_3, 2, 5},
                        }));

class permute_quant_u8: public PermuteFusingTest {};
TEST_P(permute_quant_u8, basic) {
        auto p = GetParam();
        create_topologies(
        input_layout("input", get_input_layout(p)),
        data("in_lo", get_mem(get_single_element_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_single_element_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        permute("permute", "input", p.permute_order),
        quantize("quant", "permute", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::u8),
        reorder("reorder_bfyx", "quant", p.default_format, p.default_type)
        );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, permute_quant_u8,
                        ::testing::ValuesIn(std::vector<permute_params> {
                            permute_params{CASE_PERMUTE_F32_0, 2, 3},
                            permute_params{CASE_PERMUTE_F32_1, 2, 3},

                            permute_params{CASE_PERMUTE_F16_0, 2, 3},
                            permute_params{CASE_PERMUTE_F16_1, 2, 3},
                        }));

class permute_scale_actv_eltw_scale_actv_quant_i8: public PermuteFusingTest {};
TEST_P(permute_scale_actv_eltw_scale_actv_quant_i8, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("scale1_data", get_mem(get_per_channel_layout(p), 1e-1f)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        data("eltw_data", get_mem(layout(p.data_type, p.input_format, p.out_shape))),
        data("scale2_data", get_mem(get_per_channel_layout(p), 1e-1f)),
        permute("permute", "input", p.permute_order),
        scale("scale1", "permute", "scale1_data"),
        activation("actv1", "scale1", activation_func::relu),
        eltwise("eltw", {"actv1", "eltw_data"}, eltwise_mode::sum, p.data_type),
        scale("scale2", "eltw", "scale2_data"),
        activation("actv2", "scale2", activation_func::relu),
        quantize("quant", "actv2", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("out", "quant", p.default_format, p.default_type)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, permute_scale_actv_eltw_scale_actv_quant_i8,
                        ::testing::ValuesIn(std::vector<permute_params> {
                            permute_params{CASE_PERMUTE_F32_0, 2, 8},
                            permute_params{CASE_PERMUTE_F32_1, 2, 8},
                            permute_params{CASE_PERMUTE_F32_2, 2, 8},
                            permute_params{CASE_PERMUTE_F32_3, 2, 8},
                            permute_params{CASE_PERMUTE_F32_4, 2, 8},
                            permute_params{CASE_PERMUTE_F32_5, 2, 8},
                            permute_params{CASE_PERMUTE_F32_6, 2, 8},
                            permute_params{CASE_PERMUTE_F32_7, 2, 8},

                            permute_params{CASE_PERMUTE_F16_0, 2, 8},
                            permute_params{CASE_PERMUTE_F16_1, 2, 8},
                            permute_params{CASE_PERMUTE_F16_2, 2, 8},
                            permute_params{CASE_PERMUTE_F16_3, 2, 8},
                            permute_params{CASE_PERMUTE_F16_4, 2, 8},
                            permute_params{CASE_PERMUTE_F16_5, 2, 8},
                            permute_params{CASE_PERMUTE_F16_6, 2, 8},

                            permute_params{CASE_PERMUTE_S8_0, 2, 8},
                            permute_params{CASE_PERMUTE_S8_1, 2, 8},
                            permute_params{CASE_PERMUTE_S8_2, 2, 8},
                            permute_params{CASE_PERMUTE_S8_3, 2, 8},

                            permute_params{CASE_PERMUTE_U8_0, 2, 8},
                            permute_params{CASE_PERMUTE_U8_1, 2, 8},
                            permute_params{CASE_PERMUTE_U8_2, 2, 8},
                            permute_params{CASE_PERMUTE_U8_3, 2, 8},

                            permute_params{CASE_PERMUTE_F32_3D_0, 2, 8},
                            permute_params{CASE_PERMUTE_F32_3D_1, 2, 8},
                            permute_params{CASE_PERMUTE_F32_3D_2, 2, 8},
                            permute_params{CASE_PERMUTE_F32_3D_3, 2, 8},
                            permute_params{CASE_PERMUTE_F32_3D_4, 2, 8},

                            permute_params{CASE_PERMUTE_F16_3D_0, 2, 8},
                            permute_params{CASE_PERMUTE_F16_3D_1, 2, 8},
                            permute_params{CASE_PERMUTE_F16_3D_2, 2, 8},
                            permute_params{CASE_PERMUTE_F16_3D_3, 2, 8},
                            permute_params{CASE_PERMUTE_F16_3D_4, 2, 8},

                            permute_params{CASE_PERMUTE_S8_3D_0, 2, 8},
                            permute_params{CASE_PERMUTE_S8_3D_1, 2, 8},
                            permute_params{CASE_PERMUTE_S8_3D_2, 2, 8},
                            permute_params{CASE_PERMUTE_S8_3D_3, 2, 8},

                            permute_params{CASE_PERMUTE_U8_3D_0, 2, 8},
                            permute_params{CASE_PERMUTE_U8_3D_1, 2, 8},
                            permute_params{CASE_PERMUTE_U8_3D_2, 2, 8},
                            permute_params{CASE_PERMUTE_U8_3D_3, 2, 8},
                        }));

class permute_scale_eltwise_actv_scale_actv: public PermuteFusingTest {};
TEST_P(permute_scale_eltwise_actv_scale_actv, basic) {
    auto p = GetParam();

        create_topologies(
            input_layout("input", get_input_layout(p)),
            data("eltwise_data", get_mem(layout{ p.data_type, p.input_format, p.out_shape})),
            data("scale_data1", get_mem(get_per_channel_layout(p), 1e-1f)),
            data("scale_data2", get_mem(get_per_channel_layout(p), 1e-1f)),
            permute("permute", "input", p.permute_order),
            scale("scale1", "permute", "scale_data1"),
            activation("actv1", "scale1", activation_func::relu),
            eltwise("eltwise", {"actv1", "eltwise_data"}, eltwise_mode::sum, p.default_type),
            scale("scale2", "eltwise", "scale_data2"),
            activation("actv2", "scale2", activation_func::relu),
            reorder("reorder_bfyx", "actv2", p.default_format, p.default_type)
        );

        tolerance = 1e-5f;
        execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, permute_scale_eltwise_actv_scale_actv,
                        ::testing::ValuesIn(std::vector<permute_params> {
                            permute_params{CASE_PERMUTE_F32_0, 2, 7},
                            permute_params{CASE_PERMUTE_F32_1, 2, 7},
                            permute_params{CASE_PERMUTE_F32_2, 2, 7},
                            permute_params{CASE_PERMUTE_F32_3, 2, 7},
                            permute_params{CASE_PERMUTE_F32_4, 2, 7},
                            permute_params{CASE_PERMUTE_F32_5, 2, 7},
                            permute_params{CASE_PERMUTE_F32_6, 2, 7},
                            permute_params{CASE_PERMUTE_F32_7, 2, 7},

                            permute_params{CASE_PERMUTE_F16_0, 2, 7},
                            permute_params{CASE_PERMUTE_F16_1, 2, 7},
                            permute_params{CASE_PERMUTE_F16_2, 2, 7},
                            permute_params{CASE_PERMUTE_F16_3, 2, 7},
                            permute_params{CASE_PERMUTE_F16_4, 2, 7},
                            permute_params{CASE_PERMUTE_F16_5, 2, 7},
                            permute_params{CASE_PERMUTE_F16_6, 2, 7},

                            permute_params{CASE_PERMUTE_S8_0, 2, 7},
                            permute_params{CASE_PERMUTE_S8_1, 2, 7},
                            permute_params{CASE_PERMUTE_S8_2, 2, 7},
                            permute_params{CASE_PERMUTE_S8_3, 2, 7},

                            permute_params{CASE_PERMUTE_U8_0, 2, 7},
                            permute_params{CASE_PERMUTE_U8_1, 2, 7},
                            permute_params{CASE_PERMUTE_U8_2, 2, 7},
                            permute_params{CASE_PERMUTE_U8_3, 2, 7},

                            permute_params{CASE_PERMUTE_F32_3D_0, 2, 7},
                            permute_params{CASE_PERMUTE_F32_3D_1, 2, 7},
                            permute_params{CASE_PERMUTE_F32_3D_2, 2, 7},
                            permute_params{CASE_PERMUTE_F32_3D_3, 2, 7},
                            permute_params{CASE_PERMUTE_F32_3D_4, 2, 7},

                            permute_params{CASE_PERMUTE_F16_3D_0, 2, 7},
                            permute_params{CASE_PERMUTE_F16_3D_1, 2, 7},
                            permute_params{CASE_PERMUTE_F16_3D_2, 2, 7},
                            permute_params{CASE_PERMUTE_F16_3D_3, 2, 7},
                            permute_params{CASE_PERMUTE_F16_3D_4, 2, 7},

                            permute_params{CASE_PERMUTE_S8_3D_0, 2, 7},
                            permute_params{CASE_PERMUTE_S8_3D_1, 2, 7},
                            permute_params{CASE_PERMUTE_S8_3D_2, 2, 7},
                            permute_params{CASE_PERMUTE_S8_3D_3, 2, 7},

                            permute_params{CASE_PERMUTE_U8_3D_0, 2, 7},
                            permute_params{CASE_PERMUTE_U8_3D_1, 2, 7},
                            permute_params{CASE_PERMUTE_U8_3D_2, 2, 7},
                            permute_params{CASE_PERMUTE_U8_3D_3, 2, 7},

                            // Fusing tests for permute_tile_8x8_4x4
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_4D_0, 2, 7},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_4D_1, 2, 7},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_4D_2, 2, 7},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_4D_3, 2, 7},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_5D_0, 2, 7},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_5D_1, 2, 7},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_5D_2, 2, 7},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_5D_3, 2, 7},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_6D_0, 2, 7},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_6D_1, 2, 7},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_6D_2, 2, 7},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_6D_3, 2, 7},

                            // Fusing tests for permute_tile_8x8_4x4_fsv16
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_0, 2, 7},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_1, 2, 7},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_2, 2, 7},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_FSV16_4D_3, 2, 7},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_0, 2, 7},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_1, 2, 7},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_2, 2, 7},
                            permute_params{CASE_PERMUTE_TILE_8x8_4x4_FSV16_5D_3, 2, 7},
                        }));

/* ------------------------------------------------------------------------------------------------------------ */
/* ---------------------------- PERMUTE FUSE REDUNDANT REORDER cases ------------------------------------------ */
/* ------------------------------------------------------------------------------------------------------------ */

struct permute_reorder_params {
    tensor in_shape;
    std::vector<uint16_t> permute_order1;
    std::vector<uint16_t> permute_order2;
    data_types permute_type;
    data_types output_type;
    format permute_format;
    format output_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

#define CASE_PERMUTE_REORDER_F32_0 {1, 16, 32, 2},   {0, 3, 2, 1},    {0, 3, 2, 1},    data_types::f32, data_types::f32, format::b_fs_yx_fsv16,  format::bfyx
#define CASE_PERMUTE_REORDER_F32_1 {2, 7, 9, 27},  {0, 3, 2, 1},    {0, 3, 2, 1},    data_types::f32, data_types::f32, format::b_fs_yx_fsv4,   format::bfyx
#define CASE_PERMUTE_REORDER_F32_2 {1, 16, 4, 5, 16}, {0, 3, 2, 1, 4}, {0, 3, 2, 1, 4}, data_types::f32, data_types::f32, format::b_fs_zyx_fsv16, format::bfzyx
#define CASE_PERMUTE_REORDER_F16_0 {1, 16, 2, 4},     {0, 3, 2, 1},    {0, 3, 2, 1},    data_types::f16, data_types::f16, format::b_fs_yx_fsv16,  format::bfyx
#define CASE_PERMUTE_REORDER_F16_1 {1, 16, 4, 5, 16}, {0, 4, 2, 3, 1}, {0, 1, 2, 3, 4}, data_types::f16, data_types::f16, format::b_fs_zyx_fsv16, format::bfzyx
#define CASE_PERMUTE_REORDER_F16_2 {1, 5, 1, 2, 14},  {0, 3, 2, 1, 4}, {0, 3, 2, 1, 4}, data_types::f16, data_types::f16, format::b_fs_zyx_fsv16, format::bfzyx

// type change
#define CASE_PERMUTE_REORDER_S8_TO_F32_0 {1, 15, 4, 5},    {0, 3, 2, 1},    {0, 3, 2, 1},    data_types::i8, data_types::f32, format::b_fs_yx_fsv4,   format::bfyx
#define CASE_PERMUTE_REORDER_S8_TO_F32_1 {1, 2, 15, 4, 5}, {0, 3, 2, 1, 4}, {0, 3, 2, 1, 4}, data_types::i8, data_types::f32, format::b_fs_zyx_fsv16, format::bfzyx
#define CASE_PERMUTE_REORDER_F32_TO_F16_0 {1, 5, 1, 2, 14}, {0, 4, 2, 3, 1}, {0, 1, 2, 3, 4}, data_types::f32, data_types::f16, format::b_fs_zyx_fsv16, format::bfzyx
#define CASE_PERMUTE_REORDER_U8_TO_F16_0 {1, 17, 1, 2, 7},  {0, 4, 2, 3, 1}, {0, 1, 2, 3, 4}, data_types::u8, data_types::f16, format::b_fs_zyx_fsv16, format::bfzyx

// dim change
#define CASE_PERMUTE_REORDER_4D_TO_5D_F32_0 {1, 16, 8, 16}, {1, 2, 0, 3}, {0, 3, 1, 4, 2}, data_types::f32, data_types::f32, format::bfyx, format::bfzyx
#define CASE_PERMUTE_REORDER_4D_TO_6D_F32_1 {1, 16, 8, 16}, {0, 2, 3, 1}, {0, 3, 5, 4, 1, 2}, data_types::f32, data_types::f32, format::bfyx, format::bfwzyx
#define CASE_PERMUTE_REORDER_5D_TO_4D_F32_0 {1, 16, 4, 5, 18},{0, 4, 2, 3, 1}, {0, 3, 1, 2}, data_types::f32, data_types::f32, format::b_fs_zyx_fsv16, format::bfyx
#define CASE_PERMUTE_REORDER_5D_TO_4D_F32_1 {1, 16, 4, 5, 16},{0, 2, 3, 4, 1}, {0, 3, 1, 2}, data_types::f32, data_types::f32, format::bfzyx, format::bfyx
#define CASE_PERMUTE_REORDER_5D_TO_6D_F32_2 {1, 16, 8, 4, 16}, {0, 4, 2, 3, 1}, {0, 3, 5, 4, 1, 2}, data_types::f32, data_types::f32, format::b_fs_zyx_fsv16, format::bfwzyx
#define CASE_PERMUTE_REORDER_6D_TO_4D_F32_0 {1, 16, 4, 5, 4, 16}, {0, 2, 5, 3, 4, 1}, {0, 3, 1, 2}, data_types::f32, data_types::f32, format::bfwzyx, format::bfyx
#define CASE_PERMUTE_REORDER_6D_TO_5D_F32_1 {1, 16, 4, 5, 4, 16}, {0, 2, 5, 3, 4, 1}, {0, 3, 4, 1, 2}, data_types::f32, data_types::f32, format::bfwzyx, format::bfzyx

// permute_opt for blocked format
#define CASE_PERMUTE_REORDER_TILED_F32_0 {1, 256, 2, 64}, {0, 3, 1, 2}, {0, 2, 3, 1},  data_types::f32, data_types::f32, format::b_fs_yx_fsv16, format::bfyx
#define CASE_PERMUTE_REORDER_TILED_F32_1 {1, 78, 2, 259}, {0, 3, 1, 2}, {0, 2, 3, 1},  data_types::f32, data_types::f32, format::b_fs_yx_fsv16, format::bfyx
#define CASE_PERMUTE_REORDER_TILED_F32_2 {1, 48, 1, 3, 259}, {0, 4, 1, 2, 3}, {0, 2, 3, 4, 1},  data_types::f32, data_types::f32, format::b_fs_zyx_fsv16, format::bfzyx

// permute_opt for blocked format => reorder to differnt dim
#define CASE_PERMUTE_REORDER_TILED_F32_3 {1, 45, 1, 3, 259}, {0, 4, 1, 2, 3}, {0, 2, 3, 1},  data_types::f32, data_types::f32, format::b_fs_zyx_fsv16, format::bfyx
#define CASE_PERMUTE_REORDER_TILED_F32_4 {2, 273, 19, 19}, {0, 3, 1, 2}, {0, 2, 3, 1},  data_types::f32, data_types::f32, format::b_fs_yx_fsv16, format::bfyx
#define CASE_PERMUTE_REORDER_TILED_F32_5 {2, 546, 2, 2}, {0, 3, 1, 2}, {0, 2, 3, 1},  data_types::f32, data_types::f32, format::b_fs_yx_fsv16, format::bfyx

// permute opt for blocked format => reorder to different dim/type
#define CASE_PERMUTE_REORDER_TILED_I8_4 {1, 45, 1, 3, 259}, {0, 4, 1, 2, 3}, {0, 2, 3, 1},  data_types::i8, data_types::f32, format::b_fs_zyx_fsv16, format::bfyx
#define CASE_PERMUTE_REORDER_TILED_F16_5 {1, 48, 3, 256}, {0, 3, 1, 2}, {0, 2, 4, 3, 1},  data_types::f16, data_types::f32, format::b_fs_yx_fsv16, format::bfzyx
#define CASE_PERMUTE_REORDER_TILED_F16_6 {1, 48, 2, 3, 256}, {0, 4, 1, 2, 3}, {0, 2, 5, 4, 3, 1},  data_types::f16, data_types::f32, format::b_fs_zyx_fsv16, format::bfwzyx

// permute opt for non_blocked format => reorder to differnt dim/type
#define CASE_PERMUTE_REORDER_TILED_F16_7 {1, 48, 2, 3, 256}, {0, 4, 1, 2, 3}, {0, 2, 3, 1},  data_types::f16, data_types::f32, format::bfzyx, format::bfyx
#define CASE_PERMUTE_REORDER_TILED_F16_8 {1, 28, 2, 2, 3, 256}, {0, 5, 1, 2, 3, 4}, {0, 2, 3, 1},  data_types::f16, data_types::f32, format::bfwzyx, format::bfyx
#define CASE_PERMUTE_REORDER_TILED_F16_9 {1, 24, 2, 3, 256}, {0, 4, 1, 2, 3}, {0, 2, 3, 1},  data_types::f16, data_types::f32, format::bfzyx, format::bfyx
#define CASE_PERMUTE_REORDER_TILED_F16_10 {1, 35, 3, 253}, {0, 3, 1, 2}, {0, 2, 4, 3, 1},  data_types::f16, data_types::f32, format::bfyx, format::bfzyx
#define CASE_PERMUTE_REORDER_TILED_F16_11 {1, 32, 3, 253}, {0, 3, 1, 2}, {0, 2, 4, 5, 3, 1},  data_types::f16, data_types::f32, format::bfyx, format::bfwzyx

class PermuteReorderFusingTest : public ::BaseFusingTest<permute_reorder_params> {
public:

    void execute(permute_reorder_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);
        compare(network_not_fused, network_fused, p, true);
    }

    layout get_input_layout(permute_reorder_params& p) {
        return layout{ p.permute_type, p.permute_format, p.in_shape, padding{} };
    }
};

class permute_redundant_reorder : public PermuteReorderFusingTest {};

TEST_P(permute_redundant_reorder, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        permute("permute1", "input",    p.permute_order1),
        reorder("reorder1", "permute1", p.output_format, p.output_type), // to be fused
        permute("permute2", "reorder1", p.permute_order2)                  // dummy last op to make reorder fused
    );
    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, permute_redundant_reorder,
                            ::testing::ValuesIn(std::vector<permute_reorder_params> {
                                permute_reorder_params{CASE_PERMUTE_REORDER_F32_0, 3, 4},
                                permute_reorder_params{CASE_PERMUTE_REORDER_F32_1, 3, 4},
                                permute_reorder_params{CASE_PERMUTE_REORDER_F32_2, 3, 4},
                                permute_reorder_params{CASE_PERMUTE_REORDER_F16_0, 3, 4},
                                permute_reorder_params{CASE_PERMUTE_REORDER_F16_1, 3, 4},
                                permute_reorder_params{CASE_PERMUTE_REORDER_F16_2, 3, 4},
                                permute_reorder_params{CASE_PERMUTE_REORDER_S8_TO_F32_0, 3, 4},
                                permute_reorder_params{CASE_PERMUTE_REORDER_S8_TO_F32_1, 3, 4},
                                permute_reorder_params{CASE_PERMUTE_REORDER_F32_TO_F16_0, 3, 4},
                                permute_reorder_params{CASE_PERMUTE_REORDER_U8_TO_F16_0, 3, 4},
                                permute_reorder_params{CASE_PERMUTE_REORDER_4D_TO_5D_F32_0, 3, 3},
                                permute_reorder_params{CASE_PERMUTE_REORDER_4D_TO_6D_F32_1, 3, 3},
                                permute_reorder_params{CASE_PERMUTE_REORDER_5D_TO_4D_F32_0, 3, 4},
                                permute_reorder_params{CASE_PERMUTE_REORDER_5D_TO_4D_F32_1, 3, 3},
                                permute_reorder_params{CASE_PERMUTE_REORDER_5D_TO_6D_F32_2, 3, 4},
                                permute_reorder_params{CASE_PERMUTE_REORDER_6D_TO_4D_F32_0, 3, 3},
                                permute_reorder_params{CASE_PERMUTE_REORDER_6D_TO_5D_F32_1, 3, 3},
                                permute_reorder_params{CASE_PERMUTE_REORDER_TILED_F32_0, 3, 4},
                                permute_reorder_params{CASE_PERMUTE_REORDER_TILED_F32_1, 3, 4},
                                permute_reorder_params{CASE_PERMUTE_REORDER_TILED_F32_2, 3, 4},
                                permute_reorder_params{CASE_PERMUTE_REORDER_TILED_F32_3, 3, 4},
                                permute_reorder_params{CASE_PERMUTE_REORDER_TILED_F32_4, 3, 4},
                                permute_reorder_params{CASE_PERMUTE_REORDER_TILED_I8_4, 3, 4},
                                permute_reorder_params{CASE_PERMUTE_REORDER_TILED_F16_5, 3, 4},
                                permute_reorder_params{CASE_PERMUTE_REORDER_TILED_F16_6, 3, 4},
                                permute_reorder_params{CASE_PERMUTE_REORDER_TILED_F16_7, 3, 4},
                                permute_reorder_params{CASE_PERMUTE_REORDER_TILED_F16_8, 3, 4},
                                permute_reorder_params{CASE_PERMUTE_REORDER_TILED_F16_9, 3, 4},
                                permute_reorder_params{CASE_PERMUTE_REORDER_TILED_F16_10, 3, 4},
                                permute_reorder_params{CASE_PERMUTE_REORDER_TILED_F16_11, 3, 4},
                                }));

class permute_act_reorder : public PermuteReorderFusingTest {};

TEST_P(permute_act_reorder, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        permute("permute1", "input",    p.permute_order1),
        activation("activation", "permute1", activation_func::abs),
        reorder("reorder1", "activation", p.output_format, p.output_type),    // to be fused
        permute("permute2", "reorder1", p.permute_order2)                // dummy last op to make reorder fused
    );
    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, permute_act_reorder,
                            ::testing::ValuesIn(std::vector<permute_reorder_params> {
                                permute_reorder_params{CASE_PERMUTE_REORDER_F32_0, 3, 5},
                                permute_reorder_params{CASE_PERMUTE_REORDER_F32_1, 3, 5},
                                permute_reorder_params{CASE_PERMUTE_REORDER_F32_2, 3, 5},
                                permute_reorder_params{CASE_PERMUTE_REORDER_F16_0, 3, 5},
                                permute_reorder_params{CASE_PERMUTE_REORDER_F16_1, 3, 5},
                                permute_reorder_params{CASE_PERMUTE_REORDER_F16_2, 3, 5},
                                permute_reorder_params{CASE_PERMUTE_REORDER_4D_TO_5D_F32_0, 3, 4},
                                permute_reorder_params{CASE_PERMUTE_REORDER_4D_TO_6D_F32_1, 3, 4},
                                permute_reorder_params{CASE_PERMUTE_REORDER_5D_TO_4D_F32_0, 3, 5},
                                permute_reorder_params{CASE_PERMUTE_REORDER_5D_TO_4D_F32_1, 3, 4},
                                permute_reorder_params{CASE_PERMUTE_REORDER_5D_TO_6D_F32_2, 3, 5},
                                permute_reorder_params{CASE_PERMUTE_REORDER_6D_TO_4D_F32_0, 3, 4},
                                permute_reorder_params{CASE_PERMUTE_REORDER_6D_TO_5D_F32_1, 3, 4},
                                permute_reorder_params{CASE_PERMUTE_REORDER_TILED_F32_0, 3, 5},
                                permute_reorder_params{CASE_PERMUTE_REORDER_TILED_F32_1, 3, 5},
                                permute_reorder_params{CASE_PERMUTE_REORDER_TILED_F32_2, 3, 5},
                                permute_reorder_params{CASE_PERMUTE_REORDER_TILED_F32_3, 3, 5},
                                permute_reorder_params{CASE_PERMUTE_REORDER_TILED_F16_5, 3, 5},
                                permute_reorder_params{CASE_PERMUTE_REORDER_TILED_F16_6, 3, 5},
                                permute_reorder_params{CASE_PERMUTE_REORDER_TILED_F16_7, 3, 5},
                                permute_reorder_params{CASE_PERMUTE_REORDER_TILED_F16_8, 3, 5},
                                permute_reorder_params{CASE_PERMUTE_REORDER_TILED_F16_9, 3, 5},
                                permute_reorder_params{CASE_PERMUTE_REORDER_TILED_F16_10, 3, 5},
                                permute_reorder_params{CASE_PERMUTE_REORDER_TILED_F16_11, 3, 5},
                                }));

class NormalizeFusingTest : public ::BaseFusingTest<normalize_test_params> {
public:
    void execute(normalize_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
    }
    layout get_input_layout(normalize_test_params& p) { return layout{p.data_type, p.input_format, p.in_shape}; }
    layout get_per_channel_layout(normalize_test_params& p) {
        return layout{p.default_type, p.default_format, tensor{1, p.in_shape.feature[0], 1, 1}};
    }
    layout get_weights_layout(normalize_test_params& p) { return layout {p.default_type, p.default_format, tensor{1, p.in_shape.feature[0], 1, 1}}; }
};

class normalize_i8_quantize : public NormalizeFusingTest {};
TEST_P(normalize_i8_quantize, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("in_lo", get_mem(get_single_element_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_single_element_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), 0)),
        data("out_hi", get_mem(get_single_element_layout(p), 255)),
        normalize("normalizel2", "input", "weights", p.across_spatial),
        quantize("quantize", "normalizel2", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::u8),
        reorder("output_reorder", "quantize", p.default_format, data_types::f32));

    tolerance = 1;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu,
                        normalize_i8_quantize,
                        ::testing::ValuesIn(std::vector<normalize_test_params>{
                            normalize_test_params{CASE_NORMALIZE_I8_1, false, 2, 3},
                            normalize_test_params{CASE_NORMALIZE_I8_1, true, 2, 3},
                        }));

class normalize_i8_float : public NormalizeFusingTest {};
TEST_P(normalize_i8_float, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/255)),
        normalize("normalizel2", "input", "weights", p.across_spatial),
        scale("scale", "normalizel2", "scale_data"),
        activation("activation", "scale", activation_func::abs),
        reorder("output_reorder", "activation", p.default_format, data_types::f32));

    tolerance = 1e-05f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu,
                        normalize_i8_float,
                        ::testing::ValuesIn(std::vector<normalize_test_params>{
                            normalize_test_params{CASE_NORMALIZE_I8_1, false, 2, 4},
                            normalize_test_params{CASE_NORMALIZE_I8_1, true, 2, 4},
                        }));

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- BatchToSpace cases ----------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
struct batch_to_space_test_params {
    tensor input_size;
    tensor output_size;
    data_types input_type;
    format input_format;
    tensor block_shape;
    tensor crops_begin;
    tensor crops_end;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

#define CASE_BATCH_TO_SPACE_F32_1 {8,  1,  1, 1}, {2, 1,   2,  2}, data_types::f32, format::bfyx,          {1, 1, 2,  2}, {0, 0, 0, 0}, {0, 0, 0, 0}, data_types::f32, format::bfyx
#define CASE_BATCH_TO_SPACE_F32_2 {64, 16, 2, 2}, {2, 112, 4,  4}, data_types::f32, format::b_fs_yx_fsv16, {1, 8, 2,  2}, {0, 8, 0, 0}, {0, 8, 0, 0}, data_types::f32, format::bfyx
#define CASE_BATCH_TO_SPACE_F16_1 {16, 4,  1, 2}, {2, 12,  1,  2}, data_types::f16, format::bfyx,          {1, 4, 2,  1}, {0, 2, 1, 0}, {0, 2, 0, 0}, data_types::f32, format::bfyx
#define CASE_BATCH_TO_SPACE_F16_2 {32, 16, 2, 1}, {1, 16,  32, 2}, data_types::f16, format::b_fs_yx_fsv16, {1, 1, 16, 2}, {0, 0, 0, 0}, {0, 0, 0, 0}, data_types::f32, format::bfyx
#define CASE_BATCH_TO_SPACE_U8_1  {30, 12, 4, 6}, {1, 52,  8,  9}, data_types::u8,  format::bfyx,          {1, 5, 2,  3}, {0, 8, 0, 9}, {0, 0, 0, 0}, data_types::f32, format::bfyx
#define CASE_BATCH_TO_SPACE_U8_2  {24, 32, 4, 5}, {2, 64,  12, 8}, data_types::u8,  format::b_fs_yx_fsv16, {1, 2, 3,  2}, {0, 0, 0, 2}, {0, 0, 0, 0}, data_types::f32, format::bfyx
#define CASE_BATCH_TO_SPACE_I8_1  {32, 1,  3, 4}, {1, 8,   6,  8}, data_types::i8,  format::bfyx,          {1, 8, 2,  2}, {0, 0, 0, 0}, {0, 0, 0, 0}, data_types::f32, format::bfyx
#define CASE_BATCH_TO_SPACE_I8_2  {16, 16, 2, 1}, {2, 32,  4,  2}, data_types::i8,  format::b_fs_yx_fsv16, {1, 2, 2,  2}, {0, 0, 0, 0}, {0, 0, 0, 0}, data_types::f32, format::bfyx

class BatchToSpaceFusingsTest : public ::BaseFusingTest<batch_to_space_test_params> {
public:
    void execute(batch_to_space_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));

        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);

        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(batch_to_space_test_params& p) {
        return layout{p.input_type, p.input_format, p.input_size};
    }

    layout get_per_channel_layout(batch_to_space_test_params& p) {
        return layout{p.default_type, p.default_format, tensor{1, p.output_size.feature[0], 1, 1}};
    }
};

class batch_to_space_quantize_i8 : public BatchToSpaceFusingsTest {};
TEST_P(batch_to_space_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      batch_to_space("batch_to_space", "input", p.block_shape, p.crops_begin, p.crops_end, p.output_size),
                      data("in_low", get_mem(get_per_channel_layout(p), min_random, 0)),
                      data("in_high", get_mem(get_per_channel_layout(p), 1, max_random)),
                      data("out_low", get_mem(get_single_element_layout(p), -128)),
                      data("out_high", get_mem(get_single_element_layout(p), 127)),
                      quantize("quant", "batch_to_space", "in_low", "in_high", "out_low", "out_high", 256, data_types::i8),
                      reorder("reorder_bfyx", "quant", p.default_format, data_types::f32));

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(
    fusings_gpu,
    batch_to_space_quantize_i8,
    ::testing::ValuesIn(std::vector<batch_to_space_test_params> {
        batch_to_space_test_params{CASE_BATCH_TO_SPACE_F32_1, 2, 3},
        batch_to_space_test_params{CASE_BATCH_TO_SPACE_F32_2, 2, 3},
        batch_to_space_test_params{CASE_BATCH_TO_SPACE_F16_1, 2, 3},
        batch_to_space_test_params{CASE_BATCH_TO_SPACE_F16_2, 2, 3},
    }));

class batch_to_space_scale_act_eltwise_quantize_u8 : public BatchToSpaceFusingsTest {};
TEST_P(batch_to_space_scale_act_eltwise_quantize_u8, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      batch_to_space("batch_to_space", "input", p.block_shape, p.crops_begin, p.crops_end, p.output_size),
                      data("scale1_data", get_mem(get_per_channel_layout(p), -0.125f)),
                      scale("scale1", "batch_to_space", "scale1_data"),
                      activation("actv1", "scale1", activation_func::relu),
                      data("eltw_data", get_mem(layout(p.default_type, p.input_format, p.output_size))),
                      eltwise("eltw", {"actv1", "eltw_data"}, eltwise_mode::sum, p.default_type),
                      data("in_low", get_mem(get_per_channel_layout(p), min_random, 0)),
                      data("in_high", get_mem(get_per_channel_layout(p), 1, max_random)),
                      data("out_low", get_mem(get_single_element_layout(p), 0)),
                      data("out_high", get_mem(get_single_element_layout(p), 255)),
                      quantize("quant", "eltw", "in_low", "in_high", "out_low", "out_high", 256, data_types::u8),
                      reorder("reorder_bfyx", "quant", p.default_format, data_types::f32));

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(
    fusings_gpu,
    batch_to_space_scale_act_eltwise_quantize_u8,
    ::testing::ValuesIn(std::vector<batch_to_space_test_params> {
        batch_to_space_test_params{CASE_BATCH_TO_SPACE_F32_1, 2, 6},
        batch_to_space_test_params{CASE_BATCH_TO_SPACE_F32_2, 2, 6},
        batch_to_space_test_params{CASE_BATCH_TO_SPACE_F16_1, 2, 6},
        batch_to_space_test_params{CASE_BATCH_TO_SPACE_F16_2, 2, 6},
        batch_to_space_test_params{CASE_BATCH_TO_SPACE_U8_1, 2, 6},
        batch_to_space_test_params{CASE_BATCH_TO_SPACE_U8_2, 2, 6},
        batch_to_space_test_params{CASE_BATCH_TO_SPACE_I8_1, 2, 6},
        batch_to_space_test_params{CASE_BATCH_TO_SPACE_I8_2, 2, 6},
    }));

class batch_to_space_scale_act_eltw : public BatchToSpaceFusingsTest {};
TEST_P(batch_to_space_scale_act_eltw, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      batch_to_space("batch_to_space", "input", p.block_shape, p.crops_begin, p.crops_end, p.output_size),
                      data("scale1_data", get_mem(get_per_channel_layout(p), -0.125f)),
                      scale("scale1", "batch_to_space", "scale1_data"),
                      activation("actv1", "scale1", activation_func::relu),
                      data("eltw_data", get_mem(layout(p.default_type, p.input_format, p.output_size))),
                      eltwise("eltw", {"actv1", "eltw_data"}, eltwise_mode::sum, p.default_type),
                      reorder("reorder_bfyx", "eltw", p.default_format, data_types::f32));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(
    fusings_gpu,
    batch_to_space_scale_act_eltw,
    ::testing::ValuesIn(std::vector<batch_to_space_test_params> {
        batch_to_space_test_params{CASE_BATCH_TO_SPACE_F32_1, 2, 5},
        batch_to_space_test_params{CASE_BATCH_TO_SPACE_F32_2, 2, 5},
        batch_to_space_test_params{CASE_BATCH_TO_SPACE_F16_1, 2, 5},
        batch_to_space_test_params{CASE_BATCH_TO_SPACE_F16_2, 2, 5},
        batch_to_space_test_params{CASE_BATCH_TO_SPACE_U8_1, 2, 5},
        batch_to_space_test_params{CASE_BATCH_TO_SPACE_U8_2, 2, 5},
        batch_to_space_test_params{CASE_BATCH_TO_SPACE_I8_1, 2, 5},
        batch_to_space_test_params{CASE_BATCH_TO_SPACE_I8_2, 2, 5},
    }));

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- SpaceToBatch cases ----------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

struct space_to_batch_test_params {
    tensor input_size;
    tensor output_size;
    data_types input_type;
    format input_format;
    tensor block_shape;
    tensor pads_begin;
    tensor pads_end;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

#define CASE_SPACE_TO_BATCH_F32_1 {1, 4,  8, 8}, {16, 2, 3, 8}, data_types::f32, format::bfyx,          {1, 2,  4, 1}, {0, 0, 4, 0}, {0, 0, 0, 0}, data_types::f32, format::bfyx
#define CASE_SPACE_TO_BATCH_F32_2 {2, 16, 4, 6}, {24, 4, 4, 3}, data_types::f32, format::b_fs_yx_fsv16, {1, 4,  1, 3}, {0, 0, 0, 0}, {0, 0, 0, 3}, data_types::f32, format::bfyx
#define CASE_SPACE_TO_BATCH_F16_1 {1, 1,  6, 8}, {48, 1, 1, 1}, data_types::f16, format::bfyx,          {1, 1,  6, 8}, {0, 0, 0, 0}, {0, 0, 0, 0}, data_types::f32, format::bfyx
#define CASE_SPACE_TO_BATCH_F16_2 {1, 32, 1, 5}, {20, 4, 1, 4}, data_types::f16, format::b_fs_yx_fsv16, {1, 10, 1, 2}, {0, 8, 0, 0}, {0, 0, 0, 3}, data_types::f32, format::bfyx
#define CASE_SPACE_TO_BATCH_U8_1  {3, 12, 4, 8}, {48, 6, 2, 3}, data_types::u8,  format::bfyx,          {1, 2,  2, 4}, {0, 0, 0, 4}, {0, 0, 0, 0}, data_types::f32, format::bfyx
#define CASE_SPACE_TO_BATCH_U8_2  {2, 16, 3, 6}, {30, 4, 1, 6}, data_types::u8,  format::b_fs_yx_fsv16, {1, 5,  3, 1}, {0, 4, 0, 0}, {0, 0, 0, 0}, data_types::f32, format::bfyx
#define CASE_SPACE_TO_BATCH_I8_1  {1, 2,  8, 1}, {4,  2, 2, 1}, data_types::i8,  format::bfyx,          {1, 1,  4, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}, data_types::f32, format::bfyx
#define CASE_SPACE_TO_BATCH_I8_2  {1, 32, 4, 8}, {48, 2, 6, 3}, data_types::i8,  format::b_fs_yx_fsv16, {1, 16, 1, 3}, {0, 0, 2, 0}, {0, 0, 0, 1}, data_types::f32, format::bfyx

class SpaceToBatchFusingsTest : public ::BaseFusingTest<space_to_batch_test_params> {
public:
    void execute(space_to_batch_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));

        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);

        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(space_to_batch_test_params& p) {
        return layout{p.input_type, p.input_format, p.input_size};
    }

    layout get_per_channel_layout(space_to_batch_test_params& p) {
        return layout{p.default_type, p.default_format, tensor{1, p.output_size.feature[0], 1, 1}};
    }
};

class space_to_batch_quantize_i8 : public SpaceToBatchFusingsTest {};
TEST_P(space_to_batch_quantize_i8, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      space_to_batch("space_to_batch", "input", p.block_shape, p.pads_begin, p.pads_end, p.output_size),
                      data("in_low", get_mem(get_per_channel_layout(p), min_random, 0)),
                      data("in_high", get_mem(get_per_channel_layout(p), 1, max_random)),
                      data("out_low", get_mem(get_single_element_layout(p), -128)),
                      data("out_high", get_mem(get_single_element_layout(p), 127)),
                      quantize("quant", "space_to_batch", "in_low", "in_high", "out_low", "out_high", 256, data_types::i8),
                      reorder("reorder_bfyx", "quant", p.default_format, data_types::f32));

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(
    fusings_gpu,
    space_to_batch_quantize_i8,
    ::testing::ValuesIn(std::vector<space_to_batch_test_params> {
        space_to_batch_test_params{CASE_SPACE_TO_BATCH_F32_1, 2, 3},
        space_to_batch_test_params{CASE_SPACE_TO_BATCH_F32_2, 2, 3},
        space_to_batch_test_params{CASE_SPACE_TO_BATCH_F16_1, 2, 3},
        space_to_batch_test_params{CASE_SPACE_TO_BATCH_F16_2, 2, 3},
    }));

class space_to_batch_scale_act_eltwise_quantize_u8 : public SpaceToBatchFusingsTest {};
TEST_P(space_to_batch_scale_act_eltwise_quantize_u8, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      space_to_batch("space_to_batch", "input", p.block_shape, p.pads_begin, p.pads_end, p.output_size),
                      data("scale1_data", get_mem(get_per_channel_layout(p), -0.125f)),
                      scale("scale1", "space_to_batch", "scale1_data"),
                      activation("actv1", "scale1", activation_func::relu),
                      data("eltw_data", get_mem(layout(p.default_type, p.input_format, p.output_size))),
                      eltwise("eltw", {"actv1", "eltw_data"}, eltwise_mode::sum, p.default_type),
                      data("in_low", get_mem(get_per_channel_layout(p), min_random, 0)),
                      data("in_high", get_mem(get_per_channel_layout(p), 1, max_random)),
                      data("out_low", get_mem(get_single_element_layout(p), 0)),
                      data("out_high", get_mem(get_single_element_layout(p), 255)),
                      quantize("quant", "eltw", "in_low", "in_high", "out_low", "out_high", 256, data_types::u8),
                      reorder("reorder_bfyx", "quant", p.default_format, data_types::f32));

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(
    fusings_gpu,
    space_to_batch_scale_act_eltwise_quantize_u8,
    ::testing::ValuesIn(std::vector<space_to_batch_test_params> {
        space_to_batch_test_params{CASE_SPACE_TO_BATCH_F32_1, 2, 6},
        space_to_batch_test_params{CASE_SPACE_TO_BATCH_F32_2, 2, 6},
        space_to_batch_test_params{CASE_SPACE_TO_BATCH_F16_1, 2, 6},
        space_to_batch_test_params{CASE_SPACE_TO_BATCH_F16_2, 2, 6},
        space_to_batch_test_params{CASE_SPACE_TO_BATCH_U8_1, 2, 6},
        space_to_batch_test_params{CASE_SPACE_TO_BATCH_U8_2, 2, 6},
        space_to_batch_test_params{CASE_SPACE_TO_BATCH_I8_1, 2, 6},
        space_to_batch_test_params{CASE_SPACE_TO_BATCH_I8_2, 2, 6},
    }));


class space_to_batch_scale_act_eltw : public SpaceToBatchFusingsTest {};
TEST_P(space_to_batch_scale_act_eltw, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      space_to_batch("space_to_batch", "input", p.block_shape, p.pads_begin, p.pads_end, p.output_size),
                      data("scale1_data", get_mem(get_per_channel_layout(p), -0.125f)),
                      scale("scale1", "space_to_batch", "scale1_data"),
                      activation("actv1", "scale1", activation_func::relu),
                      data("eltw_data", get_mem(layout(p.default_type, p.input_format, p.output_size))),
                      eltwise("eltw", {"actv1", "eltw_data"}, eltwise_mode::sum, p.default_type),
                      reorder("reorder_bfyx", "eltw", p.default_format, data_types::f32));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(
    fusings_gpu,
    space_to_batch_scale_act_eltw,
    ::testing::ValuesIn(std::vector<space_to_batch_test_params> {
        space_to_batch_test_params{CASE_SPACE_TO_BATCH_F32_1, 2, 5},
        space_to_batch_test_params{CASE_SPACE_TO_BATCH_F32_2, 2, 5},
        space_to_batch_test_params{CASE_SPACE_TO_BATCH_F16_1, 2, 5},
        space_to_batch_test_params{CASE_SPACE_TO_BATCH_F16_2, 2, 5},
        space_to_batch_test_params{CASE_SPACE_TO_BATCH_U8_1, 2, 5},
        space_to_batch_test_params{CASE_SPACE_TO_BATCH_U8_2, 2, 5},
        space_to_batch_test_params{CASE_SPACE_TO_BATCH_I8_1, 2, 5},
        space_to_batch_test_params{CASE_SPACE_TO_BATCH_I8_2, 2, 5},
    }));

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- Eltwise cases -------------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
struct eltwise_test_params {
    tensor input_size;
    data_types input_type;
    data_types input_type2;
    format input_format;
    data_types default_type;
    format default_format;
    eltwise_mode mode;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

#define CASE_ELTWISE_FP32_1         {2, 16, 4, 4}, data_types::f32, data_types::f32, format::bfyx,           data_types::f32,  format::bfyx,             eltwise_mode::sum
#define CASE_ELTWISE_FP32_2         {2, 16, 4, 4}, data_types::f32, data_types::f32, format::bfzyx,          data_types::f32,  format::bfzyx,            eltwise_mode::sum
#define CASE_ELTWISE_FP32_3         {2, 32, 4, 8}, data_types::f32, data_types::f32, format::b_fs_yx_fsv16,  data_types::f32,  format::b_fs_yx_fsv16,    eltwise_mode::sum
#define CASE_ELTWISE_FP32_4         {2, 16, 4, 4}, data_types::f32, data_types::f32, format::bfwzyx,         data_types::f32,  format::bfwzyx,           eltwise_mode::sum
#define CASE_ELTWISE_FP16_1         {2, 16, 4, 4}, data_types::f16, data_types::f16, format::bfyx,           data_types::f16,  format::bfyx,             eltwise_mode::sum
#define CASE_ELTWISE_FP16_2         {2, 16, 4, 4}, data_types::f16, data_types::f16, format::bfzyx,          data_types::f16,  format::bfzyx,            eltwise_mode::sum
#define CASE_ELTWISE_FP16_3         {2, 32, 4, 8}, data_types::f16, data_types::f16, format::b_fs_yx_fsv16,  data_types::f16,  format::b_fs_yx_fsv16,    eltwise_mode::sum
#define CASE_ELTWISE_FP16_4         {3, 32, 4, 4}, data_types::f16, data_types::f16, format::fs_b_yx_fsv32,  data_types::f16,  format::fs_b_yx_fsv32,    eltwise_mode::sum
#define CASE_ELTWISE_I8_1           {2, 16, 4, 4}, data_types::i8,  data_types::i8,  format::bfyx,           data_types::f32,  format::bfyx,             eltwise_mode::sum
#define CASE_ELTWISE_I8_2           {2, 16, 4, 4}, data_types::i8,  data_types::i8,  format::bfzyx,          data_types::f32,  format::bfzyx,            eltwise_mode::sum
#define CASE_ELTWISE_I8_3           {2, 16, 4, 4}, data_types::i8,  data_types::i8,  format::b_fs_yx_fsv16,  data_types::f32,  format::b_fs_yx_fsv16,    eltwise_mode::sum
#define CASE_ELTWISE_U8_1           {2, 16, 4, 4}, data_types::u8,  data_types::u8,  format::bfyx,           data_types::f32,  format::bfyx,             eltwise_mode::sum
#define CASE_ELTWISE_U8_2           {2, 16, 4, 4}, data_types::u8,  data_types::u8,  format::bfzyx,          data_types::f32,  format::bfzyx,            eltwise_mode::sum
#define CASE_ELTWISE_U8_3           {2, 16, 4, 4}, data_types::u8,  data_types::u8,  format::b_fs_yx_fsv16,  data_types::f32,  format::b_fs_yx_fsv16,    eltwise_mode::sum
#define CASE_ELTWISE_FP32_FP16_1    {2, 16, 4, 4}, data_types::f32, data_types::f16, format::bfyx,           data_types::f32,  format::bfyx,             eltwise_mode::sum
#define CASE_ELTWISE_FP32_FP16_2    {2, 16, 4, 4}, data_types::f32, data_types::f16, format::bfzyx,          data_types::f32,  format::bfzyx,            eltwise_mode::sum
#define CASE_ELTWISE_FP32_FP16_3    {2, 32, 4, 4}, data_types::f32, data_types::f16, format::b_fs_yx_fsv16,  data_types::f32,  format::b_fs_yx_fsv16,    eltwise_mode::sum
#define CASE_ELTWISE_FP16_FP32_1    {2, 16, 4, 4}, data_types::f16, data_types::f32, format::bfyx,           data_types::f16,  format::bfyx,             eltwise_mode::sum
#define CASE_ELTWISE_FP16_FP32_2    {2, 16, 4, 4}, data_types::f16, data_types::f32, format::bfzyx,          data_types::f16,  format::bfzyx,            eltwise_mode::sum
#define CASE_ELTWISE_FP16_FP32_3    {2, 32, 4, 4}, data_types::f16, data_types::f32, format::b_fs_yx_fsv16,  data_types::f16,  format::b_fs_yx_fsv16,    eltwise_mode::sum
#define CASE_ELTWISE_I8_FP16_1      {2, 16, 4, 4}, data_types::i8,  data_types::f16, format::bfyx,           data_types::f32,  format::bfyx,             eltwise_mode::sum
#define CASE_ELTWISE_I8_FP16_2      {2, 16, 4, 4}, data_types::i8,  data_types::f16, format::bfzyx,          data_types::f32,  format::bfzyx,            eltwise_mode::sum
#define CASE_ELTWISE_I8_FP16_3      {2, 32, 4, 4}, data_types::i8,  data_types::f16, format::b_fs_yx_fsv16,  data_types::f32,  format::b_fs_yx_fsv16,    eltwise_mode::sum
#define CASE_ELTWISE_I8_FP32_1      {2, 16, 4, 4}, data_types::i8,  data_types::f32, format::bfyx,           data_types::f16,  format::bfyx,             eltwise_mode::sum
#define CASE_ELTWISE_I8_FP32_2      {2, 16, 4, 4}, data_types::i8,  data_types::f32, format::bfzyx,          data_types::f16,  format::bfzyx,            eltwise_mode::sum
#define CASE_ELTWISE_I8_FP32_3      {2, 32, 4, 4}, data_types::i8,  data_types::f32, format::b_fs_yx_fsv16,  data_types::f16,  format::b_fs_yx_fsv16,    eltwise_mode::sum
#define CASE_ELTWISE_U8_FP16_1      {2, 16, 4, 4}, data_types::u8,  data_types::f16, format::bfyx,           data_types::f32,  format::bfyx,             eltwise_mode::sum
#define CASE_ELTWISE_U8_FP16_2      {2, 16, 4, 4}, data_types::u8,  data_types::f16, format::bfzyx,          data_types::f32,  format::bfzyx,            eltwise_mode::sum
#define CASE_ELTWISE_U8_FP16_3      {2, 32, 4, 4}, data_types::u8,  data_types::f16, format::b_fs_yx_fsv16,  data_types::f32,  format::b_fs_yx_fsv16,    eltwise_mode::sum
#define CASE_ELTWISE_U8_FP32_1      {2, 16, 4, 4}, data_types::u8,  data_types::f32, format::bfyx,           data_types::f16,  format::bfyx,             eltwise_mode::sum
#define CASE_ELTWISE_U8_FP32_2      {2, 16, 4, 4}, data_types::u8,  data_types::f32, format::bfzyx,          data_types::f16,  format::bfzyx,            eltwise_mode::sum
#define CASE_ELTWISE_U8_FP32_3      {2, 32, 4, 4}, data_types::u8,  data_types::f32, format::b_fs_yx_fsv16,  data_types::f16,  format::b_fs_yx_fsv16,    eltwise_mode::sum

#define CASE_ELTWISE_FP32_5         {1,  5, 4, 4}, data_types::f32, data_types::f32, format::b_fs_yx_fsv4,  data_types::f32,  format::b_fs_yx_fsv4,    eltwise_mode::sum
#define CASE_ELTWISE_FP32_6         {2, 32, 4, 8}, data_types::f32, data_types::f32, format::b_fs_yx_fsv4,  data_types::f32,  format::b_fs_yx_fsv4,    eltwise_mode::sum
#define CASE_ELTWISE_FP16_5         {2, 32, 4, 8}, data_types::f16, data_types::f16, format::b_fs_yx_fsv4,  data_types::f16,  format::b_fs_yx_fsv4,    eltwise_mode::sum
#define CASE_ELTWISE_I8_4           {2, 16, 4, 4}, data_types::i8,  data_types::i8,  format::b_fs_yx_fsv4,  data_types::f32,  format::b_fs_yx_fsv4,    eltwise_mode::sum
#define CASE_ELTWISE_U8_4           {2, 16, 4, 4}, data_types::u8,  data_types::u8,  format::b_fs_yx_fsv4,  data_types::f32,  format::b_fs_yx_fsv4,    eltwise_mode::sum


class EltwiseFusingTest : public ::BaseFusingTest<eltwise_test_params> {
public:
    void execute(eltwise_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        auto input_prim2 = get_mem(get_input_layout2(p));

        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);

        auto inputs = network_fused.get_input_ids();
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);
        if (std::find(inputs.begin(), inputs.end(), "input2") != inputs.end()) {
            network_fused.set_input_data("input2", input_prim2);
            network_not_fused.set_input_data("input2", input_prim2);
        }

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(eltwise_test_params& p) { return layout{p.input_type, p.input_format, p.input_size}; }
    layout get_input_layout2(eltwise_test_params& p) { return layout{p.input_type2, p.input_format, p.input_size}; }

    layout get_per_channel_layout(eltwise_test_params& p) {
        return layout{p.default_type, p.default_format, tensor{1, p.input_size.feature[0], 1, 1}};
    }
};

class eltwise_quantize : public EltwiseFusingTest {};
TEST_P(eltwise_quantize, u8) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      input_layout("input2", get_input_layout2(p)),
                      eltwise("eltwise", {"input", "input2"}, p.mode, p.default_type),
                      data("in_lo", get_mem(get_single_element_layout(p), min_random, 0)),
                      data("in_hi", get_mem(get_single_element_layout(p), 1, max_random)),
                      data("out_lo", get_mem(get_single_element_layout(p), 0)),
                      data("out_hi", get_mem(get_single_element_layout(p), 255)),
                      quantize("quantize", "eltwise", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::u8),
                      reorder("out", "quantize", p.default_format, data_types::f32));

    tolerance = 1.f;
    execute(p);
}

TEST_P(eltwise_quantize, i8_per_channel) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      input_layout("input2", get_input_layout2(p)),
                      eltwise("eltwise", {"input", "input2"}, p.mode, p.default_type),
                      data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                      data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                      data("out_lo", get_mem(get_single_element_layout(p), -128)),
                      data("out_hi", get_mem(get_single_element_layout(p), 127)),
                      quantize("quantize", "eltwise", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::i8),
                      reorder("out", "quantize", p.default_format, data_types::f32));

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu,
                        eltwise_quantize,
                        ::testing::ValuesIn(std::vector<eltwise_test_params>{
                            eltwise_test_params{CASE_ELTWISE_FP16_1, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_FP16_2, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_FP16_3, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_FP32_1, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_FP32_2, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_FP32_3, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_FP32_FP16_1, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_FP32_FP16_2, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_FP32_FP16_3, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_FP16_FP32_1, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_FP16_FP32_2, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_FP16_FP32_3, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_I8_FP32_1, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_I8_FP32_2, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_I8_FP32_3, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_U8_FP32_1, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_U8_FP32_2, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_U8_FP32_3, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_I8_FP16_1, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_I8_FP16_2, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_I8_FP16_3, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_U8_FP16_1, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_U8_FP16_2, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_U8_FP16_3, 3, 4},
                            // fsv4
                            eltwise_test_params{CASE_ELTWISE_FP16_5, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_FP32_5, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_FP32_6, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_I8_4, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_U8_4, 3, 4},
                        }));

class eltwise_const_path : public EltwiseFusingTest {};
TEST_P(eltwise_const_path, not_fuse_to_const_eltwise) {
    auto p = GetParam();
    create_topologies(data("const1", get_mem(get_input_layout2(p), -10, 10)),
                      data("const2", get_mem(get_input_layout2(p), -10, 10)),
                      input_layout("input", get_input_layout2(p)),
                      eltwise("eltwise", {"const1", "const2"}, p.mode, p.default_type),
                      eltwise("add", {"eltwise", "input"}, eltwise_mode::sum),
                      activation("activation", "add", activation_func::negative),
                      reorder("out", "activation", p.default_format, data_types::f32));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu,
                        eltwise_const_path,
                        ::testing::ValuesIn(std::vector<eltwise_test_params>{
                            eltwise_test_params{CASE_ELTWISE_FP16_3, 2, 3},
                            eltwise_test_params{CASE_ELTWISE_FP32_3, 2, 3},
                            eltwise_test_params{CASE_ELTWISE_FP32_5, 2, 3},
                            eltwise_test_params{CASE_ELTWISE_FP32_6, 2, 3},
                            eltwise_test_params{CASE_ELTWISE_I8_4, 2, 3},
                            eltwise_test_params{CASE_ELTWISE_U8_4, 2, 3},
                        }));

class eltwise_fp32_fsv16 : public EltwiseFusingTest {};
TEST_P(eltwise_fp32_fsv16, add) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      input_layout("input2", get_input_layout2(p)),
                      data("add_data", get_mem(get_per_channel_layout(p), -10, 10)),
                      eltwise("eltwise", {"input", "input2"}, p.mode, p.default_type),
                      eltwise("add", {"eltwise", "add_data"}, eltwise_mode::sum),
                      activation("activation", "add", activation_func::negative),
                      reorder("out", "activation", p.default_format, data_types::f32));

    implementation_desc eltw_impl = { format::b_fs_yx_fsv16, "eltwise_b_fs_yx_fsv16" };
    bo_fused.set_option(build_option::force_implementations({ {"eltwise", eltw_impl} }));

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(eltwise_fp32_fsv16, add_per_element) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      input_layout("input2", get_input_layout2(p)),
                      data("add_data", get_mem(get_input_layout(p), -10, 10)),
                      eltwise("eltwise", {"input", "input2"}, p.mode, p.default_type),
                      eltwise("add", {"eltwise", "add_data"}, eltwise_mode::sum),
                      activation("activation", "add", activation_func::negative),
                      reorder("out", "activation", p.default_format, data_types::f32));

    implementation_desc eltw_impl = { format::b_fs_yx_fsv16, "eltwise_b_fs_yx_fsv16" };
    bo_fused.set_option(build_option::force_implementations({ {"eltwise", eltw_impl} }));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu,
                        eltwise_fp32_fsv16,
                        ::testing::ValuesIn(std::vector<eltwise_test_params>{
                            eltwise_test_params{CASE_ELTWISE_FP16_3, 3, 5},
                            eltwise_test_params{CASE_ELTWISE_FP32_3, 3, 5},
                        }));

class eltwise_fp32_fsv32 : public EltwiseFusingTest {};
TEST_P(eltwise_fp32_fsv32, add) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      input_layout("input2", get_input_layout2(p)),
                      data("add_data", get_mem(get_per_channel_layout(p), -10, 10)),
                      eltwise("eltwise", {"input", "input2"}, p.mode, p.default_type),
                      eltwise("add", {"eltwise", "add_data"}, eltwise_mode::sum),
                      activation("activation", "add", activation_func::negative),
                      reorder("out", "activation", p.default_format, data_types::f32));

    implementation_desc eltw_impl = { format::fs_b_yx_fsv32, "eltwise_fs_b_yx_fsv32" };
    bo_fused.set_option(build_option::force_implementations({ {"eltwise", eltw_impl} }));

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(eltwise_fp32_fsv32, add_per_element) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      input_layout("input2", get_input_layout2(p)),
                      data("add_data", get_mem(get_input_layout(p), -10, 10)),
                      eltwise("eltwise", {"input", "input2"}, p.mode, p.default_type),
                      eltwise("add", {"eltwise", "add_data"}, eltwise_mode::sum),
                      activation("activation", "add", activation_func::negative),
                      reorder("out", "activation", p.default_format, data_types::f32));

    implementation_desc eltw_impl = { format::fs_b_yx_fsv32, "eltwise_fs_b_yx_fsv32" };
    bo_fused.set_option(build_option::force_implementations({ {"eltwise", eltw_impl} }));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu,
                        eltwise_fp32_fsv32,
                        ::testing::ValuesIn(std::vector<eltwise_test_params>{
                            // There's no optimized eltwise kernel yet for fsv32 layout that supports fused_ops
                            // So only activation is fused via legacy mechanism
                            eltwise_test_params{CASE_ELTWISE_FP16_4, 4, 5},
                        }));

class eltwise_fp32_fsv4 : public EltwiseFusingTest {};
TEST_P(eltwise_fp32_fsv4, add) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      input_layout("input2", get_input_layout2(p)),
                      data("add_data", get_mem(get_per_channel_layout(p), -10, 10)),
                      eltwise("eltwise", {"input", "input2"}, p.mode, p.default_type),
                      eltwise("add", {"eltwise", "add_data"}, eltwise_mode::sum),
                      activation("activation", "add", activation_func::negative),
                      reorder("out", "activation", p.default_format, data_types::f32));

    implementation_desc eltw_impl = { format::b_fs_yx_fsv4, "eltwise_b_fs_yx_fsv4" };
    bo_fused.set_option(build_option::force_implementations({ {"eltwise", eltw_impl} }));

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(eltwise_fp32_fsv4, add_per_element) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      input_layout("input2", get_input_layout2(p)),
                      data("add_data", get_mem(get_input_layout(p), -10, 10)),
                      eltwise("eltwise", {"input", "input2"}, p.mode, p.default_type),
                      eltwise("add", {"eltwise", "add_data"}, eltwise_mode::sum),
                      activation("activation", "add", activation_func::negative),
                      reorder("out", "activation", p.default_format, data_types::f32));

    implementation_desc eltw_impl = { format::b_fs_yx_fsv4, "eltwise_b_fs_yx_fsv4" };
    bo_fused.set_option(build_option::force_implementations({ {"eltwise", eltw_impl} }));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu,
                        eltwise_fp32_fsv4,
                        ::testing::ValuesIn(std::vector<eltwise_test_params>{
                            eltwise_test_params{CASE_ELTWISE_FP32_5, 3, 5},
                            eltwise_test_params{CASE_ELTWISE_FP32_6, 3, 5},
                            eltwise_test_params{CASE_ELTWISE_I8_4,   3, 5},
                            eltwise_test_params{CASE_ELTWISE_U8_4,   3, 5},
                        }));

//
class eltwise_fp32_fused_prims : public EltwiseFusingTest {};
TEST_P(eltwise_fp32_fused_prims, scale_activation) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      input_layout("input2", get_input_layout2(p)),
                      data("scale_data", get_mem(get_per_channel_layout(p), -10, 10)),
                      eltwise("eltwise", {"input", "input2"}, p.mode, p.default_type),
                      scale("scale", "eltwise", "scale_data"),
                      activation("activation", "scale", activation_func::abs),
                      reorder("out", "activation", p.default_format, data_types::f32));

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(eltwise_fp32_fused_prims, eltwise_activation) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      input_layout("input2", get_input_layout2(p)),
                      data("eltwise_data", get_mem(get_input_layout2(p), -10, 10)),
                      eltwise("eltwise1", {"input", "input2"}, p.mode, data_types::f32),
                      eltwise("eltwise2", {"eltwise1", "eltwise_data"}, eltwise_mode::prod, p.default_type),
                      activation("activation", "eltwise2", activation_func::abs),
                      reorder("out", "activation", p.default_format, data_types::f32));

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(eltwise_fp32_fused_prims, eltwise_activation_with_broadcast) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      input_layout("input2", get_input_layout2(p)),
                      data("eltwise_data", get_mem(get_per_channel_layout(p), -10, 10)),
                      eltwise("eltwise1", {"input", "input2"}, p.mode, p.default_type),
                      eltwise("eltwise2", {"eltwise1", "eltwise_data"}, eltwise_mode::prod, p.default_type),
                      activation("activation", "eltwise2", activation_func::abs),
                      reorder("out", "activation", p.default_format, data_types::f32));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu,
                        eltwise_fp32_fused_prims,
                        ::testing::ValuesIn(std::vector<eltwise_test_params>{
                            eltwise_test_params{CASE_ELTWISE_FP16_1, 3, 5},
                            eltwise_test_params{CASE_ELTWISE_FP16_2, 3, 5},
                            eltwise_test_params{CASE_ELTWISE_FP16_3, 3, 5},
                            eltwise_test_params{CASE_ELTWISE_FP32_1, 3, 5},
                            eltwise_test_params{CASE_ELTWISE_FP32_2, 3, 5},
                            eltwise_test_params{CASE_ELTWISE_FP32_3, 3, 5},
                            eltwise_test_params{CASE_ELTWISE_FP32_FP16_1, 3, 5},
                            eltwise_test_params{CASE_ELTWISE_FP32_FP16_2, 3, 5},
                            eltwise_test_params{CASE_ELTWISE_FP32_FP16_3, 3, 5},
                            eltwise_test_params{CASE_ELTWISE_FP16_FP32_1, 3, 5},
                            eltwise_test_params{CASE_ELTWISE_FP16_FP32_2, 3, 5},
                            eltwise_test_params{CASE_ELTWISE_FP16_FP32_3, 3, 5},
                            eltwise_test_params{CASE_ELTWISE_I8_FP32_1, 3, 5},
                            eltwise_test_params{CASE_ELTWISE_I8_FP32_2, 3, 5},
                            eltwise_test_params{CASE_ELTWISE_I8_FP32_3, 3, 5},
                            eltwise_test_params{CASE_ELTWISE_U8_FP32_1, 3, 5},
                            eltwise_test_params{CASE_ELTWISE_U8_FP32_2, 3, 5},
                            eltwise_test_params{CASE_ELTWISE_U8_FP32_3, 3, 5},
                            eltwise_test_params{CASE_ELTWISE_I8_FP16_1, 3, 5},
                            eltwise_test_params{CASE_ELTWISE_I8_FP16_2, 3, 5},
                            eltwise_test_params{CASE_ELTWISE_I8_FP16_3, 3, 5},
                            eltwise_test_params{CASE_ELTWISE_U8_FP16_1, 3, 5},
                            eltwise_test_params{CASE_ELTWISE_U8_FP16_2, 3, 5},
                            eltwise_test_params{CASE_ELTWISE_U8_FP16_3, 3, 5},
                            // fsv4
                            eltwise_test_params{CASE_ELTWISE_FP32_5, 3, 5},
                            eltwise_test_params{CASE_ELTWISE_FP32_6, 3, 5},
                            eltwise_test_params{CASE_ELTWISE_I8_4, 3, 5},
                            eltwise_test_params{CASE_ELTWISE_U8_4, 3, 5},
                        }));

class eltwise_fp32_scale : public EltwiseFusingTest {};
TEST_P(eltwise_fp32_scale, 6d) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      input_layout("input2", get_input_layout2(p)),
                      data("scale_data", get_mem(get_per_channel_layout(p), -10, 10)),
                      eltwise("eltwise", {"input", "input2"}, p.mode, p.default_type),
                      scale("scale", "eltwise", "scale_data"),
                      reorder("out", "scale", p.default_format, data_types::f32));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu,
                        eltwise_fp32_scale,
                        ::testing::ValuesIn(std::vector<eltwise_test_params>{
                            eltwise_test_params{CASE_ELTWISE_FP32_4, 3, 4},
                        }));

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- Scale cases ------------------------------------------------ */
/* ----------------------------------------------------------------------------------------------------- */
struct scale_test_params {
    tensor input_size;
    data_types input_type;
    format input_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

// Scale uses the same kernel as eltwise primitive, so the kernel is well covered by the eltwise tests above
// So here we can just check that fused scale kernel is constructed correctly (inputs are set correctly, fused precision is propagated, etc)
// and fusing conditions in the graph are correct
#define CASE_SCALE_FP32_1         {2, 16, 4, 4}, data_types::f32, format::bfyx,           data_types::f32,  format::bfyx
#define CASE_SCALE_FP32_2         {2, 16, 4, 4}, data_types::f32, format::bfzyx,          data_types::f32,  format::bfzyx
#define CASE_SCALE_FP32_3         {2, 16, 4, 4}, data_types::f32, format::b_fs_yx_fsv16,  data_types::f32,  format::b_fs_yx_fsv16

class ScaleFusingTest : public ::BaseFusingTest<scale_test_params> {
public:
    void execute(scale_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));

        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);

        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(scale_test_params& p) { return layout{p.input_type, p.input_format, p.input_size}; }

    layout get_per_channel_layout(scale_test_params& p) {
        return layout{p.default_type, p.default_format, tensor{1, p.input_size.feature[0], 1, 1}};
    }
};

class scale_basic : public ScaleFusingTest {};
TEST_P(scale_basic, no_bias_act_eltwise) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      data("scale_data", get_mem(get_per_channel_layout(p), -10, 10)),
                      scale("scale", "input", "scale_data"),
                      activation("activation", "scale", activation_func::negative),
                      data("eltwise_data", get_mem(get_per_channel_layout(p), -10, 10)),
                      eltwise("eltwise", {"activation", "eltwise_data"}, eltwise_mode::prod, p.default_type),
                      reorder("out", "eltwise", p.default_format, data_types::f32));

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(scale_basic, bias_act_eltwise) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      data("scale_data", get_mem(get_per_channel_layout(p), -10, 10)),
                      data("bias_data", get_mem(get_per_channel_layout(p), -10, 10)),
                      scale("scale", "input", "scale_data", "bias_data"),
                      activation("activation", "scale", activation_func::negative),
                      data("eltwise_data", get_mem(get_per_channel_layout(p), -10, 10)),
                      eltwise("eltwise", {"activation", "eltwise_data"}, eltwise_mode::prod, p.default_type),
                      reorder("out", "eltwise", p.default_format, data_types::f32));

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(scale_basic, bias_act_scale) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      data("scale_data", get_mem(get_per_channel_layout(p), -10, 10)),
                      data("bias_data", get_mem(get_per_channel_layout(p), -10, 10)),
                      scale("scale", "input", "scale_data", "bias_data"),
                      activation("activation", "scale", activation_func::negative),
                      data("scale_data2", get_mem(get_per_channel_layout(p), -10, 10)),
                      scale("scale2", "activation", "scale_data2"),
                      reorder("out", "scale2", p.default_format, data_types::f32));

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(scale_basic, bias_act_quantize) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      data("scale_data", get_mem(get_per_channel_layout(p), -10, 10)),
                      data("bias_data", get_mem(get_per_channel_layout(p), -10, 10)),
                      scale("scale", "input", "scale_data", "bias_data"),
                      activation("activation", "scale", activation_func::negative),
                      data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                      data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                      data("out_lo", get_mem(get_single_element_layout(p), -128)),
                      data("out_hi", get_mem(get_single_element_layout(p), 127)),
                      quantize("quantize", "activation", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::i8),
                      reorder("out", "quantize", p.default_format, data_types::f32));

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu,
                        scale_basic,
                        ::testing::ValuesIn(std::vector<scale_test_params>{
                            scale_test_params{CASE_SCALE_FP32_1, 2, 4},
                            scale_test_params{CASE_SCALE_FP32_2, 2, 4},
                            scale_test_params{CASE_SCALE_FP32_3, 2, 4},
                        }));

class eltwise_no_pitches_same_dims_quantize : public EltwiseFusingTest {};
TEST_P(eltwise_no_pitches_same_dims_quantize, quantize_f32_output) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      input_layout("input2", get_input_layout2(p)),
                      eltwise("eltwise", {"input", "input2"}, p.mode, p.default_type),
                      data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                      data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                      data("out_lo", get_mem(get_single_element_layout(p), -128)),
                      data("out_hi", get_mem(get_single_element_layout(p), 127)),
                      quantize("quantize", "eltwise", "in_lo", "in_hi", "out_lo", "out_hi", 256, p.input_type),
                      reorder("out", "quantize", p.default_format, data_types::f32));

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu,
                        eltwise_no_pitches_same_dims_quantize,
                        ::testing::ValuesIn(std::vector<eltwise_test_params>{
                            eltwise_test_params{CASE_ELTWISE_FP16_1, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_FP16_2, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_FP16_3, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_FP32_1, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_FP32_2, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_FP32_3, 3, 4},
                        }));

class eltwise_activation : public EltwiseFusingTest {};
TEST_P(eltwise_activation, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      input_layout("input2", get_input_layout2(p)),
                      eltwise("eltwise", {"input", "input2"}, p.mode, p.default_type),
                      activation("activation", "eltwise", activation_func::relu, {6.0f, 0.0f}),
                      reorder("out", "activation", p.default_format, data_types::f32));

    tolerance = 1e-5f;
    execute(p);
}

TEST_P(eltwise_activation, fp16_out) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      input_layout("input2", get_input_layout2(p)),
                      eltwise("eltwise", {"input", "input2"}, p.mode, data_types::f16),
                      activation("activation", "eltwise", activation_func::relu, {6.0f, 0.0f}),
                      reorder("out", "activation", p.default_format, data_types::f32));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu,
                        eltwise_activation,
                        ::testing::ValuesIn(std::vector<eltwise_test_params>{
                            eltwise_test_params{CASE_ELTWISE_FP16_1, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_FP16_2, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_FP16_3, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_FP32_1, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_FP32_2, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_FP32_3, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_FP32_FP16_1, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_FP32_FP16_2, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_FP32_FP16_3, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_FP16_FP32_1, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_FP16_FP32_2, 3, 4},
                            eltwise_test_params{CASE_ELTWISE_FP16_FP32_3, 3, 4}
                        }));

/* ----------------------------------------------------------------------------------------------------- */
/* ---------------------------------------- Reduce cases ----------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
struct reduce_test_params {
    cldnn::tensor in_shape;
    cldnn::tensor out_shape;
    cldnn::data_types data_type;
    cldnn::format input_format;
    data_types default_type;
    cldnn::format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
    cldnn::reduce_mode reduce_mode;
    std::vector<uint16_t> reduce_axes;
    bool keep_dims;
    std::string kernel_name;
};

#define CASE_REDUCE_F32_0 {3, 7, 5, 7}, {3, 7, 5, 7}, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_REDUCE_F32_1 {3, 7, 5, 7}, {3, 7, 5, 7}, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_REDUCE_F32_2 {2, 4, 8, 4, 4}, {2, 4, 8, 4, 4}, data_types::f32, format::bfzyx, data_types::f32, format::bfyx
#define CASE_REDUCE_F32_3 {16, 16, 16, 8, 8, 8}, {16, 16, 16, 8, 8, 8}, data_types::f32, format::bfwzyx, data_types::f32, format::bfyx
#define CASE_REDUCE_F32_4 {2, 8, 4, 4}, {2, 8, 4, 4}, data_types::f32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx

#define CASE_REDUCE_F16_0 {3, 7, 5, 7}, {3, 7, 5, 7}, data_types::f16, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_REDUCE_F16_1 {2, 8, 4, 4}, {2, 8, 4, 4}, data_types::f16, format::bfyx, data_types::f32, format::bfyx
#define CASE_REDUCE_F16_2 {2, 4, 8, 4, 4}, {2, 4, 8, 4, 4}, data_types::f16, format::bfzyx, data_types::f32, format::bfyx
#define CASE_REDUCE_F16_3 {3, 5, 3, 5, 7, 7}, {3, 5, 3, 5, 7, 7}, data_types::f16, format::bfwzyx, data_types::f32, format::bfyx
#define CASE_REDUCE_F16_4 {2, 8, 4, 4}, {2, 8, 4, 4}, data_types::f16, format::b_fs_yx_fsv16, data_types::f32, format::bfyx

#define CASE_REDUCE_I32_0 {3, 7, 5, 7}, {3, 7, 5, 7}, data_types::i32, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_REDUCE_I32_1 {2, 8, 4, 4}, {2, 8, 4, 4}, data_types::i32, format::bfyx, data_types::f32, format::bfyx
#define CASE_REDUCE_I32_2 {2, 4, 8, 4, 4}, {2, 4, 8, 4, 4}, data_types::i32, format::bfzyx, data_types::f32, format::bfyx
#define CASE_REDUCE_I32_3 {3, 5, 3, 5, 7, 7}, {3, 5, 3, 5, 7, 7}, data_types::i32, format::bfwzyx, data_types::f32, format::bfyx
#define CASE_REDUCE_I32_4 {3, 5, 3, 5, 7, 7}, {3, 5, 3, 5, 7, 7}, data_types::i32, format::bfwzyx, data_types::f32, format::bfyx

#define CASE_REDUCE_I8_0 {3, 7, 5, 7}, {3, 7, 5, 7}, data_types::i8, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_REDUCE_I8_1 {2, 8, 4, 4}, {2, 8, 4, 4}, data_types::i8, format::bfyx, data_types::f32, format::bfyx
#define CASE_REDUCE_I8_2 {2, 4, 8, 4, 4}, {2, 4, 8, 4, 4}, data_types::i8, format::bfzyx, data_types::f32, format::bfyx
#define CASE_REDUCE_I8_3 {3, 5, 3, 5, 7, 7}, {3, 5, 3, 5, 7, 7}, data_types::i8, format::bfwzyx, data_types::f32, format::bfyx
#define CASE_REDUCE_I8_4 {2, 8, 4, 4}, {2, 8, 4, 4}, data_types::i8, format::b_fs_yx_fsv16, data_types::f32, format::bfyx

#define CASE_REDUCE_U8_0 {3, 7, 5, 7}, {3, 7, 5, 7},data_types::u8, format::b_fs_yx_fsv16, data_types::f32, format::bfyx
#define CASE_REDUCE_U8_1 {2, 8, 4, 4}, {2, 8, 4, 4}, data_types::u8, format::bfyx, data_types::f32, format::bfyx
#define CASE_REDUCE_U8_2 {2, 4, 8, 4, 4}, {2, 4, 8, 4, 4}, data_types::u8, format::bfzyx, data_types::f32, format::bfyx
#define CASE_REDUCE_U8_3 {3, 5, 3, 5, 7, 7}, {3, 5, 3, 5, 7, 7}, data_types::u8, format::bfwzyx, data_types::f32, format::bfyx
#define CASE_REDUCE_U8_4 {2, 8, 4, 4}, {2, 8, 4, 4}, data_types::u8, format::b_fs_yx_fsv16, data_types::f32, format::bfyx


class ReduceFusingTest : public ::BaseFusingTest<reduce_test_params> {
public:
    void execute(reduce_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));

        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);

        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
    }

    void update_out_shape(reduce_test_params& p) {
       for (auto& axis : p.reduce_axes) {
            switch (axis) {
                case 0:  // batch
                    p.out_shape.batch[0] = 1;
                    break;
                case 1:  // feature
                    p.out_shape.feature[0] = 1;
                    break;
                case 2:  // x
                    p.out_shape.spatial[0] = 1;
                    break;
                case 3:  // y
                    p.out_shape.spatial[1] = 1;
                    break;
                case 4:  // z
                    p.out_shape.spatial[2] = 1;
                    break;
                case 5:  // w
                    p.out_shape.spatial[3] = 1;
                    break;
            }
        }
    }

    layout get_input_layout(reduce_test_params& p) { return layout{p.data_type, p.input_format, p.in_shape}; }
    layout get_per_channel_layout(reduce_test_params& p) {
        return layout{p.default_type, p.default_format, tensor{1, p.in_shape.feature[0], 1, 1}};
    }
};

class reduce_eltwise_activation_quantize : public ReduceFusingTest {};
TEST_P(reduce_eltwise_activation_quantize, basic) {
    auto p = GetParam();
    update_out_shape(p);
    create_topologies(input_layout("input", get_input_layout(p)),
                      data("in_lo", get_mem(get_single_element_layout(p), min_random, 0)),
                      data("in_hi", get_mem(get_single_element_layout(p), 1, max_random)),
                      data("out_lo", get_mem(get_single_element_layout(p), -128)),
                      data("out_hi", get_mem(get_single_element_layout(p), 127)),
                      data("eltwise_data", get_mem(get_output_layout(p))),
                      reduce("reduce", "input", p.reduce_mode, p.reduce_axes, p.keep_dims),
                      eltwise("eltwise", {"reduce", "eltwise_data"}, eltwise_mode::sum, p.default_type),
                      activation("activation", "eltwise", activation_func::relu),
                      quantize("quantize", "activation", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::i8),
                      reorder("output_reorder", "quantize", p.default_format, data_types::f32));

    tolerance = 1.f;
    execute(p);
}

TEST_P(reduce_eltwise_activation_quantize, per_channel) {
    auto p = GetParam();
    update_out_shape(p);
    create_topologies(input_layout("input", get_input_layout(p)),
                      data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
                      data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
                      data("out_lo", get_mem(get_single_element_layout(p), -128)),
                      data("out_hi", get_mem(get_single_element_layout(p), 127)),
                      data("eltwise_data", get_mem(get_output_layout(p))),
                      reduce("reduce", "input", p.reduce_mode, p.reduce_axes, p.keep_dims),
                      eltwise("eltwise", {"reduce", "eltwise_data"}, eltwise_mode::sum, p.default_type),
                      activation("activation", "eltwise", activation_func::relu),
                      quantize("quantize", "activation", "in_lo", "in_hi", "out_lo", "out_hi", 256, data_types::i8),
                      reorder("output_reorder", "quantize", p.default_format, data_types::f32));

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu,
                        reduce_eltwise_activation_quantize,
                        ::testing::ValuesIn(std::vector<reduce_test_params>{
                            reduce_test_params{CASE_REDUCE_F32_0, 2, 5, reduce_mode::mean, {reduce::along_x, reduce::along_f, reduce::along_y, reduce::along_b}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_F32_4, 2, 5, reduce_mode::sum, {reduce::along_x, reduce::along_f, reduce::along_b}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_F32_0, 2, 5, reduce_mode::max, {reduce::along_y, reduce::along_f, reduce::along_b}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_F32_4, 2, 5, reduce_mode::sum, {reduce::along_x, reduce::along_y, reduce::along_b}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_F32_0, 2, 5, reduce_mode::min, {reduce::along_x, reduce::along_y, reduce::along_f}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_F32_1, 2, 5, reduce_mode::sum, {reduce::along_f, reduce::along_b}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_F32_2, 2, 5, reduce_mode::mean, {reduce::along_f, reduce::along_x}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_F32_1, 2, 5, reduce_mode::max, {reduce::along_y, reduce::along_b}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_F32_2, 2, 5, reduce_mode::sum, {reduce::along_x, reduce::along_b}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_F32_4, 2, 5, reduce_mode::sum, {reduce::along_b}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_F32_0, 2, 5, reduce_mode::max, {reduce::along_f}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_F32_4, 2, 5, reduce_mode::sum, {reduce::along_y}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_F32_0, 2, 5, reduce_mode::min, {reduce::along_x}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_F32_1, 2, 5, reduce_mode::sum, {reduce::along_b}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_F32_2, 2, 5, reduce_mode::max, {reduce::along_f}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_F32_1, 2, 5, reduce_mode::mean, {reduce::along_x}, true, "reduce_ref"},

                            reduce_test_params{CASE_REDUCE_F16_1, 2, 5, reduce_mode::mean, {reduce::along_x, reduce::along_f, reduce::along_y, reduce::along_b}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_F16_2, 2, 5, reduce_mode::sum, {reduce::along_x, reduce::along_f, reduce::along_b}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_F16_1, 2, 5, reduce_mode::max, {reduce::along_y, reduce::along_f, reduce::along_b}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_F16_2, 2, 5, reduce_mode::sum, {reduce::along_x, reduce::along_y, reduce::along_b}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_F16_1, 2, 5, reduce_mode::min, {reduce::along_x, reduce::along_y, reduce::along_f}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_F16_0, 2, 5, reduce_mode::sum, {reduce::along_f, reduce::along_b}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_F16_4, 2, 5, reduce_mode::mean, {reduce::along_f, reduce::along_x}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_F16_0, 2, 5, reduce_mode::max, {reduce::along_y, reduce::along_b}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_F16_4, 2, 5, reduce_mode::sum, {reduce::along_x, reduce::along_b}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_F16_1, 2, 5, reduce_mode::sum, {reduce::along_b}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_F16_2, 2, 5, reduce_mode::max, {reduce::along_f}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_F16_1, 2, 5, reduce_mode::sum, {reduce::along_y}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_F16_2, 2, 5, reduce_mode::min, {reduce::along_x}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_F16_4, 2, 5, reduce_mode::sum, {reduce::along_b}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_F16_0, 2, 5, reduce_mode::max, {reduce::along_f}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_F16_4, 2, 5, reduce_mode::mean, {reduce::along_x}, true, "reduce_gpu_b_fs_yx_fsv16"},

                            reduce_test_params{CASE_REDUCE_I8_0, 2, 5, reduce_mode::mean, {reduce::along_x, reduce::along_f, reduce::along_y, reduce::along_b}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_I8_4, 2, 5, reduce_mode::sum, {reduce::along_x, reduce::along_f, reduce::along_b}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_I8_0, 2, 5, reduce_mode::max, {reduce::along_y, reduce::along_f, reduce::along_b}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_I8_4, 2, 5, reduce_mode::sum, {reduce::along_x, reduce::along_y, reduce::along_b}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_I8_0, 2, 5, reduce_mode::min, {reduce::along_x, reduce::along_y, reduce::along_f}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_I8_1, 2, 5, reduce_mode::sum, {reduce::along_f, reduce::along_b}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_I8_2, 2, 5, reduce_mode::mean, {reduce::along_f, reduce::along_x}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_I8_1, 2, 5, reduce_mode::max, {reduce::along_y, reduce::along_b}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_I8_2, 2, 5, reduce_mode::sum, {reduce::along_x, reduce::along_b}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_I8_4, 2, 5, reduce_mode::sum, {reduce::along_b}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_I8_0, 2, 5, reduce_mode::max, {reduce::along_f}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_I8_4, 2, 5, reduce_mode::sum, {reduce::along_y}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_I8_0, 2, 5, reduce_mode::min, {reduce::along_x}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_I8_1, 2, 5, reduce_mode::sum, {reduce::along_b}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_I8_2, 2, 5, reduce_mode::max, {reduce::along_f}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_I8_1, 2, 5, reduce_mode::mean, {reduce::along_x}, true, "reduce_ref"},

                            reduce_test_params{CASE_REDUCE_U8_1, 2, 5, reduce_mode::mean, {reduce::along_x, reduce::along_f, reduce::along_y, reduce::along_b}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_U8_2, 2, 5, reduce_mode::sum, {reduce::along_x, reduce::along_f, reduce::along_b}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_U8_1, 2, 5, reduce_mode::max, {reduce::along_y, reduce::along_f, reduce::along_b}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_U8_2, 2, 5, reduce_mode::sum, {reduce::along_x, reduce::along_y, reduce::along_b}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_U8_1, 2, 5, reduce_mode::min, {reduce::along_x, reduce::along_y, reduce::along_f}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_U8_0, 2, 5, reduce_mode::sum, {reduce::along_f, reduce::along_b}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_U8_4, 2, 5, reduce_mode::mean, {reduce::along_f, reduce::along_x}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_U8_0, 2, 5, reduce_mode::max, {reduce::along_y, reduce::along_b}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_U8_4, 2, 5, reduce_mode::sum, {reduce::along_x, reduce::along_b}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_U8_1, 2, 5, reduce_mode::sum, {reduce::along_b}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_U8_2, 2, 5, reduce_mode::max, {reduce::along_f}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_U8_1, 2, 5, reduce_mode::sum, {reduce::along_y}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_U8_2, 2, 5, reduce_mode::min, {reduce::along_x}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_U8_4, 2, 5, reduce_mode::sum, {reduce::along_b}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_U8_0, 2, 5, reduce_mode::max, {reduce::along_f}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_U8_4, 2, 5, reduce_mode::mean, {reduce::along_x}, true, "reduce_gpu_b_fs_yx_fsv16"}
                        }));

class reduce_scale_activation : public ReduceFusingTest {};
TEST_P(reduce_scale_activation, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      data("scale_data", get_mem(get_single_element_layout(p), -0.125f)),
                      reduce("reduce", "input", p.reduce_mode, p.reduce_axes, p.keep_dims),
                      scale("scale", "reduce", "scale_data"),
                      activation("activation", "scale", activation_func::cos),
                      reorder("output_reorder", "activation", p.default_format, data_types::f32));
    tolerance = 1e-02f;
    execute(p);
}

TEST_P(reduce_scale_activation, per_channel) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
                      data("scale_data", get_mem(get_per_channel_layout(p), -0.125f)),
                      reduce("reduce", "input", p.reduce_mode, p.reduce_axes, p.keep_dims),
                      scale("scale", "reduce", "scale_data"),
                      activation("activation", "scale", activation_func::cos),
                      reorder("output_reorder", "activation", p.default_format, data_types::f32));
    tolerance = 1e-02f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu,
                        reduce_scale_activation,
                        ::testing::ValuesIn(std::vector<reduce_test_params>{
                            reduce_test_params{CASE_REDUCE_F32_0, 2, 4, reduce_mode::max, {reduce::along_x, reduce::along_y, reduce::along_b}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_F32_1, 2, 4, reduce_mode::sum, {reduce::along_x, reduce::along_y, reduce::along_b}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_F32_0, 2, 4, reduce_mode::min, {reduce::along_x, reduce::along_y}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_F32_2, 2, 4, reduce_mode::mean, {reduce::along_x, reduce::along_y}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_F32_0, 2, 4, reduce_mode::l1, {reduce::along_x}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_F32_0, 2, 4, reduce_mode::l1, {reduce::along_x}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_F32_0, 2, 4, reduce_mode::min, {reduce::along_y}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_F32_0, 2, 4, reduce_mode::sum, {reduce::along_y}, true, "reduce_gpu_b_fs_yx_fsv16"},

                            reduce_test_params{CASE_REDUCE_F16_0, 2, 4, reduce_mode::max, {reduce::along_x, reduce::along_y, reduce::along_b}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_F16_1, 2, 4, reduce_mode::sum, {reduce::along_x, reduce::along_y, reduce::along_b}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_F16_0, 2, 4, reduce_mode::min, {reduce::along_x, reduce::along_y}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_F16_2, 2, 4, reduce_mode::mean, {reduce::along_x, reduce::along_y}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_F16_0, 2, 4, reduce_mode::min, {reduce::along_x}, true, "reduce_gpu_b_fs_yx_fsv16"},
                            reduce_test_params{CASE_REDUCE_F16_0, 2, 4, reduce_mode::sum, {reduce::along_x}, true, "reduce_gpu_b_fs_yx_fsv16"},
                        }));

INSTANTIATE_TEST_SUITE_P(DISABLED_fusings_gpu,
                        reduce_eltwise_activation_quantize,
                        ::testing::ValuesIn(std::vector<reduce_test_params>{
                            // No layout format available for quantize/scale
                            reduce_test_params{CASE_REDUCE_F32_3, 2, 4, reduce_mode::l1, {reduce::along_x}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_F16_3, 2, 4, reduce_mode::min, {reduce::along_x}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_I32_2, 2, 4, reduce_mode::max, {reduce::along_x, reduce::along_y}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_I32_3, 2, 4, reduce_mode::sum, {reduce::along_x}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_I8_3, 2, 4, reduce_mode::mean, {reduce::along_x}, true, "reduce_ref"},
                            reduce_test_params{CASE_REDUCE_U8_3, 2, 4, reduce_mode::l2, {reduce::along_x}, true, "reduce_ref"}
                        }));


/* ----------------------------------------------------------------------------------------------------- */
/* ------------------------------------------ ScatterNDUpdate cases ------------------------------ */
/* ----------------------------------------------------------------------------------------------------- */
struct scatter_nd_update_test_params {
    tensor input_shape;
    tensor indices_shape;
    tensor updates_shape;
    int max_number_in_indices;
    int indices_rank;
    data_types data_type;
    format input_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

#define CASE_SCATTER_ND_UPDATE_FP16_4D_1 {6, 1, 1, 1}, {3, 1, 1, 1}, {3, 1, 1, 1}, 6, 1, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_4D_2 {6, 6, 1, 1}, {3, 2, 1, 1}, {3, 1, 1, 1}, 6, 2, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_4D_3 {6, 7, 8, 9}, {5, 1, 1, 1}, {5, 7, 8, 9}, 6, 2, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_4D_4 {6, 7, 8, 9}, {5, 1, 1, 1}, {5, 7, 8, 9}, 6, 2, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_4D_5 {6, 7, 8, 9}, {6, 2, 1, 1}, {6, 9, 1, 8}, 6, 2, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_4D_6 {6, 7, 8, 9}, {6, 3, 1, 1}, {6, 8, 1, 1}, 6, 2, data_types::f16, format::bfyx, data_types::f16, format::bfyx

#define CASE_SCATTER_ND_UPDATE_FP16_5D_1 {6, 7, 8, 9, 10}, {5, 1, 1, 1, 1}, {5, 7, 8, 9, 10}, 6, 1, data_types::f16, format::bfzyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_5D_2 {6, 7, 8, 9, 10}, {5, 2, 1, 1, 1}, {5, 10, 1, 8, 9}, 6, 2, data_types::f16, format::bfzyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_5D_3 {6, 7, 8, 9, 10}, {5, 3, 1, 1, 1}, {5, 9, 1, 1, 8}, 6, 2, data_types::f16, format::bfzyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_5D_4 {6, 7, 8, 9, 10}, {5, 4, 1, 1, 1}, {5, 8, 1, 1, 1}, 6, 2, data_types::f16, format::bfzyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_5D_5 {6, 7, 8, 9, 10}, {5, 5, 1, 1, 1}, {5, 1, 1, 1, 1}, 6, 2, data_types::f16, format::bfzyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_5D_6 {6, 7, 8, 9, 10}, {5, 2, 1, 1, 2}, {5, 2, 8, 9, 10}, 6, 3, data_types::f16, format::bfzyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_5D_7 {6, 7, 8, 9, 10}, {5, 2, 1, 1, 3}, {5, 2, 1, 8, 9}, 6, 3, data_types::f16, format::bfzyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_5D_8 {6, 7, 8, 9, 10}, {5, 2, 1, 4, 3}, {5, 2, 1, 8, 3}, 6, 4, data_types::f16, format::bfzyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_5D_9 {6, 7, 8, 9, 10}, {5, 2, 1, 3, 3}, {5, 2, 8, 9, 3}, 6, 4, data_types::f16, format::bfzyx, data_types::f16, format::bfyx

#define CASE_SCATTER_ND_UPDATE_FP16_6D_1 {6, 7, 8, 9, 10, 11}, {5, 1, 1, 1}, {5, 7, 8, 9, 10, 11}, 6, 1, data_types::f16, format::bfwzyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_6D_2 {6, 7, 8, 9, 10, 11}, {5, 2, 1, 1}, {5, 11, 1, 8, 9, 10}, 6, 2, data_types::f16, format::bfwzyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_6D_3 {6, 7, 8, 9, 10, 11}, {5, 3, 1, 1}, {5, 10, 1, 1, 8, 9}, 6, 2, data_types::f16, format::bfwzyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_6D_4 {6, 7, 8, 9, 10, 11}, {5, 4, 1, 1}, {5, 9, 1, 1, 1, 8}, 6, 2, data_types::f16, format::bfwzyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_6D_5 {6, 7, 8, 9, 2, 2}, {5, 5, 1, 1}, {5, 8, 1, 1, 1, 1}, 6, 2, data_types::f16, format::bfwzyx, data_types::f16, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP16_6D_6 {6, 7, 8, 9, 2, 2}, {5, 6, 1, 1}, {5, 1, 1, 1, 1, 1}, 6, 2, data_types::f16, format::bfwzyx, data_types::f16, format::bfyx

#define CASE_SCATTER_ND_UPDATE_FP32_4D_1 {6, 1, 1, 1}, {3, 1, 1, 1}, {3, 1, 1, 1}, 6, 1, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_4D_2 {6, 6, 1, 1}, {3, 2, 1, 1}, {3, 1, 1, 1}, 6, 2, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_4D_3 {6, 7, 8, 1}, {5, 1, 1, 1}, {5, 7, 8, 1}, 6, 2, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_4D_4 {6, 7, 8, 9}, {5, 1, 1, 1}, {5, 7, 8, 9}, 6, 2, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_4D_5 {6, 7, 8, 9}, {6, 2, 1, 1}, {6, 9, 1, 8}, 6, 2, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_4D_6 {6, 7, 8, 9}, {6, 3, 1, 1}, {6, 8, 1, 1}, 6, 2, data_types::f32, format::bfyx, data_types::f32, format::bfyx

#define CASE_SCATTER_ND_UPDATE_FP32_5D_1 {6, 7, 8, 9, 10}, {5, 1, 1, 1, 1}, {5, 7, 8, 9, 10}, 6, 1, data_types::f32, format::bfzyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_5D_2 {6, 7, 8, 9, 10}, {5, 2, 1, 1, 1}, {5, 10, 1, 8, 9}, 6, 2, data_types::f32, format::bfzyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_5D_3 {6, 7, 8, 9, 10}, {5, 3, 1, 1, 1}, {5, 9, 1, 1, 8}, 6, 2, data_types::f32, format::bfzyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_5D_4 {6, 7, 8, 9, 10}, {5, 4, 1, 1, 1}, {5, 8, 1, 1, 1}, 6, 2, data_types::f32, format::bfzyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_5D_5 {6, 7, 8, 9, 10}, {5, 5, 1, 1, 1}, {5, 1, 1, 1, 1}, 6, 2, data_types::f32, format::bfzyx, data_types::f32, format::bfyx

#define CASE_SCATTER_ND_UPDATE_FP32_6D_1 {6, 7, 8, 9, 10, 11}, {5, 1, 1, 1}, {5, 7, 8, 9, 10, 11}, 6, 1, data_types::f32, format::bfwzyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_6D_2 {6, 7, 8, 9, 10, 11}, {5, 2, 1, 1}, {5, 11, 1, 8, 9, 10}, 6, 2, data_types::f32, format::bfwzyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_6D_3 {6, 7, 8, 9, 10, 11}, {5, 3, 1, 1}, {5, 10, 1, 1, 8, 9}, 6, 2, data_types::f32, format::bfwzyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_6D_4 {6, 7, 8, 9, 10, 11}, {5, 4, 1, 1}, {5, 9, 1, 1, 1, 8}, 6, 2, data_types::f32, format::bfwzyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_6D_5 {6, 7, 8, 9, 2, 2}, {5, 5, 1, 1}, {5, 8, 1, 1, 1, 1}, 6, 2, data_types::f32, format::bfwzyx, data_types::f32, format::bfyx
#define CASE_SCATTER_ND_UPDATE_FP32_6D_6 {6, 7, 8, 9, 2, 2}, {5, 6, 1, 1}, {5, 1, 1, 1, 1, 1}, 6, 2, data_types::f32, format::bfwzyx, data_types::f32, format::bfyx

class ScatterNDUpdatePrimitiveFusingTest : public ::BaseFusingTest<scatter_nd_update_test_params> {
public:
    void execute(scatter_nd_update_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);
        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(scatter_nd_update_test_params& p) {
        return layout{ p.data_type, p.input_format, p.input_shape };
    }

    layout get_indices_layout(scatter_nd_update_test_params& p) {
        return layout{ p.data_type, p.input_format, p.indices_shape };
    }

    layout get_updates_layout(scatter_nd_update_test_params& p) {
        return layout{ p.data_type, p.input_format, p.updates_shape };
    }

    layout get_per_channel_layout(scatter_nd_update_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{1, p.input_shape.feature[0], 1, 1} };
    }
};

class scatter_nd_update_quantize : public ScatterNDUpdatePrimitiveFusingTest {};
TEST_P(scatter_nd_update_quantize, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
        data("scatter_nd_update_indices", get_mem(get_indices_layout(p), 0, p.max_number_in_indices)),
        data("scatter_nd_update_updates", get_mem(get_updates_layout(p), 0, 100)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        scatter_nd_update("scatter_nd_update_prim", "input", "scatter_nd_update_indices", "scatter_nd_update_updates", p.indices_rank),
        quantize("quantize", "scatter_nd_update_prim", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );
    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, scatter_nd_update_quantize,
    ::testing::ValuesIn(std::vector<scatter_nd_update_test_params>{
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_1, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_2, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_3, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_4, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_5, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_6, 2, 3 },

        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_1, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_2, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_3, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_4, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_5, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_7, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_9, 2, 3 },

        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_1, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_2, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_3, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_4, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_5, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_6, 2, 3 },

        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_1, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_2, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_3, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_4, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_5, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_6, 2, 3 },

        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_1, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_2, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_3, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_4, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_5, 2, 3 },

        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_1, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_2, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_3, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_4, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_5, 2, 3 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_6, 2, 3 },
}));

class scatter_nd_update_scale_activation_eltwise : public ScatterNDUpdatePrimitiveFusingTest {};
TEST_P(scatter_nd_update_scale_activation_eltwise, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
        data("scatter_nd_update_indices", get_mem(get_indices_layout(p), 0, p.max_number_in_indices)),
        data("scatter_nd_update_updates", get_mem(get_updates_layout(p), 0, 100)),
        data("scale_data", get_mem(get_per_channel_layout(p), -1, 1)),
        data("eltwise_data", get_mem(layout{ p.data_type, p.input_format, p.input_shape })),
        scatter_nd_update("scatter_nd_update_prim", "input", "scatter_nd_update_indices", "scatter_nd_update_updates", p.indices_rank),
        activation("activation", "scatter_nd_update_prim", activation_func::abs),
        scale("scale", "activation", "scale_data"),
        eltwise("eltwise", { "scale", "eltwise_data" }, eltwise_mode::sum, p.data_type),
        reorder("reorder_bfyx", "eltwise", p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, scatter_nd_update_scale_activation_eltwise,
    ::testing::ValuesIn(std::vector<scatter_nd_update_test_params>{
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_1, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_2, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_3, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_4, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_5, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_4D_6, 2, 5 },

        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_1, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_2, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_3, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_4, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_5, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_6, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_7, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_9, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_5D_8, 2, 5 },

        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_1, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_2, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_3, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_4, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_5, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP16_6D_6, 2, 5 },

        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_1, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_2, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_3, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_4, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_5, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_4D_6, 2, 5 },

        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_1, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_2, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_3, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_4, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_5D_5, 2, 5 },

        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_1, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_2, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_3, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_4, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_5, 2, 5 },
        scatter_nd_update_test_params{ CASE_SCATTER_ND_UPDATE_FP32_6D_6, 2, 5 },
}));


/* ----------------------------------------------------------------------------------------------------- */
/* ------------------------------------------ GatherND cases ------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
struct gather_nd_test_params {
    data_types data_type;

    format input_format;
    tensor input_shape;

    format indices_format;
    tensor indices_shape;

    format output_format;
    tensor output_shape;

    int max_number_in_indices;
    int indices_rank;
    int batch_dims;

    data_types default_type;
    format default_format;

    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

#define CASE_GATHER_ND_FP16_4D_1 data_types::f16, format::bfyx, {6, 7, 9, 8}, format::bfyx, {3, 1, 1, 1}, format::bfyx, {3, 7, 9, 8}, 6, 2, 0, data_types::f16, format::bfyx
#define CASE_GATHER_ND_FP16_4D_2 data_types::f16, format::bfyx, {6, 7, 9, 8}, format::bfyx, {6, 1, 1, 1}, format::bfyx, {6, 8, 1, 9}, 6, 2, 1, data_types::f16, format::bfyx
#define CASE_GATHER_ND_FP16_4D_3 data_types::f16, format::bfyx, {5, 4, 7, 2}, format::bfyx, {5, 4, 1, 2}, format::bfyx, {40, 1, 1, 1}, 6, 4, 3, data_types::f16, format::bfyx

#define CASE_GATHER_ND_FP16_5D_1 data_types::f16, format::bfzyx, {5, 6, 7, 8, 5}, format::bfyx, {5, 1, 1, 1}, format::bfzyx, {5, 6, 7, 8, 5}, 5, 2, 0, data_types::f16, format::bfyx
#define CASE_GATHER_ND_FP16_5D_2 data_types::f16, format::bfzyx, {5, 6, 7, 8, 5}, format::bfyx, {5, 1, 1, 1}, format::bfyx, {5, 5, 7, 8}, 5, 2, 1, data_types::f16, format::bfyx
#define CASE_GATHER_ND_FP16_5D_3 data_types::f16, format::bfzyx, {5, 4, 7, 8, 5}, format::bfyx, {5, 4, 1, 3}, format::bfyx, {20, 1, 1, 1}, 4, 3, 2, data_types::f16, format::bfyx
#define CASE_GATHER_ND_FP16_5D_4 data_types::f16, format::bfzyx, {5, 4, 7, 8, 3}, format::bfyx, {5, 4, 1, 3}, format::bfyx, {60, 7, 1, 1}, 4, 4, 3, data_types::f16, format::bfyx
#define CASE_GATHER_ND_FP16_5D_5 data_types::f16, format::bfzyx, {5, 4, 7, 2, 3}, format::bfzyx, {5, 4, 1, 2, 3}, format::bfyx, {120, 1, 1, 1}, 4, 5, 4, data_types::f16, format::bfyx
#define CASE_GATHER_ND_FP16_5D_6 data_types::f16, format::bfzyx, {5, 4, 7, 4, 4}, format::bfzyx, {5, 4, 1, 1, 3}, format::bfzyx, {20, 3, 7, 4, 1}, 4, 5, 2, data_types::f16, format::bfyx

#define CASE_GATHER_ND_FP16_6D_1 data_types::f16, format::bfwzyx, {5, 4, 6, 7, 8, 5}, format::bfyx, {5, 4, 2, 2}, format::bfyx, {20, 2, 6, 7}, 5, 4, 2, data_types::f16, format::bfyx
#define CASE_GATHER_ND_FP16_6D_2 data_types::f16, format::bfwzyx, {5, 4, 6, 7, 8, 2}, format::bfyx, {5, 4, 2, 2}, format::bfyx, {40, 6, 1, 1}, 5, 4, 3, data_types::f16, format::bfyx
#define CASE_GATHER_ND_FP16_6D_3 data_types::f16, format::bfwzyx, {5, 4, 6, 7, 2, 2}, format::bfzyx, {5, 4, 1, 2, 2}, format::bfyx, {80, 6, 1, 1}, 5, 5, 4, data_types::f16, format::bfyx
#define CASE_GATHER_ND_FP16_6D_4 data_types::f16, format::bfwzyx, {5, 4, 6, 3, 2, 2}, format::bfwzyx, {5, 4, 1, 3, 2, 2}, format::bfyx, {240, 1, 1, 1}, 5, 6, 5, data_types::f16, format::bfyx

#define CASE_GATHER_ND_FP32_4D_1 data_types::f32, format::bfyx, {6, 7, 9, 8}, format::bfyx, {3, 1, 1, 1}, format::bfyx, {3, 7, 9, 8}, 6, 2, 0, data_types::f32, format::bfyx
#define CASE_GATHER_ND_FP32_4D_2 data_types::f32, format::bfyx, {6, 7, 9, 8}, format::bfyx, {6, 1, 1, 1}, format::bfyx, {6, 8, 1, 9}, 6, 2, 1, data_types::f32, format::bfyx
#define CASE_GATHER_ND_FP32_4D_3 data_types::f32, format::bfyx, {5, 4, 7, 2}, format::bfyx, {5, 4, 1, 2}, format::bfyx, {40, 1, 1, 1}, 6, 4, 3, data_types::f32, format::bfyx

#define CASE_GATHER_ND_FP32_5D_1 data_types::f32, format::bfzyx, {5, 6, 7, 8, 5}, format::bfyx, {5, 1, 1, 1}, format::bfzyx, {5, 6, 7, 8, 5}, 5, 2, 0, data_types::f32, format::bfyx
#define CASE_GATHER_ND_FP32_5D_2 data_types::f32, format::bfzyx, {5, 6, 7, 8, 5}, format::bfyx, {5, 1, 1, 1}, format::bfyx, {5, 5, 7, 8}, 5, 2, 1, data_types::f32, format::bfyx
#define CASE_GATHER_ND_FP32_5D_3 data_types::f32, format::bfzyx, {5, 4, 7, 8, 5}, format::bfyx, {5, 4, 1, 3}, format::bfyx, {20, 1, 1, 1}, 4, 3, 2, data_types::f32, format::bfyx
#define CASE_GATHER_ND_FP32_5D_4 data_types::f32, format::bfzyx, {5, 4, 7, 8, 3}, format::bfyx, {5, 4, 1, 3}, format::bfyx, {60, 7, 1, 1}, 4, 4, 3, data_types::f32, format::bfyx
#define CASE_GATHER_ND_FP32_5D_5 data_types::f32, format::bfzyx, {5, 4, 7, 2, 3}, format::bfzyx, {5, 4, 1, 2, 3}, format::bfyx, {120, 1, 1, 1}, 4, 5, 4, data_types::f32, format::bfyx
#define CASE_GATHER_ND_FP32_5D_6 data_types::f32, format::bfzyx, {5, 4, 7, 4, 4}, format::bfzyx, {5, 4, 1, 1, 3}, format::bfzyx, {20, 3, 7, 4, 1}, 4, 5, 2, data_types::f32, format::bfyx

#define CASE_GATHER_ND_FP32_6D_1 data_types::f32, format::bfwzyx, {5, 4, 6, 7, 8, 5}, format::bfyx, {5, 4, 2, 2}, format::bfyx, {20, 2, 6, 7}, 5, 4, 2, data_types::f32, format::bfyx
#define CASE_GATHER_ND_FP32_6D_2 data_types::f32, format::bfwzyx, {5, 4, 6, 7, 8, 2}, format::bfyx, {5, 4, 2, 2}, format::bfyx, {40, 6, 1, 1}, 5, 4, 3, data_types::f32, format::bfyx
#define CASE_GATHER_ND_FP32_6D_3 data_types::f32, format::bfwzyx, {5, 4, 6, 7, 2, 2}, format::bfzyx, {5, 4, 1, 2, 2}, format::bfyx, {80, 6, 1, 1}, 5, 5, 4, data_types::f32, format::bfyx
#define CASE_GATHER_ND_FP32_6D_4 data_types::f32, format::bfwzyx, {5, 4, 6, 3, 2, 2}, format::bfwzyx, {5, 4, 1, 3, 2, 2}, format::bfyx, {240, 1, 1, 1}, 5, 6, 5, data_types::f32, format::bfyx



class GatherNDPrimitiveFusingTest : public ::BaseFusingTest<gather_nd_test_params> {
public:
    void execute(gather_nd_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, bo_not_fused);
        network network_fused(this->engine, this->topology_fused, bo_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);
        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(gather_nd_test_params& p) {
        return layout{ p.data_type, p.input_format, p.input_shape };
    }

    layout get_indices_layout(gather_nd_test_params& p) {
        return layout{ p.data_type, p.indices_format, p.indices_shape };
    }

    layout get_output_layout(gather_nd_test_params& p) {
        return layout{ p.data_type, p.output_format, p.output_shape };
    }

    layout get_per_channel_layout(gather_nd_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{1, p.output_shape.feature[0], 1, 1} };
    }
};

class gather_nd_quantize : public GatherNDPrimitiveFusingTest {};
TEST_P(gather_nd_quantize, basic) {
    auto p = GetParam();
    create_topologies(input_layout("input", get_input_layout(p)),
        data("gather_nd_indices", get_mem(get_indices_layout(p), 0, p.max_number_in_indices - 1)),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        gather_nd("gather_nd_prim", "input", "gather_nd_indices", p.indices_rank, p.batch_dims),
        quantize("quantize", "gather_nd_prim", "in_lo", "in_hi", "out_lo", "out_hi", 255, data_types::i8),
        reorder("reorder_bfyx", "quantize", p.default_format, data_types::f32)
    );
    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gather_nd_quantize,
    ::testing::ValuesIn(std::vector<gather_nd_test_params>{
        gather_nd_test_params{ CASE_GATHER_ND_FP16_4D_1, 2, 3 },
        gather_nd_test_params{ CASE_GATHER_ND_FP16_4D_2, 2, 3 },
        gather_nd_test_params{ CASE_GATHER_ND_FP16_4D_3, 2, 3 },

        gather_nd_test_params{ CASE_GATHER_ND_FP16_5D_1, 2, 3 },
        gather_nd_test_params{ CASE_GATHER_ND_FP16_5D_2, 2, 3 },
        gather_nd_test_params{ CASE_GATHER_ND_FP16_5D_3, 2, 3 },
        gather_nd_test_params{ CASE_GATHER_ND_FP16_5D_4, 2, 3 },
        gather_nd_test_params{ CASE_GATHER_ND_FP16_5D_5, 2, 3 },
        gather_nd_test_params{ CASE_GATHER_ND_FP16_5D_6, 2, 3 },

        gather_nd_test_params{ CASE_GATHER_ND_FP16_6D_1, 2, 3 },
        gather_nd_test_params{ CASE_GATHER_ND_FP16_6D_2, 2, 3 },
        gather_nd_test_params{ CASE_GATHER_ND_FP16_6D_3, 2, 3 },
        gather_nd_test_params{ CASE_GATHER_ND_FP16_6D_4, 2, 3 },

        gather_nd_test_params{ CASE_GATHER_ND_FP32_4D_1, 2, 3 },
        gather_nd_test_params{ CASE_GATHER_ND_FP32_4D_2, 2, 3 },
        gather_nd_test_params{ CASE_GATHER_ND_FP32_4D_3, 2, 3 },

        gather_nd_test_params{ CASE_GATHER_ND_FP32_5D_1, 2, 3 },
        gather_nd_test_params{ CASE_GATHER_ND_FP32_5D_2, 2, 3 },
        gather_nd_test_params{ CASE_GATHER_ND_FP32_5D_3, 2, 3 },
        gather_nd_test_params{ CASE_GATHER_ND_FP32_5D_4, 2, 3 },
        gather_nd_test_params{ CASE_GATHER_ND_FP32_5D_5, 2, 3 },
        gather_nd_test_params{ CASE_GATHER_ND_FP32_5D_6, 2, 3 },

        gather_nd_test_params{ CASE_GATHER_ND_FP32_6D_1, 2, 3 },
        gather_nd_test_params{ CASE_GATHER_ND_FP32_6D_2, 2, 3 },
        gather_nd_test_params{ CASE_GATHER_ND_FP32_6D_3, 2, 3 },
        gather_nd_test_params{ CASE_GATHER_ND_FP32_6D_4, 2, 3 },
}));

class gather_nd_activation_scale_eltwise : public GatherNDPrimitiveFusingTest {};
TEST_P(gather_nd_activation_scale_eltwise, basic) {
    auto p = GetParam();

    create_topologies(input_layout("input", get_input_layout(p)),
        data("gather_nd_indices", get_mem(get_indices_layout(p), 0, p.max_number_in_indices - 1)),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f / 255)),
        data("eltwise_data", get_mem(get_output_layout(p))),
        gather_nd("gather_nd_prim", "input", "gather_nd_indices", p.indices_rank, p.batch_dims),
        activation("activation", "gather_nd_prim", activation_func::abs),
        scale("scale", "activation", "scale_data"),
        eltwise("eltwise", { "scale", "eltwise_data" }, eltwise_mode::sum, p.data_type),
        reorder("reorder_bfyx", "eltwise", p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gather_nd_activation_scale_eltwise,
    ::testing::ValuesIn(std::vector<gather_nd_test_params>{
        gather_nd_test_params{ CASE_GATHER_ND_FP16_4D_1, 2, 5 },
        gather_nd_test_params{ CASE_GATHER_ND_FP16_4D_2, 2, 5 },
        gather_nd_test_params{ CASE_GATHER_ND_FP16_4D_3, 2, 5 },

        gather_nd_test_params{ CASE_GATHER_ND_FP16_5D_1, 2, 5 },
        gather_nd_test_params{ CASE_GATHER_ND_FP16_5D_2, 2, 5 },
        gather_nd_test_params{ CASE_GATHER_ND_FP16_5D_3, 2, 5 },
        gather_nd_test_params{ CASE_GATHER_ND_FP16_5D_4, 2, 5 },
        gather_nd_test_params{ CASE_GATHER_ND_FP16_5D_5, 2, 5 },
        gather_nd_test_params{ CASE_GATHER_ND_FP16_5D_6, 2, 5 },

        gather_nd_test_params{ CASE_GATHER_ND_FP16_6D_1, 2, 5 },
        gather_nd_test_params{ CASE_GATHER_ND_FP16_6D_2, 2, 5 },
        gather_nd_test_params{ CASE_GATHER_ND_FP16_6D_3, 2, 5 },
        gather_nd_test_params{ CASE_GATHER_ND_FP16_6D_4, 2, 5 },

        gather_nd_test_params{ CASE_GATHER_ND_FP32_4D_1, 2, 5 },
        gather_nd_test_params{ CASE_GATHER_ND_FP32_4D_2, 2, 5 },
        gather_nd_test_params{ CASE_GATHER_ND_FP32_4D_3, 2, 5 },

        gather_nd_test_params{ CASE_GATHER_ND_FP32_5D_1, 2, 5 },
        gather_nd_test_params{ CASE_GATHER_ND_FP32_5D_2, 2, 5 },
        gather_nd_test_params{ CASE_GATHER_ND_FP32_5D_3, 2, 5 },
        gather_nd_test_params{ CASE_GATHER_ND_FP32_5D_4, 2, 5 },
        gather_nd_test_params{ CASE_GATHER_ND_FP32_5D_5, 2, 5 },
        gather_nd_test_params{ CASE_GATHER_ND_FP32_5D_6, 2, 5 },

        gather_nd_test_params{ CASE_GATHER_ND_FP32_6D_1, 2, 5 },
        gather_nd_test_params{ CASE_GATHER_ND_FP32_6D_2, 2, 5 },
        gather_nd_test_params{ CASE_GATHER_ND_FP32_6D_3, 2, 5 },
        gather_nd_test_params{ CASE_GATHER_ND_FP32_6D_4, 2, 5 },
}));
