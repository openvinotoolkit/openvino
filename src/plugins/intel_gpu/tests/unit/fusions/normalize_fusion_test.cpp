// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/mvn.hpp>
#include <intel_gpu/primitives/permute.hpp>
#include <intel_gpu/primitives/gather.hpp>
#include <intel_gpu/primitives/gather_nd.hpp>
#include <intel_gpu/primitives/gather_elements.hpp>
#include <intel_gpu/primitives/scatter_update.hpp>
#include <intel_gpu/primitives/scatter_nd_update.hpp>
#include <intel_gpu/primitives/scatter_elements_update.hpp>
#include <intel_gpu/primitives/depth_to_space.hpp>
#include <intel_gpu/primitives/space_to_depth.hpp>
#include <intel_gpu/primitives/batch_to_space.hpp>
#include <intel_gpu/primitives/space_to_batch.hpp>
#include <intel_gpu/primitives/reduce.hpp>
#include <intel_gpu/primitives/crop.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {

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


class NormalizeFusingTest : public ::BaseFusingTest<normalize_test_params> {
public:
    void execute(normalize_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(normalize_test_params& p) {
        return layout{ p.data_type, p.input_format, p.in_shape };
    }

    layout get_per_channel_layout(normalize_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{ 1, p.in_shape.feature[0], 1, 1 } };
    }

    layout get_weights_layout(normalize_test_params& p) {
        return layout { p.default_type, p.default_format, tensor{ 1, p.in_shape.feature[0], 1, 1 } };
    }
};

}  // namespace

// in_shape; out_shape; kernel; stride; pad; dilation; groups; data_type; input_format; weights_type; weights_format; default_type; default_format;
#define CASE_NORMALIZE_I8_1 { 1, 2, 3, 3 }, data_types::u8, format::bfyx, data_types::f32, format::bfyx

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
        normalize("normalizel2", input_info("input"), "weights", p.across_spatial),
        quantize("quantize", input_info("normalizel2"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::u8),
        reorder("output_reorder", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, normalize_i8_quantize, ::testing::ValuesIn(std::vector<normalize_test_params>{
    normalize_test_params{ CASE_NORMALIZE_I8_1, false, 2, 3 },
    normalize_test_params{ CASE_NORMALIZE_I8_1, true, 2, 3 },
}));

class normalize_i8_float : public NormalizeFusingTest {};
TEST_P(normalize_i8_float, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("weights", get_mem(get_weights_layout(p))),
        data("scale_data", get_mem(get_per_channel_layout(p), 1.0f/255)),
        normalize("normalizel2", input_info("input"), "weights", p.across_spatial),
        eltwise("scale", { input_info("normalizel2"), input_info("scale_data") }, eltwise_mode::prod, p.default_type),
        activation("activation", input_info("scale"), activation_func::abs),
        reorder("output_reorder", input_info("activation"), p.default_format, data_types::f32)
    );

    tolerance = 1e-05f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, normalize_i8_float, ::testing::ValuesIn(std::vector<normalize_test_params>{
    normalize_test_params{ CASE_NORMALIZE_I8_1, false, 3, 4 },
    normalize_test_params{ CASE_NORMALIZE_I8_1, true, 3, 4 },
}));
