// Copyright (C) 2023-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/strided_slice.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
struct strided_slice_test_params {
    tensor input_size;
    tensor output_size;
    data_types input_type;
    format input_format;
    size_t expected_fused_primitives;
    size_t expected_fused_primitives_onednn;
    size_t expected_not_fused_primitives;
    std::vector<std::pair<activation_func,activation_additional_params>> activation_func_list;
};

class StridedSliceFusingsTest : public ::BaseFusingTest<strided_slice_test_params> {
public:
    void execute(strided_slice_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));

        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);

        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(strided_slice_test_params& p) {
        return layout{ p.input_type, p.input_format, p.input_size };
    }

    format get_input_format(strided_slice_test_params &p) {
        return p.input_format;
    }
};

}  // namespace

/* ----------------------------------------------------------------------------------------------------- */
/* -------------------------------- StridedSlice cases ------------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

#define CASE_STRIDED_SLICE_F16_1 { 1, 8, 1, 1 },  { 1, 3, 2, 2 }, data_types::f16, format::bfyx


class strided_slice_activation : public StridedSliceFusingsTest {};
TEST_P(strided_slice_activation, basic) {
    std::vector<int64_t> begin_data = { 0, 0, 0, 0 };
    std::vector<int64_t> end_data = { 1, 8, 1, 1 };
    std::vector<int64_t> strides_data = { 1, 1, 1, 1 };

    auto p = GetParam();
    if (engine.get_device_info().supports_immad)
        p.expected_fused_primitives = p.expected_fused_primitives_onednn;
    create_topologies(
        input_layout("input", get_input_layout(p)),
        strided_slice("strided_slice", input_info("input"), begin_data, end_data, strides_data, {}, {}, {}, {}, {}, { 1, 8, 1, 1 })
    );

    std::string before_name = "strided_slice";
    for (auto& act_item : p.activation_func_list) {
        std::string act_name = "actv_" + activation_type_to_str(act_item.first);
        add_topologies(activation(act_name, input_info(before_name), act_item.first, act_item.second));
        before_name = act_name;
    }

    add_topologies(
        reorder("reorder_bfyx", input_info(before_name), format::bfyx, data_types::f16));

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, strided_slice_activation, ::testing::ValuesIn(std::vector<strided_slice_test_params>{
    strided_slice_test_params{ CASE_STRIDED_SLICE_F16_1, 2, 2, 4, {{ activation_func::clamp, { } }, { activation_func::exp, { } }} },
    strided_slice_test_params{ CASE_STRIDED_SLICE_F16_1, 2, 2, 3, {{ activation_func::logistic, { } } } },
    strided_slice_test_params{ CASE_STRIDED_SLICE_F16_1, 2, 3, 3, {{ activation_func::hyperbolic_tan, { } } } },
}));
