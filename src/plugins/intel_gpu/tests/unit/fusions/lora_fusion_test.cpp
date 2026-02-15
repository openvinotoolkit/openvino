// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/lora.hpp>
#include <intel_gpu/primitives/read_value.hpp>

#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/variable_state.hpp"

#include "lora_inst.h"

#include <cmath>

using namespace cldnn;
using namespace ov::intel_gpu;
using namespace ::tests;

namespace {

struct lora_test_params {
    ov::PartialShape lora_input_pshape;
    ov::PartialShape fc_weights_pshape;
    ov::PartialShape main_input_pshape;
    std::vector<ov::PartialShape> lora_states;
    data_types input_type;
    format planar_format;
    format weights_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

class LoraFusingsTest : public ::BaseFusingTest<lora_test_params> {
public:
    void execute(lora_test_params& p, std::string check_dyn_impl_name = "") {
        cfg_not_fused.set_property(allow_new_shape_infer(true));
        cfg_fused.set_property(allow_new_shape_infer(true));

        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);

        auto input_prim = get_mem(get_lora_input_layout(p));
        auto state_a = get_mem(get_lora_state_layout(p, 0));
        auto state_alpha = get_mem(get_lora_state_layout(p, 1));
        auto state_b = get_mem(get_lora_state_layout(p, 2));

        network_fused.set_input_data("input", input_prim);
        network_fused.set_input_data("state_a", state_a);
        network_fused.set_input_data("state_alpha", state_alpha);
        network_fused.set_input_data("state_b", state_b);

        network_not_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("state_a", state_a);
        network_not_fused.set_input_data("state_alpha", state_alpha);
        network_not_fused.set_input_data("state_b", state_b);

        auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{this->engine.get_device()});
        auto var_a =     std::make_shared<VariableState>(VariableStateInfo{"var_a",     get_lora_state_layout(p, 0)}, context, network_fused.get_shape_predictor());
        auto var_alpha = std::make_shared<VariableState>(VariableStateInfo{"var_alpha", get_lora_state_layout(p, 1)}, context, network_fused.get_shape_predictor());
        auto var_b =     std::make_shared<VariableState>(VariableStateInfo{"var_b",     get_lora_state_layout(p, 2)}, context, network_fused.get_shape_predictor());

        network_fused.set_variable("var_a", var_a);
        network_fused.set_variable("var_alpha", var_alpha);
        network_fused.set_variable("var_b", var_b);

        network_not_fused.set_variable("var_a", var_a);
        network_not_fused.set_variable("var_alpha", var_alpha);
        network_not_fused.set_variable("var_b", var_b);

        compare(network_not_fused, network_fused, p);

        if (!check_dyn_impl_name.empty()) {
            auto inst = network_fused.get_primitive(check_dyn_impl_name);
            auto impl = inst->get_impl();
            ASSERT_TRUE(impl != nullptr);
            ASSERT_TRUE(impl->is_dynamic());
        }
    }

    layout get_lora_input_layout(lora_test_params& p, bool is_dynamic = false) {
        const auto& pshape = is_dynamic ? ov::PartialShape::dynamic(p.lora_input_pshape.size())
                                        : p.lora_input_pshape;
        return layout{ pshape, p.input_type, p.planar_format };
    }

    layout get_lora_state_layout(lora_test_params& p, size_t state_idx, bool is_dynamic = false) {
        const auto& pshape = is_dynamic ? ov::PartialShape::dynamic(p.lora_states[state_idx].size())
                                        : p.lora_states[state_idx];
        return layout{ pshape, p.input_type, p.planar_format };
    }

    layout get_fc_weights_layout(lora_test_params& p) {
        return layout{ p.fc_weights_pshape, p.input_type, p.weights_format };
    }

    layout get_per_last_dim_layout(lora_test_params& p) {
        return layout{ ov::PartialShape{1, 1, *p.main_input_pshape.rbegin()}, p.input_type, p.planar_format };
    }

    size_t get_fc_input_rank(lora_test_params& p) {
        return p.lora_input_pshape.size();
    }

    size_t get_fc_weights_rank(lora_test_params& p) {
        return p.fc_weights_pshape.size();
    }
};

}  // namespace

/* ----------------------------------------------------------------------------------------------------- */
/* --------------------------------------------- Lora cases -------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

#define CASE_LORA_F32_DEFAULT_OPT { 1, 1, 128 }, { 256, 128 }, { 1, 1, 256 }, {{ 16, 128 }, { 1, 16 }, { 256, 16 }}, data_types::f32, format::bfyx, format::oiyx
#define CASE_LORA_F32_DEFAULT_REF { 1, 1, 128 }, { 256, 128 }, { 1, 1, 256 }, {{ 15, 128 }, { 1, 15 }, { 256, 15 }}, data_types::f32, format::bfyx, format::oiyx
#define CASE_LORA_F32_EMPTY { 1, 1, 128 }, { 256, 128 }, { 1, 1, 256 }, {{ 0, 128 }, { 1, 0 }, { 256, 0 }}, data_types::f32, format::bfyx, format::oiyx

class lora_act_eltw : public LoraFusingsTest {};
TEST_P(lora_act_eltw, basic) {
    // Temporarily disabled
    if (engine.get_device_info().supports_immad) {
        GTEST_SKIP();
    }

    auto p = GetParam();
    create_topologies(
        input_layout("input", get_lora_input_layout(p, true)),
        data("weights", get_mem(get_fc_weights_layout(p))),
        fully_connected("fc_prim", input_info("input"), "weights", "", get_fc_input_rank(p), get_fc_weights_rank(p)),

        input_layout("state_a", get_lora_state_layout(p, 0, true)),
        input_layout("state_alpha", get_lora_state_layout(p, 1, true)),
        input_layout("state_b", get_lora_state_layout(p, 2, true)),
        read_value{"rv_a", { input_info("state_a") }, "var_a", { get_lora_state_layout(p, 0) }},
        read_value{"rv_alpha", { input_info("state_alpha") }, "var_alpha", { get_lora_state_layout(p, 1) }},
        read_value{"rv_b", { input_info("state_b") }, "var_b", { get_lora_state_layout(p, 2) }},
        lora("lora", { input_info("fc_prim"), input_info("input"), input_info("rv_a"), input_info("rv_alpha"), input_info("rv_b") }, true),

        activation("act", input_info("lora"), activation_func::swish),
        data("eltw_data", get_mem(get_per_last_dim_layout(p), 1, 9)),
        eltwise("eltw", { input_info("act"), input_info("eltw_data") }, eltwise_mode::sum, p.input_type),
        reorder("reorder_bfyx", input_info("eltw"), p.planar_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p, "lora");
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, lora_act_eltw, ::testing::ValuesIn(std::vector<lora_test_params>{
    lora_test_params{ CASE_LORA_F32_DEFAULT_OPT, 6, 11 },
    lora_test_params{ CASE_LORA_F32_DEFAULT_REF, 6, 11 },
    lora_test_params{ CASE_LORA_F32_EMPTY, 6, 10 }
}));
