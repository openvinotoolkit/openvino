// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/gather_tree.hpp>

#include <cstddef>
#include <array>

#include <iostream>

using namespace cldnn;
using namespace ::tests;

namespace {
template<typename T>
struct Params {
    tensor step_id_tensor;
    std::vector<T> step_id;
    tensor parent_id_tensor;
    std::vector<T> parent_id;
    tensor max_seq_len_tensor;
    std::vector<T> max_seq_len;
    tensor end_token_tensor;
    std::vector<T> end_token;
    tensor final_id_tensor;
    std::vector<T> final_id;
    std::string testcase_name;
};

template<typename T>
using ParamsWithLayout = std::tuple<
    Params<T>,
    format::type,   // source (plain) layout - bfyx
    format::type,   // target (blocked) layout
    bool            // is_caching_test
>;

const std::vector<format::type> layouts = {
    format::yxfb,
    format::bfyx,
    format::byxf,
    format::b_fs_yx_fsv16,
    format::b_fs_yx_fsv32,
    format::bs_fs_yx_bsv4_fsv4,
    format::bs_fs_yx_bsv8_fsv4,
    format::bs_fs_yx_bsv8_fsv2,
    format::bs_fs_yx_bsv4_fsv2,
    format::bs_fs_yx_bsv16_fsv16,
    format::bs_fs_yx_bsv32_fsv16,
    format::bs_fs_yx_bsv32_fsv32,
};

template<typename T>
std::vector<Params<T>> generateParams() {
    static const std::vector<Params<T>> result = {
        {
            tensor(1, 1, 1, 10),
            std::vector<T>{1, 4, 9, 7, 9, 1, 2, 3, 9, 9},
            tensor(1, 1, 1, 10),
            std::vector<T>{1, 4, 9, 7, 9, 1, 2, 3, 9, 9},
            tensor(1, 1, 1, 1),
            std::vector<T>{9},
            tensor(1, 1, 1, 1),
            std::vector<T>{9},
            tensor(1, 1, 1, 10),
            std::vector<T>{1, 4, 9, 7, 9, 1, 2, 3, 9, 9},
            "gather_tree_1",
        },
        {
            tensor(5, 1, 1, 10),
            std::vector<T>{
                1, 4, 9, 7, 9, 1, 2, 3, 9, 2,
                3, 1, 4, 2, 4, 4, 7, 4, 9, 5,
                8, 4, 3, 7, 5, 2, 4, 8, 3, 1,
                5, 7, 9, 4, 5, 6, 4, 2, 9, 2,
                8, 8, 7, 9, 8, 3, 1, 7, 5, 9},
            tensor(5, 1, 1, 10),
            std::vector<T>{
                1, 4, 9, 7, 9, 1, 2, 3, 9, 2,
                3, 1, 4, 2, 4, 4, 7, 4, 9, 5,
                8, 4, 3, 7, 5, 2, 4, 8, 3, 1,
                5, 7, 9, 4, 5, 6, 4, 2, 9, 2,
                8, 8, 7, 9, 8, 3, 1, 7, 5, 9},
            tensor(1, 1, 1, 1),
            std::vector<T>{9},
            tensor(1, 1, 1, 1),
            std::vector<T>{9},
            tensor(5, 1, 1, 10),
            std::vector<T>{
                4, 4, 9, 9, 4, 9, 2, 9, 9, 9,
                1, 1, 9, 9, 1, 9, 9, 9, 9, 9,
                1, 1, 9, 9, 1, 9, 9, 9, 9, 9,
                9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                9, 9, 9, 9, 9, 9, 9, 9, 9, 9},
            "gather_tree_5",
        },
        {
            tensor(20, 1, 1, 10),
            std::vector<T>{
                1, 4, 9, 7, 9, 1, 2, 3, 9, 2, 3, 1, 4, 2, 4, 4, 7, 4, 9, 5,
                8, 4, 3, 7, 5, 2, 4, 8, 3, 1, 5, 7, 9, 4, 5, 6, 4, 2, 9, 2,
                8, 8, 7, 9, 8, 3, 1, 7, 5, 8, 8, 9, 8, 1, 8, 1, 3, 2, 1, 8,
                7, 1, 6, 4, 7, 9, 4, 5, 2, 7, 3, 3, 2, 7, 8, 8, 4, 1, 1, 7,
                6, 9, 6, 7, 3, 3, 5, 8, 2, 1, 1, 5, 5, 9, 1, 3, 9, 3, 2, 2,
                5, 1, 1, 7, 9, 2, 9, 3, 3, 5, 6, 1, 6, 6, 6, 2, 9, 6, 3, 7,
                3, 1, 5, 4, 9, 7, 5, 4, 5, 1, 7, 5, 1, 6, 2, 5, 8, 9, 1, 6,
                8, 9, 5, 2, 5, 2, 9, 8, 4, 4, 5, 2, 6, 9, 4, 4, 6, 7, 6, 7,
                2, 8, 7, 6, 6, 7, 4, 4, 7, 3, 4, 9, 7, 4, 8, 9, 1, 6, 5, 6,
                1, 2, 8, 9, 1, 5, 4, 6, 9, 4, 4, 3, 7, 9, 7, 6, 3, 1, 7, 9},
            tensor(20, 1, 1, 10),
            std::vector<T>{
                1, 4, 9, 7, 9, 1, 2, 3, 9, 2, 3, 1, 4, 2, 4, 4, 7, 4, 9, 5,
                8, 4, 3, 7, 5, 2, 4, 8, 3, 1, 5, 7, 9, 4, 5, 6, 4, 2, 9, 2,
                8, 8, 7, 9, 8, 3, 1, 7, 5, 8, 8, 9, 8, 1, 8, 1, 3, 2, 1, 8,
                7, 1, 6, 4, 7, 9, 4, 5, 2, 7, 3, 3, 2, 7, 8, 8, 4, 1, 1, 7,
                6, 9, 6, 7, 3, 3, 5, 8, 2, 1, 1, 5, 5, 9, 1, 3, 9, 3, 2, 2,
                5, 1, 1, 7, 9, 2, 9, 3, 3, 5, 6, 1, 6, 6, 6, 2, 9, 6, 3, 7,
                3, 1, 5, 4, 9, 7, 5, 4, 5, 1, 7, 5, 1, 6, 2, 5, 8, 9, 1, 6,
                8, 9, 5, 2, 5, 2, 9, 8, 4, 4, 5, 2, 6, 9, 4, 4, 6, 7, 6, 7,
                2, 8, 7, 6, 6, 7, 4, 4, 7, 3, 4, 9, 7, 4, 8, 9, 1, 6, 5, 6,
                1, 2, 8, 9, 1, 5, 4, 6, 9, 4, 4, 3, 7, 9, 7, 6, 3, 1, 7, 9},
            tensor(1, 1, 1, 1),
            std::vector<T>{9},
            tensor(1, 1, 1, 1),
            std::vector<T>{9},
            tensor(20, 1, 1, 10),
            std::vector<T>{
                9, 4, 9, 4, 4, 4, 9, 4, 9, 9, 9, 1, 9, 1, 1, 1, 9, 1, 9, 9,
                9, 1, 9, 1, 1, 1, 9, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9},
            "gather_tree_10",
        },
    };
    return result;
}

struct PrintToStringParamName {
    template<class T>
    std::string operator()(const testing::TestParamInfo<ParamsWithLayout<T> > &param) {
        std::stringstream buf;
        Params<T> p;
        format::type plain_layout;
        format::type target_layout;
        bool is_caching_test;
        std::tie(p, plain_layout, target_layout, is_caching_test) = param.param;
        buf << " test case " << p.testcase_name
            << " plain layout " << plain_layout
            << " target layout " << target_layout
            << " is_caching_test " << is_caching_test;
        return buf.str();
    }
};
};

template<typename T>
struct gather_tree_test
    : public ::testing::TestWithParam<ParamsWithLayout<T> > {
public:
    void test() {
        const auto data_type = ov::element::from<T>();
        Params<T> params;
        format::type plain_layout;
        format::type target_layout;
        bool is_caching_test;

        std::tie(params, plain_layout, target_layout, is_caching_test) = this->GetParam();

        auto &engine = get_test_engine();
        topology topology;

        auto step_input = engine.allocate_memory({data_type, plain_layout, params.step_id_tensor});
        set_values(step_input, params.step_id);
        const std::string step_id = "step_id";
        topology.add(input_layout(step_id, step_input->get_layout()));
        const std::string reorder_step_id = step_id + "_reordered";
        topology.add(reorder(reorder_step_id, input_info(step_id), target_layout, data_type));

        auto parent_input = engine.allocate_memory({data_type, plain_layout, params.parent_id_tensor});
        set_values(parent_input, params.parent_id);
        const std::string parent_id = "parent_id";
        topology.add(input_layout(parent_id, parent_input->get_layout()));
        const std::string reorder_parent_id = parent_id + "_reordered";
        topology.add(reorder(reorder_parent_id, input_info(parent_id), target_layout, data_type));

        auto max_seq_len_input = engine.allocate_memory({data_type, plain_layout, params.max_seq_len_tensor});
        set_values(max_seq_len_input, params.max_seq_len);
        const std::string max_seq_len_id = "max_seq_len_id";
        topology.add(input_layout(max_seq_len_id, max_seq_len_input->get_layout()));
        const std::string reorder_max_seq_len_id = max_seq_len_id + "_reordered";
        topology.add(reorder(reorder_max_seq_len_id, input_info(max_seq_len_id), target_layout, data_type));

        auto end_token_input = engine.allocate_memory({data_type, plain_layout, params.end_token_tensor});
        set_values(end_token_input, params.end_token);
        const std::string end_token_id = "end_token_id";
        topology.add(input_layout(end_token_id, end_token_input->get_layout()));
        const std::string reorder_end_token_id = end_token_id + "_reordered";
        topology.add(reorder(reorder_end_token_id, input_info(end_token_id), target_layout, data_type));

        const std::string result_id = "result_id";
        topology.add(gather_tree(result_id, input_info(reorder_step_id), input_info(reorder_parent_id), input_info(reorder_max_seq_len_id), input_info(reorder_end_token_id)));

        const primitive_id reorder_result_id = result_id + "_reordered";
        topology.add(reorder(reorder_result_id, input_info(result_id), plain_layout, data_type));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data(step_id, step_input);
        network->set_input_data(parent_id, parent_input);
        network->set_input_data(max_seq_len_id, max_seq_len_input);
        network->set_input_data(end_token_id, end_token_input);

        auto result = network->execute();

        auto out_mem = result.at(reorder_result_id).get_memory();
        cldnn::mem_lock<T> out_ptr(out_mem, get_test_stream());

        ASSERT_EQ(params.final_id_tensor.count(), out_ptr.size());

        for (size_t i = 0; i < params.final_id.size(); ++i) {
            ASSERT_NEAR(params.final_id[i], out_ptr[i], 0.005) << "at i = " << i;
        }
    }
};

using gather_tree_test_f32 = gather_tree_test<float>;
using gather_tree_test_int32 = gather_tree_test<int32_t>;

TEST_P(gather_tree_test_f32, test_case) {
    ASSERT_NO_FATAL_FAILURE(test());
}

TEST_P(gather_tree_test_int32, test_case) {
    ASSERT_NO_FATAL_FAILURE(test());
}

INSTANTIATE_TEST_SUITE_P(gather_tree,
                         gather_tree_test_f32,
                         ::testing::Combine(
                             ::testing::ValuesIn(generateParams<float>()),
                             ::testing::Values(format::bfyx),
                             ::testing::ValuesIn(layouts),
                             ::testing::Values(false)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(gather_tree,
                         gather_tree_test_int32,
                         ::testing::Combine(
                             ::testing::ValuesIn(generateParams<int32_t>()),
                             ::testing::Values(format::bfyx),
                             ::testing::ValuesIn(layouts),
                             ::testing::Values(false)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(export_import,
                         gather_tree_test_int32,
                         ::testing::Combine(
                             ::testing::Values(generateParams<int32_t>()[0]),
                             ::testing::Values(format::bfyx),
                             ::testing::Values(layouts[0]),
                             ::testing::Values(true)),
                         PrintToStringParamName());
