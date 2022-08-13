// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

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
        tensor stepIds_tensor;
        std::vector<T> stepIds;
        tensor parentIdx_tensor;
        std::vector<T> parentIdx;
        tensor maxSeqLen_tensor;
        std::vector<T> maxSeqLen;
        tensor endToken_tensor;
        std::vector<T> endToken;
        tensor finalIdx_tensor;
        std::vector<T> finalIdx;
        std::string testcaseName;
    };

    template<typename T>
    using ParamsWithLayout = std::tuple<
        Params<T>,
        format::type,   // source (plain) layout - bfyx or bfzyx
        format::type    // target (blocked) layout
    >;

    const std::vector<format::type> layouts_2d = {
        format::yxfb,
        format::bfyx,
        format::byxf,

        format::b_fs_yx_fsv16,
        //format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
    };

    template<typename T>
    std::vector<T> getValues(const std::vector<float> &values) {
        std::vector<T> result(values.begin(), values.end());
        return result;
    }

    template<typename T>
    std::vector<Params<T>> generateTileParams2D() {
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
                tensor(1, 5, 1, 10),
                std::vector<T>{
                1, 4, 9, 7, 9, 1, 2, 3, 9, 2,
                3, 1, 4, 2, 4, 4, 7, 4, 9, 5,
                8, 4, 3, 7, 5, 2, 4, 8, 3, 1,
                5, 7, 9, 4, 5, 6, 4, 2, 9, 2,
                8, 8, 7, 9, 8, 3, 1, 7, 5, 9},
                tensor(1, 5, 1, 10),
                std::vector<T>{
                1, 4, 9, 7, 9, 1, 2, 3, 9, 2,
                3, 1, 4, 2, 4, 4, 7, 4, 9, 5,
                8, 4, 3, 7, 5, 2, 4, 8, 3, 1,
                5, 7, 9, 4, 5, 6, 4, 2, 9, 2,
                8, 8, 7, 9, 8, 3, 1, 7, 5, 9},
                tensor(1, 5, 1, 1),
                std::vector<T>{9},
                tensor(1, 1, 1, 1),
                std::vector<T>{9},
                tensor(1, 5, 1, 10),
                std::vector<T>{
                    4, 4, 9, 9, 4, 9, 2, 9, 9, 9,
                    1, 1, 9, 9, 1, 9, 9, 9, 9, 9,
                    1, 1, 9, 9, 1, 9, 9, 9, 9, 9,
                    9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                    9, 9, 9, 9, 9, 9, 9, 9, 9, 9},
                "gather_tree_5",
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
            std::tie(p, plain_layout, target_layout) = param.param;
            buf << " test case " << p.testcaseName
                << " plain layout " << plain_layout
                << " target layout " << target_layout;
            return buf.str();
        }
    };
};

template<typename T>
struct gather_tree_test
    : public ::testing::TestWithParam<ParamsWithLayout<T> > {
public:
    void test() {
        const auto data_type = type_to_data_type<T>::value;
        Params<T> params;
        format::type plain_layout;
        format::type target_layout;

        std::tie(params, plain_layout, target_layout) = this->GetParam();

        const bool need_reorder = target_layout != plain_layout;

        auto &engine = get_test_engine();
        topology topology;

        const std::string step_data_id = "step_id";
        auto step_input = engine.allocate_memory({data_type, plain_layout, params.stepIds_tensor});
        set_values(step_input, params.stepIds);
        std::string step_id = step_data_id;
        topology.add(input_layout(step_id, step_input->get_layout()));
        if (need_reorder) {
            const std::string reorder_step_id = step_id + "_reordered";
            topology.add(reorder(reorder_step_id, step_id, target_layout, data_type));
            step_id = reorder_step_id;
        }

        const std::string parent_data_id = "parent_id";
        auto parent_input = engine.allocate_memory({data_type, plain_layout, params.parentIdx_tensor});
        set_values(parent_input, params.parentIdx);
        std::string parent_id = parent_data_id;
        topology.add(input_layout(parent_id, parent_input->get_layout()));
        if (need_reorder) {
            const std::string reorder_parent_id = parent_id + "_reordered";
            topology.add(reorder(reorder_parent_id, parent_id, target_layout, data_type));
            parent_id = reorder_parent_id;
        }

        const std::string max_seq_len_data_id = "max_seq_len_id";
        auto max_seq_len_input = engine.allocate_memory({data_type, plain_layout, params.maxSeqLen_tensor});
        set_values(max_seq_len_input, params.maxSeqLen);
        std::string max_seq_len_id = max_seq_len_data_id;
        topology.add(input_layout(max_seq_len_id, max_seq_len_input->get_layout()));
        if (need_reorder) {
            const std::string reorder_max_seq_len_id = max_seq_len_id + "_reordered";
            topology.add(reorder(reorder_max_seq_len_id, max_seq_len_id, target_layout, data_type));
            max_seq_len_id = reorder_max_seq_len_id;
        }

        const std::string end_token_data_id = "end_token_id";
        auto end_token_input = engine.allocate_memory({data_type, plain_layout, params.endToken_tensor});
        set_values(end_token_input, params.endToken);
        std::string end_token_id = end_token_data_id;
        topology.add(input_layout(end_token_id, end_token_input->get_layout()));
        if (need_reorder) {
            const std::string reorder_end_token_id = end_token_id + "_reordered";
            topology.add(reorder(reorder_end_token_id, end_token_id, target_layout, data_type));
            end_token_id = reorder_end_token_id;
        }

        const std::string result_data_id = "result_id";
        topology.add(gather_tree(result_data_id,
                                 step_id,
                                 parent_id,
                                 max_seq_len_id,
                                 end_token_id));

        std::string result_id = result_data_id;
        if (need_reorder) {
            const primitive_id reorder_result_id = result_data_id + "_reordered";
            topology.add(reorder(reorder_result_id, result_data_id, plain_layout, data_type));
            result_id = reorder_result_id;
        }

        network network(engine, topology);

        network.set_input_data(step_data_id, step_input);
        network.set_input_data(parent_data_id, parent_input);
        network.set_input_data(max_seq_len_data_id, max_seq_len_input);
        network.set_input_data(end_token_data_id, end_token_input);

        auto result = network.execute();

        auto out_mem = result.at(result_id).get_memory();
        cldnn::mem_lock<T> out_ptr(out_mem, get_test_stream());

        ASSERT_EQ(params.finalIdx_tensor.count(), out_ptr.size());

        for (size_t i = 0; i < params.finalIdx.size(); ++i) {
            EXPECT_NEAR(params.finalIdx[i], out_ptr[i], 0.005) << "at i = " << i;
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
                             ::testing::ValuesIn(generateTileParams2D<float>()),
                             ::testing::Values(format::bfyx),
                             ::testing::ValuesIn(layouts_2d)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(gather_tree,
                         gather_tree_test_int32,
                         ::testing::Combine(
                             ::testing::ValuesIn(generateTileParams2D<int32_t>()),
                             ::testing::Values(format::bfyx),
                             ::testing::ValuesIn(layouts_2d)),
                         PrintToStringParamName());
