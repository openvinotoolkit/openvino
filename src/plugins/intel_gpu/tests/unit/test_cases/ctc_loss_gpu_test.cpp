// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <test_utils/test_utils.h>

#include <intel_gpu/primitives/ctc_loss.hpp>
#include <vector>

using namespace cldnn;
using namespace tests;

namespace {

template <class TF, class TI>
struct ctc_loss_test_inputs {
    bool preprocess_collapse_repeated;
    bool ctc_merge_repeated;
    bool unique;
    std::vector<int> logits_shape;
    std::vector<TF> logits;
    std::vector<TI> logit_length;
    std::vector<TI> labels;
    std::vector<TI> label_length;
    TI blank_index;
    std::vector<TF> expected_values;
};

template <class TF, class TI>
using ctc_loss_test_params = std::tuple<ctc_loss_test_inputs<TF, TI>, format::type, bool>;

template <class TF, class TI>
struct ctc_loss_gpu_test : public testing::TestWithParam<ctc_loss_test_params<TF, TI>> {
public:
    void test() {
        const auto& [p, fmt, is_caching_test] = testing::TestWithParam<ctc_loss_test_params<TF, TI>>::GetParam();

        auto& engine = get_test_engine();
        const auto float_data_type = ov::element::from<TF>();
        const auto int_data_type = ov::element::from<TI>();
        const auto plane_format = format::bfyx;
        std::vector<std::tuple<primitive_id, memory::ptr, data_types>> inputs;

        const auto batch_num = p.logits_shape[0];
        const auto max_time = p.logits_shape[1];
        const auto classes_num = p.logits_shape[2];

        const layout logits_layout(float_data_type,
                                   plane_format,
                                   tensor(plane_format, {batch_num, max_time, classes_num, 1}));
        auto logits = engine.allocate_memory(logits_layout);
        set_values(logits, p.logits);
        inputs.emplace_back("logits", logits, float_data_type);

        const layout logit_length_layout(int_data_type, plane_format, tensor(plane_format, {1, batch_num, 1, 1}));
        auto logit_length = engine.allocate_memory(logit_length_layout);
        set_values(logit_length, p.logit_length);
        inputs.emplace_back("logit_length", logit_length, int_data_type);

        const layout labels_layout(int_data_type, plane_format, tensor(plane_format, {batch_num, max_time, 1, 1}));
        auto labels = engine.allocate_memory(labels_layout);
        set_values(labels, p.labels);
        inputs.emplace_back("labels", labels, int_data_type);

        const layout label_length_layout(int_data_type, plane_format, tensor(plane_format, {1, batch_num, 1, 1}));
        auto label_length = engine.allocate_memory(label_length_layout);
        set_values(label_length, p.label_length);
        inputs.emplace_back("label_length", label_length, int_data_type);

        const layout blank_index_layout(int_data_type, plane_format, tensor(plane_format, {1, 1, 1, 1}));
        auto blank_index = engine.allocate_memory(blank_index_layout);
        set_values(blank_index, {p.blank_index});
        inputs.emplace_back("blank_index", blank_index, int_data_type);

        std::vector<input_info> inputs_ids;
        std::transform(inputs.begin(),
                       inputs.end(),
                       std::back_inserter(inputs_ids),
                       [](const decltype(inputs)::value_type& input) {
                           return input_info("reordered_" + std::get<0>(input));
                       });

        topology topology;
        for (const auto& input : inputs) {
            topology.add(input_layout(std::get<0>(input), std::get<1>(input)->get_layout()));
            topology.add(reorder("reordered_" + std::get<0>(input), input_info(std::get<0>(input)), fmt, std::get<2>(input)));
        }

        topology.add(ctc_loss("ctc_loss", inputs_ids, p.preprocess_collapse_repeated, p.ctc_merge_repeated, p.unique));
        topology.add(reorder("reordered_ctc_loss", input_info("ctc_loss"), plane_format, float_data_type));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        for (auto& input : inputs) {
            network->set_input_data(std::get<0>(input), std::get<1>(input));
        }
        const auto outputs = network->execute();

        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "reordered_ctc_loss");

        auto output = outputs.at("reordered_ctc_loss").get_memory();
        cldnn::mem_lock<TF> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), p.expected_values.size());
        for (size_t i = 0; i < output_ptr.size(); ++i) {
            ASSERT_NEAR(p.expected_values[i], output_ptr[i], 0.1);
        }
    }

    static std::string PrintToStringParamName(const testing::TestParamInfo<ctc_loss_test_params<TF, TI>>& info) {
        const auto& [p, fmt, is_caching_test] = info.param;

        std::ostringstream result;
        result << "PreprocessCollapseRepeated=" << p.preprocess_collapse_repeated << "_";
        result << "CtcMergeRepeated=" << p.ctc_merge_repeated << "_";
        result << "Unique=" << p.unique << "_";
        result << "LogitsShape=" << vec2str(p.logits_shape) << "_";
        result << "LogitsLength=" << vec2str(p.logit_length) << "_";
        result << "Labels=" << vec2str(p.labels) << "_";
        result << "LabelLength=" << vec2str(p.label_length) << "_";
        result << "BlankIndex=" << p.blank_index << "_";
        result << "Format=" << fmt_to_str(fmt);
        result << "is_caching_test=" << is_caching_test;
        return result.str();
    }
};

template <class TF, class TI>
std::vector<ctc_loss_test_inputs<TF, TI>> getCTCLossParams() {
    return {
        {
            false,                                                    // preprocess_collapse_repeated
            false,                                                    // ctc_merge_repeated
            false,                                                    // unique
            {2, 3, 3},                                                // logits_shape
            {0, 1, 8, 5, 5, 2, 0, 7, 7, 10, 4, 5, 9, 0, 0, 5, 7, 0},  // logits
            {3, 3},                                                   // logit_length
            {0, 1, 2, 1, 1, 1},                                       // labels
            {2, 1},                                                   // label_length
            2,                                                        // blank_index
            {1.41223f, 14.1359f},                                     // expected_values
        },
        {
            false,                                                    // preprocess_collapse_repeated
            false,                                                    // ctc_merge_repeated
            true,                                                     // unique
            {2, 3, 3},                                                // logits_shape
            {0, 1, 8, 5, 5, 2, 0, 7, 7, 10, 4, 5, 9, 0, 0, 5, 7, 0},  // logits
            {3, 3},                                                   // logit_length
            {0, 1, 2, 1, 1, 1},                                       // labels
            {2, 1},                                                   // label_length
            2,                                                        // blank_index
            {1.41223f, 14.1359f},                                     // expected_values
        },
        {
            false,                                                    // preprocess_collapse_repeated
            true,                                                     // ctc_merge_repeated
            false,                                                    // unique
            {2, 3, 3},                                                // logits_shape
            {0, 1, 8, 5, 5, 2, 0, 7, 7, 10, 4, 5, 9, 0, 0, 5, 7, 0},  // logits
            {3, 3},                                                   // logit_length
            {0, 1, 2, 1, 1, 1},                                       // labels
            {2, 1},                                                   // label_length
            2,                                                        // blank_index
            {1.41156f, 13.2745f},                                     // expected_values
        },
        {
            true,                                                     // preprocess_collapse_repeated
            false,                                                    // ctc_merge_repeated
            false,                                                    // unique
            {2, 3, 3},                                                // logits_shape
            {0, 1, 8, 5, 5, 2, 0, 7, 7, 10, 4, 5, 9, 0, 0, 5, 7, 0},  // logits
            {3, 3},                                                   // logit_length
            {0, 1, 2, 1, 1, 1},                                       // labels
            {2, 1},                                                   // label_length
            2,                                                        // blank_index
            {1.41223f, 14.1359f},                                     // expected_values
        },
        {
            false,                                                    // preprocess_collapse_repeated
            true,                                                     // ctc_merge_repeated
            true,                                                     // unique
            {2, 3, 3},                                                // logits_shape
            {0, 1, 8, 5, 5, 2, 0, 7, 7, 10, 4, 5, 9, 0, 0, 5, 7, 0},  // logits
            {3, 3},                                                   // logit_length
            {0, 1, 2, 1, 1, 1},                                       // labels
            {2, 1},                                                   // label_length
            2,                                                        // blank_index
            {1.41156f, 13.2745f},                                     // expected_values
        },
        {
            true,                                                     // preprocess_collapse_repeated
            true,                                                     // ctc_merge_repeated
            true,                                                     // unique
            {2, 3, 3},                                                // logits_shape
            {0, 1, 8, 5, 5, 2, 0, 7, 7, 10, 4, 5, 9, 0, 0, 5, 7, 0},  // logits
            {3, 3},                                                   // logit_length
            {0, 1, 2, 1, 1, 1},                                       // labels
            {2, 1},                                                   // label_length
            2,                                                        // blank_index
            {1.41223f, 13.2745f},                                     // expected_values
        },
    };
}

const std::vector<format::type> layout_formats = {
    format::bfyx,
    format::b_fs_yx_fsv16,
    format::b_fs_yx_fsv32,
    format::bs_fs_yx_bsv16_fsv16,
    format::bs_fs_yx_bsv32_fsv32,
    format::bs_fs_yx_bsv32_fsv16,
};

#define INSTANTIATE_CTC_LOSS_TEST_SUITE(float_type, int_type)                                              \
    using ctc_loss_gpu_test_##float_type##int_type = ctc_loss_gpu_test<float_type, int_type>;              \
    TEST_P(ctc_loss_gpu_test_##float_type##int_type, test) { ASSERT_NO_FATAL_FAILURE(test()); }            \
    INSTANTIATE_TEST_SUITE_P(smoke_ctc_loss_##float_type##int_type,                                        \
                             ctc_loss_gpu_test_##float_type##int_type,                                     \
                             testing::Combine(testing::ValuesIn(getCTCLossParams<float_type, int_type>()), \
                                              testing::ValuesIn(layout_formats),                           \
                                              testing::Values(false)),                                     \
                             ctc_loss_gpu_test_##float_type##int_type::PrintToStringParamName);

using ov::float16;
INSTANTIATE_CTC_LOSS_TEST_SUITE(float, int64_t);
INSTANTIATE_CTC_LOSS_TEST_SUITE(float16, int32_t);
INSTANTIATE_TEST_SUITE_P(export_import,
                         ctc_loss_gpu_test_float16int32_t,
                         testing::Combine(testing::Values(getCTCLossParams<ov::float16, int32_t>()[0]),
                                         testing::Values(layout_formats[0]),
                                         testing::Values(true)),
                         ctc_loss_gpu_test_float16int32_t::PrintToStringParamName);

}  // namespace
