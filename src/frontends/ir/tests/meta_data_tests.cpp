// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>

#include "common_test_utils/common_utils.hpp"
#include "openvino/core/any.hpp"
#include "openvino/openvino.hpp"
#include "openvino/util/file_util.hpp"

class MetaData : public ::testing::Test {
public:
    ov::Core core;

    std::string ir_with_meta = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="ReLU" version="opset1">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
    <meta_data>
        <MO_version value="TestVersion"/>
        <Runtime_version value="TestVersion"/>
        <cli_parameters>
            <input_shape value="[1, 3, 22, 22]"/>
            <transform value=""/>
            <use_new_frontend value="False"/>
        </cli_parameters>
    </meta_data>
    <quantization_parameters>
        <config>{
        'compression': {
            'algorithms': [
                {
                    'name': 'DefaultQuantization',
                    'params': {
                        'num_samples_for_tuning': 2000,
                        'preset': 'performance',
                        'stat_subset_size': 300,
                        'use_layerwise_tuning': false
                    }
                }
            ],
            'dump_intermediate_model': true,
            'target_device': 'ANY'
        },
        'engine': {
            'models': [
                {
                    'name': 'bert-small-uncased-whole-word-masking-squad-0001',
                    'launchers': [
                        {
                            'framework': 'openvino',
                            'adapter': {
                                'type': 'bert_question_answering',
                                'start_token_logits_output': 'output_s',
                                'end_token_logits_output': 'output_e'
                            },
                            'inputs': [
                                {
                                    'name': 'input_ids',
                                    'type': 'INPUT',
                                    'value': 'input_ids'
                                },
                                {
                                    'name': 'attention_mask',
                                    'type': 'INPUT',
                                    'value': 'input_mask'
                                },
                                {
                                    'name': 'token_type_ids',
                                    'type': 'INPUT',
                                    'value': 'segment_ids'
                                }
                            ],
                            'device': 'cpu'
                        }
                    ],
                    'datasets': [
                        {
                            'name': 'squad_v1_1_msl384_mql64_ds128_lowercase',
                            'annotation_conversion': {
                                'converter': 'squad',
                                'testing_file': 'PATH',
                                'max_seq_length': 384,
                                'max_query_length': 64,
                                'doc_stride': 128,
                                'lower_case': true,
                                'vocab_file': 'PATH'
                            },
                            'reader': {
                                'type': 'annotation_features_extractor',
                                'features': [
                                    'input_ids',
                                    'input_mask',
                                    'segment_ids'
                                ]
                            },
                            'postprocessing': [
                                {
                                    'type': 'extract_answers_tokens',
                                    'max_answer': 30,
                                    'n_best_size': 20
                                }
                            ],
                            'metrics': [
                                {
                                    'name': 'F1',
                                    'type': 'f1',
                                    'reference': 0.9157
                                },
                                {
                                    'name': 'EM',
                                    'type': 'exact_match',
                                    'reference': 0.8504
                                }
                            ],
                            '_command_line_mapping': {
                                'testing_file': 'PATH',
                                'vocab_file': [
                                    'PATH'
                                ]
                            }
                        }
                    ]
                }
            ],
            'stat_requests_number': null,
            'eval_requests_number': null,
            'type': 'accuracy_checker'
        }
    }</config>
        <version value="invalid version"/>
        <cli_params value="{'quantize': None, 'preset': None, 'model': None, 'weights': None, 'name': None, 'engine': None, 'ac_config': None, 'max_drop': None, 'evaluate': False, 'output_dir': 'PATH', 'direct_dump': True, 'log_level': 'INFO', 'pbar': False, 'stream_output': False, 'keep_uncompressed_weights': False, 'data_source': None}"/>
    </quantization_parameters>
</net>
)V0G0N";

    std::string ir_with_new_meta = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="ReLU" version="opset1">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
    <rt_info>
        <MO_version value="TestVersion" />
        <Runtime_version value="TestVersion" />
        <conversion_parameters>
            <input_shape value="[1, 3, 22, 22]" />
            <transform value="" />
            <use_new_frontend value="False" />
        </conversion_parameters>
        <optimization>
            <cli_params value="{'quantize': None, 'preset': None, 'model': None, 'weights': None, 'name': None, 'engine': None, 'ac_config': None, 'max_drop': None, 'evaluate': False, 'output_dir': 'PATH', 'direct_dump': True, 'log_level': 'INFO', 'pbar': False, 'stream_output': False, 'keep_uncompressed_weights': False, 'data_source': None}" />
            <config value="{ 'compression': { 'algorithms': [ { 'name': 'DefaultQuantization', 'params': { 'num_samples_for_tuning': 2000, 'preset': 'performance', 'stat_subset_size': 300, 'use_layerwise_tuning': false } } ], 'dump_intermediate_model': true, 'target_device': 'ANY' }, 'engine': { 'models': [ { 'name': 'bert-small-uncased-whole-word-masking-squad-0001', 'launchers': [ { 'framework': 'openvino', 'adapter': { 'type': 'bert_question_answering', 'start_token_logits_output': 'output_s', 'end_token_logits_output': 'output_e' }, 'inputs': [ { 'name': 'input_ids', 'type': 'INPUT', 'value': 'input_ids' }, { 'name': 'attention_mask', 'type': 'INPUT', 'value': 'input_mask' }, { 'name': 'token_type_ids', 'type': 'INPUT', 'value': 'segment_ids' } ], 'device': 'cpu' } ], 'datasets': [ { 'name': 'squad_v1_1_msl384_mql64_ds128_lowercase', 'annotation_conversion': { 'converter': 'squad', 'testing_file': 'PATH', 'max_seq_length': 384, 'max_query_length': 64, 'doc_stride': 128, 'lower_case': true, 'vocab_file': 'PATH' }, 'reader': { 'type': 'annotation_features_extractor', 'features': [ 'input_ids', 'input_mask', 'segment_ids' ] }, 'postprocessing': [ { 'type': 'extract_answers_tokens', 'max_answer': 30, 'n_best_size': 20 } ], 'metrics': [ { 'name': 'F1', 'type': 'f1', 'reference': 0.9157 }, { 'name': 'EM', 'type': 'exact_match', 'reference': 0.8504 } ], '_command_line_mapping': { 'testing_file': 'PATH', 'vocab_file': [ 'PATH' ] } } ] } ], 'stat_requests_number': null, 'eval_requests_number': null, 'type': 'accuracy_checker' } }" />
            <version value="invalid version" />
        </optimization>
        <framework>
            <batch value="1"/>
            <chunk_size value="16"/>
        </framework>
    </rt_info>
</net>
)V0G0N";

    std::string ir_without_meta = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="ReLU" version="opset1">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</net>
)V0G0N";

    void SetUp() override {}
    void check_rt_info(const std::shared_ptr<ov::Model>& model, bool with_framework = false) {
        const std::string pot_conf_ref =
            "{ 'compression': { 'algorithms': [ { 'name': 'DefaultQuantization', 'params': { 'num_samples_for_tuning': "
            "2000, 'preset': 'performance', 'stat_subset_size': 300, 'use_layerwise_tuning': false } } ], "
            "'dump_intermediate_model': true, 'target_device': 'ANY' }, 'engine': { 'models': [ { 'name': "
            "'bert-small-uncased-whole-word-masking-squad-0001', 'launchers': [ { 'framework': 'openvino', 'adapter': "
            "{ 'type': 'bert_question_answering', 'start_token_logits_output': 'output_s', 'end_token_logits_output': "
            "'output_e' }, 'inputs': [ { 'name': 'input_ids', 'type': 'INPUT', 'value': 'input_ids' }, { 'name': "
            "'attention_mask', 'type': 'INPUT', 'value': 'input_mask' }, { 'name': 'token_type_ids', 'type': 'INPUT', "
            "'value': 'segment_ids' } ], 'device': 'cpu' } ], 'datasets': [ { 'name': "
            "'squad_v1_1_msl384_mql64_ds128_lowercase', 'annotation_conversion': { 'converter': 'squad', "
            "'testing_file': 'PATH', 'max_seq_length': 384, 'max_query_length': 64, 'doc_stride': 128, 'lower_case': "
            "true, 'vocab_file': 'PATH' }, 'reader': { 'type': 'annotation_features_extractor', 'features': [ "
            "'input_ids', 'input_mask', 'segment_ids' ] }, 'postprocessing': [ { 'type': 'extract_answers_tokens', "
            "'max_answer': 30, 'n_best_size': 20 } ], 'metrics': [ { 'name': 'F1', 'type': 'f1', 'reference': 0.9157 "
            "}, { 'name': 'EM', 'type': 'exact_match', 'reference': 0.8504 } ], '_command_line_mapping': { "
            "'testing_file': 'PATH', 'vocab_file': [ 'PATH' ] } } ] } ], 'stat_requests_number': null, "
            "'eval_requests_number': null, 'type': 'accuracy_checker' } }";
        auto& rt_info = model->get_rt_info();
        ASSERT_TRUE(!rt_info.empty());
        std::string value;
        EXPECT_NO_THROW(value = model->get_rt_info<std::string>("MO_version"));
        EXPECT_EQ(value, "TestVersion");
        value = "";

        EXPECT_NO_THROW(value = model->get_rt_info<std::string>("Runtime_version"));
        EXPECT_EQ(value, "TestVersion");

        ov::AnyMap cli_map;
        EXPECT_NO_THROW(cli_map = model->get_rt_info<ov::AnyMap>("conversion_parameters"));

        auto it = cli_map.find("input_shape");
        ASSERT_NE(it, cli_map.end());
        EXPECT_TRUE(it->second.is<std::string>());
        EXPECT_EQ(it->second.as<std::string>(), "[1, 3, 22, 22]");

        it = cli_map.find("transform");
        ASSERT_NE(it, cli_map.end());
        EXPECT_TRUE(it->second.is<std::string>());
        EXPECT_EQ(it->second.as<std::string>(), "");

        it = cli_map.find("use_new_frontend");
        ASSERT_NE(it, cli_map.end());
        EXPECT_TRUE(it->second.is<std::string>());
        EXPECT_EQ(it->second.as<std::string>(), "False");

        EXPECT_NO_THROW(value = model->get_rt_info<std::string>("optimization", "config"));
        EXPECT_EQ(value, pot_conf_ref);
        if (with_framework) {
            EXPECT_NO_THROW(value = model->get_rt_info<std::string>("framework", "batch"));
            EXPECT_EQ(value, "1");
            EXPECT_NO_THROW(value = model->get_rt_info<std::string>("framework", "chunk_size"));
            EXPECT_EQ(value, "16");
        }
    }
};

TEST_F(MetaData, get_meta_data_from_model_without_info) {
    auto model = core.read_model(ir_without_meta, ov::Tensor());

    auto& rt_info = model->get_rt_info();
    ASSERT_EQ(rt_info.find("meta_data"), rt_info.end());
}

TEST_F(MetaData, get_meta_data_as_map_from_model_without_info) {
    auto model = core.read_model(ir_without_meta, ov::Tensor());

    auto& rt_info = model->get_rt_info();
    auto it = rt_info.find("meta_data");
    EXPECT_EQ(it, rt_info.end());
    it = rt_info.find("MO_version");
    EXPECT_EQ(it, rt_info.end());
    ov::AnyMap meta;
    ASSERT_THROW(meta = model->get_rt_info<ov::AnyMap>("meta_data"), ov::Exception);
    ASSERT_TRUE(meta.empty());
}

TEST_F(MetaData, get_meta_data) {
    auto model = core.read_model(ir_with_meta, ov::Tensor());

    auto& rt_info = model->get_rt_info();
    ASSERT_NE(rt_info.find("MO_version"), rt_info.end());
    ASSERT_NE(rt_info.find("Runtime_version"), rt_info.end());
    ASSERT_NE(rt_info.find("conversion_parameters"), rt_info.end());
    ASSERT_NE(rt_info.find("optimization"), rt_info.end());
}

TEST_F(MetaData, get_meta_data_as_map) {
    auto model = core.read_model(ir_with_meta, ov::Tensor());

    check_rt_info(model);
}

TEST_F(MetaData, get_meta_data_from_removed_file) {
    std::string file_path = ov::util::get_ov_lib_path() + ov::util::FileTraits<char>::file_separator +
                            ov::test::utils::generateTestFilePrefix() + "_test_model.xml";
    // Create file
    {
        std::ofstream ir(file_path);
        ir << ir_with_meta;
    }
    auto model = core.read_model(file_path);

    // Remove file (meta section wasn't read)
    std::remove(file_path.c_str());

    check_rt_info(model);
}

TEST_F(MetaData, get_meta_data_as_map_from_new_format) {
    auto model = core.read_model(ir_with_new_meta, ov::Tensor());

    check_rt_info(model, true);
}
