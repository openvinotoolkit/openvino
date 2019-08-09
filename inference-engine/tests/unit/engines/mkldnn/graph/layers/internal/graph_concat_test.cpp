// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "mkldnn_plugin/mkldnn_graph.h"

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include <mkldnn_plugin/mkldnn_extension_utils.h>
#include <unordered_set>
#include <inference_engine/cnn_network_impl.hpp>
#include "tests_common.hpp"

using namespace ::testing;
using namespace std;
using namespace mkldnn;


struct concat_test_params {
    // Formats: NCHW, NCDHW
    vector<size_t> in1;
    vector<size_t> in2;

    size_t axis;

    size_t num_prim_desc;

    MKLDNNPlugin::impl_desc_type selectedType;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

class MKLDNNGraphConcatTests: public TestsCommon,
                              public WithParamInterface<concat_test_params> {
    std::string model_t = R"V0G0N(
<net name="ConcatOnly" version="3" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">__SRC_DIMS_1__
                </port>
            </output>
        </layer>
        <layer name="in2" type="Input" precision="FP32" id="2">
            <output>
                <port id="2">__SRC_DIMS_2__
                </port>
            </output>
        </layer>
        <layer name="con" id="3" type="Concat" precision="FP32">
            <concat_data axis="_AXIS_"/>
            <input>
                <port id="1">__SRC_DIMS_1__
                </port>
                <port id="2">__SRC_DIMS_2__
                </port>
            </input>
            <output>
                <port id="3">__DST_DIMS__
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="2"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(concat_test_params p) {
        std::string model = model_t;
        std::string s_dims;
        for (auto& dim : p.in1) {
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(dim) + "</dim>";
        }
	REPLACE_WITH_STR(model, "__SRC_DIMS_1__", s_dims);

        s_dims = "";
        for (auto& dim : p.in2) {
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(dim) + "</dim>";
        }
	REPLACE_WITH_STR(model, "__SRC_DIMS_2__", s_dims);

        s_dims = "";
        for (size_t i = 0; i < p.in1.size(); i++) {
            size_t dim = p.axis == i ? p.in1[i] + p.in2[i] : p.in1[i];
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(dim) + "</dim>";
        }
	REPLACE_WITH_STR(model, "__DST_DIMS__", s_dims);

        REPLACE_WITH_NUM(model, "_AXIS_", p.axis);
        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            concat_test_params p = ::testing::WithParamInterface<concat_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());
            auto& nodes = graph.getNodes();
            for (int i = 0; i < nodes.size(); i++) {
                if (nodes[i]->getType() == MKLDNNPlugin::Concatenation) {
                    ASSERT_EQ(p.num_prim_desc, nodes[i]->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(nodes[i]->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, nodes[i]->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType, nodes[i]->getSelectedPrimitiveDescriptor()->getImplementationType());
                }
            }
            ASSERT_LE(3, nodes.size());

            InferenceEngine::SizeVector dims_src1 = p.in1;
            InferenceEngine::SizeVector dims_src2 = p.in2;
            InferenceEngine::Layout layout = InferenceEngine::ANY;
            switch (p.in1.size()) {
                case 4:
                    layout = InferenceEngine::NCHW;
                    break;
                case 5:
                    layout = InferenceEngine::NCDHW;
                    break;
            }

            InferenceEngine::Blob::Ptr src1 = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src1, layout});
            src1->allocate();

            fill_data(src1->buffer(), src1->size());
            InferenceEngine::Blob::Ptr src2 = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src2, layout});
            src2->allocate();
            fill_data(src2->buffer(), src2->size());
            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src1));
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in2", src2));

            InferenceEngine::OutputsDataMap out;
            out = net_reader.getNetwork().getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            graph.Infer(srcs, outputBlobs);

            // Compare
            float *src1_ptr = src1->buffer();
            size_t src1_size = src1->size();
            float *src2_ptr = src2->buffer();
            size_t src2_size = src2->size();
            float *dst_ptr = output->buffer();
            size_t dst_size = output->size();

            int len1 = 1, len2 = 1, cycles;
            for (int dim = p.axis; dim < output->getTensorDesc().getDims().size(); dim++) {
                len1 *= src1->getTensorDesc().getDims()[dim];
                len2 *= src2->getTensorDesc().getDims()[dim];
            }
            cycles = p.axis;


            int index1 = 0, index2 = 0, index = 0;
            for (int cycle = 0; cycle < cycles; cycle ++) {
                for (int i1 = 0; i1 < len1; i1++) {
                    if (src1_ptr[index1] != dst_ptr[index])
                    {
                        FAIL() << "index: " << index << " src: " << src1_ptr[index1] << ", dst: " << dst_ptr[index];
                    }
                    index1++; index++;
                }
                for (int i2 = 0; i2 < len2; i2++) {
                    if (src2_ptr[index2] != dst_ptr[index])
                    {
                        FAIL() << "index: " << index << " src: " << src2_ptr[index2] << ", dst: " << dst_ptr[index];
                    }
                    index2++; index++;
                }
            }
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphConcatTests, TestsConcat) {}

INSTANTIATE_TEST_CASE_P(
        TestsConcat, MKLDNNGraphConcatTests,
        ::testing::Values(
                concat_test_params {
                        {1, 3, 3, 5},
                        {1, 3, 3, 5},
                        1, 2
                },
                concat_test_params {
                        {1, 7, 1, 5},
                        {1, 7, 9, 5},
                        2, 1, MKLDNNPlugin::impl_desc_type::ref
                },
                concat_test_params {
                        {1, 2, 3, 5, 3},
                        {1, 5, 3, 5, 3},
                        1, 2
                },
                concat_test_params {
                        {1, 32, 3, 4, 5},
                        {1, 32, 3, 4, 5},
                        1, 6, MKLDNNPlugin::impl_desc_type::unknown
                },
                concat_test_params {
                        {1, 64, 16, 16, 16, 1},
                        {1, 64, 16, 16, 16, 1},
                        5, 1, MKLDNNPlugin::impl_desc_type::ref
                }));

class MKLDNNGraphDynBatchConcatTests: public TestsCommon, public WithParamInterface<concat_test_params> {
    std::string model_t = R"V0G0N(
<net name="ConcatOnly" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    <dim>1</dim>__SRC_DIMS_1__
                </port>
            </output>
        </layer>
        <layer name="in2" type="Input" precision="FP32" id="2">
            <output>
                <port id="2">
                    <dim>1</dim>__SRC_DIMS_2__
                </port>
            </output>
        </layer>
        <layer name="con" id="3" type="Concat" precision="FP32">
            <concat_data axis="_AXIS_"/>
            <input>
                <port id="1">
                    <dim>1</dim>__SRC_DIMS_1__
                </port>
                <port id="2">
                    <dim>1</dim>__SRC_DIMS_2__
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>__DST_DIMS__
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="2"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(concat_test_params p) {
        std::string model = model_t;
        std::string s_dims;
        for (size_t i = 1; i < p.in1.size(); i++) {
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(p.in1[i]) + "</dim>";
        }
	REPLACE_WITH_STR(model, "__SRC_DIMS_1__", s_dims);

        s_dims = "";
        for (size_t i = 1; i < p.in2.size(); i++) {
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(p.in2[i]) + "</dim>";
        }
	REPLACE_WITH_STR(model, "__SRC_DIMS_2__", s_dims);

        s_dims = "";
        for (size_t i = 1; i < p.in1.size(); i++) {
            size_t dim = p.axis == i ? p.in1[i] + p.in2[i] : p.in1[i];
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(dim) + "</dim>";
        }
	REPLACE_WITH_STR(model, "__DST_DIMS__", s_dims);

        REPLACE_WITH_NUM(model, "_AXIS_", p.axis);
        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            concat_test_params p = ::testing::WithParamInterface<concat_test_params>::GetParam();
            std::string model = getModel(p);
            size_t MB = p.in1[0];
            if (MB < 2)
                MB = 2;

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));
            InferenceEngine::CNNNetwork network = net_reader.getNetwork();
            auto implNet = dynamic_cast<InferenceEngine::details::CNNNetworkImpl *>(&((InferenceEngine::ICNNNetwork&)network));
            ASSERT_NE(nullptr, implNet) << "Failed to cast ICNNNetwork to CNNNetworkImpl";
            InferenceEngine::ResponseDesc resp;
            InferenceEngine::StatusCode sts  = implNet->setBatchSizeReshape(MB, &resp);
            ASSERT_EQ((int)InferenceEngine::StatusCode::OK, sts) << resp.msg;

            MKLDNNGraphTestClass graph;
            graph.setProperty({{InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED, InferenceEngine::PluginConfigParams::YES}});
            graph.CreateGraph(net_reader.getNetwork());

            InferenceEngine::SizeVector dims_src1 = p.in1;
            InferenceEngine::SizeVector dims_src2 = p.in2;
            InferenceEngine::Layout layout = InferenceEngine::ANY;
            switch (p.in1.size()) {
                case 4:
                    layout = InferenceEngine::NCHW;
                    break;
                case 5:
                    layout = InferenceEngine::NCDHW;
                    break;
            }

            InferenceEngine::Blob::Ptr src1 = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src1, layout});
            src1->allocate();

            fill_data(src1->buffer(), src1->size());
            InferenceEngine::Blob::Ptr src2 = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src2, layout});
            src2->allocate();
            fill_data(src2->buffer(), src2->size());
            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src1));
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in2", src2));

            InferenceEngine::OutputsDataMap out;
            out = net_reader.getNetwork().getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;


            auto checkConcat = [](const MKLDNNPlugin::MKLDNNNodePtr& node) {
                return node->getType() == MKLDNNPlugin::Concatenation;
            };

            MKLDNNGraphTestClass::CheckDynBatchType checkType = MKLDNNGraphTestClass::CheckDynBatchType::Both;
            if (p.selectedType == MKLDNNPlugin::impl_desc_type::unknown)
                checkType = MKLDNNGraphTestClass::CheckDynBatchType::Child;

            graph.checkDynBatch(srcs, outputBlobs, MB, MB, checkConcat, checkType);
            graph.checkDynBatch(srcs, outputBlobs, 1, MB, checkConcat, checkType);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphDynBatchConcatTests, TestsDynBatchConcat) {}


INSTANTIATE_TEST_CASE_P(
        TestsDynBatchConcat, MKLDNNGraphDynBatchConcatTests,
        ::testing::Values(
                concat_test_params {
                        {1, 7, 2, 5},
                        {1, 7, 2, 5},
                        2, 1, MKLDNNPlugin::impl_desc_type::ref
                },
                concat_test_params {
                        {1, 7, 2, 5},
                        {1, 13, 2, 5},
                        1, 2, MKLDNNPlugin::impl_desc_type::unknown
                },
                concat_test_params {
                        {3, 7, 2, 5},
                        {3, 13, 2, 5},
                        1, 2, MKLDNNPlugin::impl_desc_type::unknown
                },
                concat_test_params {
                        {1, 7, 2, 13},
                        {1, 7, 2, 17},
                        3, 1, MKLDNNPlugin::impl_desc_type::ref
                },
                concat_test_params {
                        {1, 8, 8, 16},
                        {1, 16, 8, 16},
                        1, 4, MKLDNNPlugin::impl_desc_type::unknown
                },
                concat_test_params {
                        {2, 2, 3, 3},
                        {2, 3, 3, 3},
                        1, 2, MKLDNNPlugin::impl_desc_type::unknown
                },
                concat_test_params {
                        {2, 2, 3, 3, 3},
                        {2, 3, 3, 3, 3},
                        1, 2, MKLDNNPlugin::impl_desc_type::unknown
                }));

struct concat_param {
    std::string name;
    size_t axis;
    size_t input1;
    size_t input2;
};

struct two_concat_test_params {
    // Formats: NCHW, NCDHW
    vector<size_t> in1;
    vector<size_t> in2;
    vector<size_t> in3;

    concat_param concat1;
    concat_param concat2;
};

class MKLDNNGraphTwoConcatTests: public TestsCommon,
                                 public WithParamInterface<two_concat_test_params>  {
    std::string model_t = R"V0G0N(
<net name="TwoConcatsDiffFwd" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">__SRC_DIMS_1__
                </port>
            </output>
        </layer>
        <layer name="in2" type="Input" precision="FP32" id="2">
            <output>
                <port id="1">__SRC_DIMS_2__
                </port>
            </output>
        </layer>
        <layer name="in3" type="Input" precision="FP32" id="3">
            <output>
                <port id="1">__SRC_DIMS_3__
                </port>
            </output>
        </layer>
        <layer name="_CONCAT1_NAME_" id="4" type="Concat" precision="FP32">
            <concat_data axis="_CONCAT1_AXIS_"/>
            <input>
                <port id="1">
                    <dim>_CI41N_</dim>
                    <dim>_CI41C_</dim>
                    <dim>_CI41D_</dim>
                    <dim>_CI41H_</dim>
                    <dim>_CI41W_</dim>
                </port>
                <port id="2">
                    <dim>_CI42N_</dim>
                    <dim>_CI42C_</dim>
                    <dim>_CI42D_</dim>
                    <dim>_CI42H_</dim>
                    <dim>_CI42W_</dim>
                </port>
            </input>
            <output>
                <port id="3">__CO_DIMS_1__
                </port>
            </output>
        </layer>
        <layer name="_CONCAT2_NAME_" id="5" type="Concat" precision="FP32">
            <concat_data axis="_CONCAT2_AXIS_"/>
            <input>
                <port id="1">
                    <dim>_CI51N_</dim>
                    <dim>_CI51C_</dim>
                    <dim>_CI51D_</dim>
                    <dim>_CI51H_</dim>
                    <dim>_CI51W_</dim>
                </port>
                <port id="2">
                    <dim>_CI52N_</dim>
                    <dim>_CI52C_</dim>
                    <dim>_CI52D_</dim>
                    <dim>_CI52H_</dim>
                    <dim>_CI52W_</dim>
                </port>
            </input>
            <output>
                <port id="3">__CO_DIMS_2__
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="_FL11_" to-port="_FP11_"/>
        <edge from-layer="2" from-port="1" to-layer="_FL21_" to-port="_FP21_"/>
        <edge from-layer="3" from-port="1" to-layer="_FL31_" to-port="_FP31_"/>
        <edge from-layer="_FSL_" from-port="_FSP_" to-layer="_FSLTL_" to-port="_FSLTP_"/>
    </edges>
</net>
)V0G0N";
    void changeEdgeToLayer(std::string& model, int f_l, int f_p, int t_l, int t_p, vector<size_t> dims) {
        std::string TL = "_FL" + std::to_string(f_l) + std::to_string(f_p) + "_";
        std::string TP = "_FP" + std::to_string(f_l) + std::to_string(f_p) + "_";
        if (!FIND_STR(model, TL) || !FIND_STR(model, TP)) {
            if (!FIND_STR(model, "_FSL_") || !FIND_STR(model, "_FSP_") ||
                    !FIND_STR(model, "_FSLTL_") || !FIND_STR(model, "_FSLTP_")) {
                THROW_IE_EXCEPTION << "Incorrect configuration!";
            }
            REPLACE_WITH_NUM(model, "_FSL_", f_l);
            REPLACE_WITH_NUM(model, "_FSP_", f_p);
            REPLACE_WITH_NUM(model, "_FSLTL_", t_l);
            REPLACE_WITH_NUM(model, "_FSLTP_", t_p);
        } else {
            REPLACE_WITH_NUM(model, TL, t_l);
            REPLACE_WITH_NUM(model, TP, t_p);
        }

        std::string CI = "_CI" + std::to_string(t_l) + std::to_string(t_p);
        auto dims_size = dims.size();
        REPLACE_WITH_NUM(model, CI + "N_", dims[0]);
        REPLACE_WITH_NUM(model, CI + "C_", dims[1]);
        REPLACE_WITH_NUM(model, CI + "H_", dims[dims_size - 2]);
        REPLACE_WITH_NUM(model, CI + "W_", dims[dims_size - 1]);
        if (dims_size < 5) REMOVE_LINE(model, std::string("<dim>") + CI + std::string("D_") + "</dim>");
        else REPLACE_WITH_NUM(model, CI + "D_", dims[dims_size - 3]);
    }


    std::string getModel(two_concat_test_params p) {
        std::string model = model_t;
        std::string s_dims;
        for (size_t i = 0; i < p.in1.size(); i++) {
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(p.in1[i]) + "</dim>";
        }
	REPLACE_WITH_STR(model, "__SRC_DIMS_1__", s_dims);

        s_dims = "";
        for (size_t i = 0; i < p.in2.size(); i++) {
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(p.in2[i]) + "</dim>";
        }
	REPLACE_WITH_STR(model, "__SRC_DIMS_2__", s_dims);

        s_dims = "";
        for (size_t i = 0; i < p.in3.size(); i++) {
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(p.in3[i]) + "</dim>";
        }
	REPLACE_WITH_STR(model, "__SRC_DIMS_3__", s_dims);

        vector<size_t> concat11;
        switch (p.concat1.input1) {
            case 1:
                changeEdgeToLayer(model, 2, 1, 4, 1, p.in2);
                concat11 = p.in2;
                break;
            case 2:
                changeEdgeToLayer(model, 3, 1, 4, 1, p.in3);
                concat11 = p.in3;
                break;
            default:
                changeEdgeToLayer(model, 1, 1, 4, 1, p.in1);
                concat11 = p.in1;
        }

        vector<size_t> concat12;
        switch (p.concat1.input2) {
            case 1:
                changeEdgeToLayer(model, 2, 1, 4, 2, p.in2);
                concat12 = p.in2;
                break;
            case 2:
                changeEdgeToLayer(model, 3, 1, 4, 2, p.in3);
                concat12 = p.in3;
                break;
            default:
                changeEdgeToLayer(model, 1, 1, 4, 2, p.in1);
                concat12 = p.in1;
        }

        vector<size_t> concat21;
        switch (p.concat2.input1) {
            case 1:
                changeEdgeToLayer(model, 2, 1, 5, 1, p.in2);
                concat21 = p.in2;
                break;
            case 2:
                changeEdgeToLayer(model, 3, 1, 5, 1, p.in3);
                concat21 = p.in3;
                break;
            default:
                changeEdgeToLayer(model, 1, 1, 5, 1, p.in1);
                concat21 = p.in1;
        }

        vector<size_t> concat22;
        switch (p.concat2.input2) {
            case 1:
                changeEdgeToLayer(model, 2, 1, 5, 2, p.in2);
                concat22 = p.in2;
                break;
            case 2:
                changeEdgeToLayer(model, 3, 1, 5, 2, p.in3);
                concat22 = p.in3;
                break;
            default:
                changeEdgeToLayer(model, 1, 1, 5, 2, p.in1);
                concat22 = p.in1;
        }

        s_dims = "";
        for (size_t i = 0; i < p.in2.size(); i++) {
            size_t concat = p.concat1.axis == i ? concat11[i] + concat12[i] : concat21[i];
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(concat) + "</dim>";
        }
	REPLACE_WITH_STR(model, "__CO_DIMS_1__", s_dims);

        REPLACE_WITH_NUM(model, "_CONCAT1_AXIS_", p.concat1.axis);
        REPLACE_WITH_STR(model, "_CONCAT1_NAME_", p.concat1.name);

        s_dims = "";
        for (size_t i = 0; i < p.in2.size(); i++) {
            size_t concat = p.concat2.axis == i ? concat21[i] + concat22[i] : concat21[i];
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(concat) + "</dim>";
        }
	REPLACE_WITH_STR(model, "__CO_DIMS_2__", s_dims);

        REPLACE_WITH_NUM(model, "_CONCAT2_AXIS_", p.concat2.axis);
        REPLACE_WITH_STR(model, "_CONCAT2_NAME_", p.concat2.name);
        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            two_concat_test_params p = ::testing::WithParamInterface<two_concat_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());

            InferenceEngine::SizeVector dims_src1 = p.in1;
            InferenceEngine::SizeVector dims_src2 = p.in2;
            InferenceEngine::SizeVector dims_src3 = p.in3;
            InferenceEngine::Layout layout = InferenceEngine::ANY;
            switch (p.in1.size()) {
                case 4:
                    layout = InferenceEngine::NCHW;
                    break;
                case 5:
                    layout = InferenceEngine::NCDHW;
                    break;
            }

            InferenceEngine::Blob::Ptr src1 = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src1, layout});
            src1->allocate();
            fill_data(src1->buffer(), src1->size());

            InferenceEngine::Blob::Ptr src2 = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src2, layout});
            src2->allocate();
            fill_data(src2->buffer(), src2->size());

            InferenceEngine::Blob::Ptr src3 = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src3, layout});
            src3->allocate();
            fill_data(src3->buffer(), src3->size());

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src1));
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in2", src2));
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in3", src3));

            InferenceEngine::OutputsDataMap out;
            out = net_reader.getNetwork().getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            for (auto & it : out) {
                std::pair<std::string, InferenceEngine::DataPtr> item = it;
                InferenceEngine::TBlob<float>::Ptr output;
                output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
                output->allocate();
                outputBlobs[item.first] = output;
            }

            graph.Infer(srcs, outputBlobs);

            for (const auto& concat : {p.concat1, p.concat2}) {
                float *src1_ptr;
                size_t src1_size;
                float *src2_ptr;
                size_t src2_size;
                InferenceEngine::Blob::Ptr src1_c;
                InferenceEngine::Blob::Ptr src2_c;

                switch (concat.input1) {
                    case 1:
                        src1_ptr = src2->buffer();
                        src1_size = src2->size();
                        src1_c = src2;
                        break;
                    case 2:
                        src1_ptr = src3->buffer();
                        src1_size = src3->size();
                        src1_c = src3;
                        break;
                    default:
                        src1_ptr = src1->buffer();
                        src1_size = src1->size();
                        src1_c = src1;
                }

                switch (concat.input2) {
                    case 1:
                        src2_ptr = src2->buffer();
                        src2_size = src2->size();
                        src2_c = src2;
                        break;
                    case 2:
                        src2_ptr = src3->buffer();
                        src2_size = src3->size();
                        src2_c = src3;
                        break;
                    default:
                        src2_ptr = src1->buffer();
                        src2_size = src1->size();
                        src2_c = src1;
                }

                float *dst_ptr = outputBlobs[concat.name]->buffer();
                size_t dst_size = outputBlobs[concat.name]->size();

                int len1 = 1, len2 = 1, cycles;
                for (int dim = concat.axis; dim < outputBlobs[concat.name]->getTensorDesc().getDims().size(); dim++) {
                    len1 *= src1_c->getTensorDesc().getDims()[dim];
                    len2 *= src2_c->getTensorDesc().getDims()[dim];
                }
                cycles = concat.axis;

                int index1 = 0, index2 = 0, index = 0;
                for (int cycle = 0; cycle < cycles; cycle ++) {
                    for (int i1 = 0; i1 < len1; i1++) {
                        if (src1_ptr[index1] != dst_ptr[index])
                        {
                            FAIL() << concat.name << " index: " << index << " src: "
                                   << src1_ptr[index1] << ", dst: " << dst_ptr[index];
                        }
                        index1++; index++;
                    }
                    for (int i2 = 0; i2 < len2; i2++) {
                        if (src2_ptr[index2] != dst_ptr[index])
                        {
                            FAIL() << concat.name << " index: " << index << " src: "
                                   << src2_ptr[index2] << ", dst: " << dst_ptr[index];
                        }
                        index2++; index++;
                    }
                }
            }
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphTwoConcatTests, TestsTwoConcat) {}

INSTANTIATE_TEST_CASE_P(
        TestsTwoConcat, MKLDNNGraphTwoConcatTests,
        ::testing::Values(
                two_concat_test_params {
                        {1, 5, 2, 5},
                        {3, 5, 2, 5},
                        {1, 5, 2, 5},
                        {"concat1", 0, 0, 1},
                        {"concat2", 0, 1, 2}
                },
                two_concat_test_params {
                        {1, 2, 2, 5},
                        {1, 5, 2, 5},
                        {3, 5, 2, 5},
                        {"concat1", 1, 0, 1},
                        {"concat2", 0, 1, 2}
                },
                two_concat_test_params {
                        {1, 2, 2, 2},
                        {1, 1, 2, 2},
                        {1, 3, 2, 2},
                        {"concat1", 1, 0, 1},
                        {"concat2", 1, 1, 2}
                },
                two_concat_test_params {
                        {1, 5, 2, 5},
                        {3, 5, 2, 5},
                        {1, 5, 2, 5},
                        {"concat1", 0, 0, 1},
                        {"concat2", 0, 2, 1}
                },
                two_concat_test_params {
                        {1, 2, 2, 5},
                        {1, 5, 2, 5},
                        {3, 5, 2, 5},
                        {"concat1", 1, 0, 1},
                        {"concat2", 0, 2, 1}
                },
                two_concat_test_params {
                        {1, 2, 2, 2},
                        {1, 1, 2, 2},
                        {1, 3, 2, 2},
                        {"concat1", 1, 0, 1},
                        {"concat2", 1, 2, 1}
                }));


class MKLDNNGraphTwoInputInConcatTests: public TestsCommon {
    std::string model_t = R"V0G0N(
<net name="TwoConcatsDiffFwd" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer name="in2" type="Input" precision="FP32" id="2">
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer name="norm" id="3" type="ReLU" precision="FP32">
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer name="power" id="4" type="Power" precision="FP32">
            <power_data power="-1" scale="-1" shift="0"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer name="o_concat" id="5" type="Concat" precision="FP32">
            <concat_data axis="1"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>5</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="1" from-port="1" to-layer="5" to-port="2"/>
        <edge from-layer="1" from-port="1" to-layer="4" to-port="1"/>
        <edge from-layer="2" from-port="1" to-layer="5" to-port="1"/>
    </edges>
</net>
)V0G0N";

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            std::string model = model_t;

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());

            InferenceEngine::SizeVector dims_src1 = {1, 3, 2, 2};
            InferenceEngine::SizeVector dims_src2 = {1, 2, 2, 2};

            InferenceEngine::Blob::Ptr src1 = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src1, InferenceEngine::NCHW});
            src1->allocate();
            float *src1_data = src1->buffer();
            for (size_t i = 0; i < src1->size(); i++) {
                src1_data[i] = i + 1;
            }

            InferenceEngine::Blob::Ptr src2 = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src2, InferenceEngine::NCHW});
            src2->allocate();
            fill_data(src2->buffer(), src2->size());

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src1));
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in2", src2));

            InferenceEngine::OutputsDataMap out;
            out = net_reader.getNetwork().getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            for (auto & it : out) {
                std::pair<std::string, InferenceEngine::DataPtr> item = it;
                InferenceEngine::TBlob<float>::Ptr output;
                output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
                output->allocate();
                outputBlobs[item.first] = output;
            }

            graph.Infer(srcs, outputBlobs);

            float *src1_ptr = src2->buffer();
            size_t src1_size = src2->size();
            float *src2_ptr = src1->buffer();
            size_t src2_size = src1->size();

            float *dst_ptr = outputBlobs["o_concat"]->buffer();
            size_t dst_size = outputBlobs["o_concat"]->size();

            int len1 = 1, len2 = 1, cycles;
            for (int dim = 1; dim < outputBlobs["o_concat"]->getTensorDesc().getDims().size(); dim++) {
                len1 *= src2->getTensorDesc().getDims()[dim];
                len2 *= src1->getTensorDesc().getDims()[dim];
            }
            cycles = 1;

            int index1 = 0, index2 = 0, index = 0;
            for (int cycle = 0; cycle < cycles; cycle ++) {
                for (int i1 = 0; i1 < len1; i1++) {
                    if (src1_ptr[index1] != dst_ptr[index])
                    {
                        FAIL() << "concat index: " << index << " src: "
                               << src1_ptr[index1] << ", dst: " << dst_ptr[index];
                    }
                    index1++; index++;
                }
                for (int i2 = 0; i2 < len2; i2++) {
                    if (src2_ptr[index2] != dst_ptr[index])
                    {
                        FAIL() << "concat index: " << index << " src: "
                               << src2_ptr[index2] << ", dst: " << dst_ptr[index];
                    }
                    index2++; index++;
                }
            }
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_F(MKLDNNGraphTwoInputInConcatTests, TestSecondInputToConcat) {}

class MKLDNNGraphIncorrectConcatTests: public TestsCommon,
                              public WithParamInterface<concat_test_params> {
    std::string model_t = R"V0G0N(
<net name="ConcatOnly" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">__SRC_DIMS_1__
                </port>
            </output>
        </layer>
        <layer name="in2" type="Input" precision="FP32" id="2">
            <output>
                <port id="2">__SRC_DIMS_2__
                </port>
            </output>
        </layer>
        <layer name="con" id="3" type="Concat" precision="FP32">
            <concat_data axis="_AXIS_"/>
            <input>
                <port id="1">__SRC_DIMS_1__
                </port>
                <port id="2">__SRC_DIMS_2__
                </port>
            </input>
            <output>
                <port id="3">__DST_DIMS__
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="2"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(concat_test_params p) {
        std::string model = model_t;
        std::string s_dims;
        for (auto& dim : p.in1) {
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(dim) + "</dim>";
        }
	REPLACE_WITH_STR(model, "__SRC_DIMS_1__", s_dims);

        s_dims = "";
        for (auto& dim : p.in2) {
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(dim) + "</dim>";
        }
	REPLACE_WITH_STR(model, "__SRC_DIMS_2__", s_dims);

        s_dims = "";
        for (size_t i = 0; i < p.in1.size(); i++) {
            size_t dim = p.axis == i ? p.in1[i] + p.in2[i] : p.in1[i];
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(dim) + "</dim>";
        }
	REPLACE_WITH_STR(model, "__DST_DIMS__", s_dims);

        REPLACE_WITH_NUM(model, "_AXIS_", p.axis);
        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            concat_test_params p = ::testing::WithParamInterface<concat_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_THROW(net_reader.ReadNetwork(model.data(), model.length()), 
                         InferenceEngine::details::InferenceEngineException);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphIncorrectConcatTests, TestsIncorrectConcat) {}


INSTANTIATE_TEST_CASE_P(
        TestsIncorrectConcat, MKLDNNGraphIncorrectConcatTests,
        ::testing::Values(
                concat_test_params {
                        {1, 7, 2, 5},
                        {1, 7, 3, 5},
                        1
                },
                concat_test_params {
                        {1, 7, 2, 5},
                        {1, 7, 4, 4},
                        2
                }));
