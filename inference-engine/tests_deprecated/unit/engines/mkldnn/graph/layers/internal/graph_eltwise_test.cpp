// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "test_graph.hpp"

#include <ie_plugin_config.hpp>
#include "common_test_utils/data_utils.hpp"
#include "single_layer_common.hpp"
#include <mkldnn_extension_utils.h>
#include <legacy/cnn_network_impl.hpp>
#include "tests_common.hpp"
#include <ie_core.hpp>

using namespace ::testing;
using namespace std;
using namespace mkldnn;

struct eltwise_test_params {
    // Formats: NCHW, NCDHW
    vector<size_t> dims1;
    vector<size_t> dims2;
    vector<size_t> dims3;

    enum opType {
        Sum = 0, Prod, Max, Min, Sub, Div, Squared_diff, Floor_mod, Pow,
        Logical_AND, Logical_OR, Logical_XOR,
        Less, Less_equal, Greater, Greater_equal, Equal, Not_equal
    };

    opType op;

    std::string scales;

    size_t num_prim_desc;

    MKLDNNPlugin::impl_desc_type selectedType;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

template<typename data_t>
void ref_eltwise(const std::vector<InferenceEngine::TBlob<data_t>> &src, InferenceEngine::TBlob<data_t> &dst, eltwise_test_params prm) {
    std::vector<float> scales;
    if (prm.scales != "") {
        std::istringstream stream(prm.scales);
        stream.imbue(std::locale("C"));
        std::string str;
        while (getline(stream, str, ',')) {
            float val = InferenceEngine::CNNLayer::ie_parse_float(str);
            scales.push_back(val);
        }
    } else {
        for (int i = 0; i < src.size(); i++) {
            scales.push_back(1.0f);
        }
    }

    data_t *dst_data = dst.data();

    const data_t *src_data = src[0].readOnly();
    auto& dims = dst.getTensorDesc().getDims();
    auto& dims0 = src[0].getTensorDesc().getDims();

    int offset_in[5] = {1, 1, 1, 1, 1};
    int offset_out[5] = {1, 1, 1, 1, 1};

    for (int i = 0; i < dims0.size(); i++)
        offset_in[5 - dims0.size() + i] = dims0[i];
    for (int i = 0; i < dims.size(); i++)
        offset_out[5 - dims.size() + i] = dims[i];

    unsigned long j = 0, k = 0;

    for (int i0 = 0; i0 < offset_out[0]; i0++) {
        if (i0 > offset_in[0] - 1) {
            k -= offset_in[1]*offset_in[2]*offset_in[3]*offset_in[4];
        }
        for (int i1 = 0; i1 < offset_out[1]; i1++) {
            if (i1 > offset_in[1] - 1) {
                k -= offset_in[2]*offset_in[3]*offset_in[4];
            }
            for (int i2 = 0; i2 < offset_out[2]; i2++) {
                if (i2 > offset_in[2] - 1) {
                    k -= offset_in[3]*offset_in[4];
                }
                for (int i3 = 0; i3 < offset_out[3]; i3++) {
                    if (i3 > offset_in[3] - 1) {
                        k -= offset_in[4];
                    }
                    for (int i4 = 0; i4 < offset_out[4]; i4++) {
                        if (i4 > offset_in[4] - 1) {
                            k -= 1;
                        }
                        if (prm.op == eltwise_test_params::Sum) {
                            dst_data[j++] = scales[0] * src_data[k++];
                        } else {
                            dst_data[j++] = src_data[k++];
                        }
                    }
                }
            }
        }
    }

    for (int n = 1; n < src.size(); n++) {
        j = 0;
        k = 0;
        src_data = src[n].readOnly();
        auto& dims1 = src[n].getTensorDesc().getDims();
        int offset_in1[5] = {1, 1, 1, 1, 1};
        for (int i = 0; i < dims1.size(); i++)
            offset_in1[5 - dims1.size() + i] = dims1[i];

        for (int i0 = 0; i0 < offset_out[0]; i0++) {
            if (i0 > offset_in1[0] - 1) {
                k -= offset_in1[1]*offset_in1[2]*offset_in1[3]*offset_in1[4];
            }
            for (int i1 = 0; i1 < offset_out[1]; i1++) {
                if (i1 > offset_in1[1] - 1) {
                    k -= offset_in1[2]*offset_in1[3]*offset_in1[4];
                }
                for (int i2 = 0; i2 < offset_out[2]; i2++) {
                    if (i2 > offset_in1[2] - 1) {
                        k -= offset_in1[3]*offset_in1[4];
                    }
                    for (int i3 = 0; i3 < offset_out[3]; i3++) {
                        if (i3 > offset_in1[3] - 1) {
                            k -= offset_in1[4];
                        }
                        for (int i4 = 0; i4 < offset_out[4]; i4++, j++, k++) {
                            if (i4 > offset_in1[4] - 1) {
                                k -= 1;
                            }
                            switch (prm.op) {
                                case eltwise_test_params::Sum:
                                    dst_data[j] += scales[n] * src_data[k];
                                    break;
                                case eltwise_test_params::Sub:
                                    dst_data[j] = dst_data[j] - src_data[k];
                                    break;
                                case eltwise_test_params::Min:
                                    dst_data[j] = (std::min)(dst_data[j], src_data[k]);
                                    break;
                                case eltwise_test_params::Max:
                                    dst_data[j] = (std::max)(dst_data[j], src_data[k]);
                                    break;
                                case eltwise_test_params::Prod:
                                    dst_data[j] = dst_data[j] * src_data[k];
                                    break;
                                case eltwise_test_params::Div:
                                    dst_data[j] = dst_data[j] / src_data[k];
                                    break;
                                case eltwise_test_params::Squared_diff:
                                    dst_data[j] = (dst_data[j] - src_data[k]) * (dst_data[j] - src_data[k]);
                                    break;
                                case eltwise_test_params::Logical_OR:
                                    dst_data[j] = dst_data[j] || src_data[k];
                                    break;
                                case eltwise_test_params::Logical_AND:
                                    dst_data[j] = dst_data[j] && src_data[k];
                                    break;
                                case eltwise_test_params::Logical_XOR:
                                    dst_data[j] = (dst_data[j] || src_data[k]) - (dst_data[j] && src_data[k]);
                                    break;
                                case eltwise_test_params::Less:
                                    dst_data[j] = dst_data[j] < src_data[k];
                                    break;
                                case eltwise_test_params::Less_equal:
                                    dst_data[j] = dst_data[j] <= src_data[k];
                                    break;
                                case eltwise_test_params::Greater:
                                    dst_data[j] = dst_data[j] > src_data[k];
                                    break;
                                case eltwise_test_params::Greater_equal:
                                    dst_data[j] = dst_data[j] >= src_data[k];
                                    break;
                                case eltwise_test_params::Equal:
                                    dst_data[j] = dst_data[j] == src_data[k];
                                    break;
                                case eltwise_test_params::Not_equal:
                                    dst_data[j] = dst_data[j] != src_data[k];
                                    break;
                                case eltwise_test_params::Pow:
                                    dst_data[j] = std::pow(dst_data[j], src_data[k]);
                                    break;
                                case eltwise_test_params::Floor_mod:
                                    dst_data[j] = dst_data[j] - dst_data[j] / src_data[k] * src_data[k];
                                    break;
                            }
                        }
                    }
                }
            }
        }
    }
}

std::string select_op(eltwise_test_params::opType op) {
    std::string str_op;
    switch(op){
        case eltwise_test_params::opType::Sum:
            str_op = "sum";
            break;
        case eltwise_test_params::opType::Prod:
            str_op = "prod";
            break;
        case eltwise_test_params::opType::Max:
            str_op = "max";
            break;
        case eltwise_test_params::opType::Min:
            str_op = "min";
            break;
        case eltwise_test_params::opType::Sub:
            str_op = "sub";
            break;
        case eltwise_test_params::opType::Div:
            str_op = "div";
            break;
        case eltwise_test_params::opType::Squared_diff:
            str_op = "squared_diff";
            break;
        case eltwise_test_params::opType::Logical_AND:
            str_op = "logical_and";
            break;
        case eltwise_test_params::opType::Logical_OR:
            str_op = "logical_or";
            break;
        case eltwise_test_params::opType::Logical_XOR:
            str_op = "logical_xor";
            break;
        case eltwise_test_params::opType ::Less:
            str_op = "less";
            break;
        case eltwise_test_params::opType::Less_equal:
            str_op = "less_equal";
            break;
        case eltwise_test_params::opType::Greater:
            str_op = "greater";
            break;
        case eltwise_test_params::opType::Greater_equal:
            str_op = "greater_equal";
            break;
        case eltwise_test_params::opType::Equal:
            str_op = "equal";
            break;
        case eltwise_test_params::opType::Not_equal:
            str_op = "not_equal";
            break;
        case eltwise_test_params::opType::Pow:
            str_op = "pow";
            break;
        case eltwise_test_params::opType::Floor_mod:
            str_op = "floor_mod";
            break;
    }
    return str_op;
}

struct precisions_test_2params {
    struct {
        std::string precision0;
        std::string precision1;
    } in;

    size_t num_nodes;
    size_t num_reorder_nodes;
};

class MKLDNNGraphEltwise2PrecisionsTests : public TestsCommon,
                                     public WithParamInterface<precisions_test_2params> {

    std::string model_t = R"V0G0N(
<net name="default" version="2" batch="1">
    <layers>
        <layer name="second_input" type="Input" precision="_IP1_" id="1">
            <output>
                <port id="1" precision="_IP1_">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer name="data" type="Input" precision="_IP0_" id="0">
            <output>
                <port id="0" precision="_IP0_">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Eltwise" precision="FP32" id="2">
            <elementwise_data operation="sum" coeff=""/>
            <input>
                <port id="2">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="4" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="2"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="3"/>
    </edges>
</net>
)V0G0N";

protected:
    std::string getModel(precisions_test_2params p) {
        std::string model = model_t;

        REPLACE_WITH_STR(model, "_IP0_", p.in.precision0);
        REPLACE_WITH_STR(model, "_IP1_", p.in.precision1);
        return model;
    }

    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            precisions_test_2params p = ::testing::WithParamInterface<precisions_test_2params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

            MKLDNNGraphTestClass graph;
            ASSERT_NO_THROW(graph.CreateGraph(network));

            auto& nodes = graph.getNodes();
            nodes = graph.getNodes();
            ASSERT_EQ(nodes.size(), p.num_nodes);

            size_t actual_reorder_nodes = 0;
            for (size_t i = 0; i < nodes.size(); i++) {
                if(nodes[i].get()->getType() == MKLDNNPlugin::Type::Reorder &&
                    FIND_STR(nodes[i].get()->getName(), "_U8_FP32_"))
                    actual_reorder_nodes ++;
            }
            ASSERT_EQ(actual_reorder_nodes, p.num_reorder_nodes);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphEltwise2PrecisionsTests, TestsEltwise2Precisions) {}

INSTANTIATE_TEST_CASE_P(
        TestsEltwise2Precisions, MKLDNNGraphEltwise2PrecisionsTests,
        ::testing::Values(
            precisions_test_2params{ {"FP32", "FP32"}, 4, 0 },
            precisions_test_2params{ {  "U8", "FP32"}, 4, 0 },
            precisions_test_2params{ {"FP32",   "U8"}, 4, 0 },
            precisions_test_2params{ {  "U8",   "U8"}, 4, 0 }
        ));
