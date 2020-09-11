// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_core.hpp>

#include "tests_common.hpp"
#include "single_layer_common.hpp"
#include "common_test_utils/common_layers_params.hpp"
#include "common_test_utils/data_utils.hpp"

using namespace ::testing;
using namespace InferenceEngine;

struct eltwise_test_params {
    std::string device_name;

    struct {
        size_t w;
        size_t h;
        size_t c;
    } in;

    enum opType {
        Sum = 0, Prod = 1, Max = 2, Sub = 3, Min = 4, Div = 5, Squared_diff = 6, Equal = 7, Not_equal = 8,
        Less = 9, Less_equal = 10, Greater = 11, Greater_equal = 12, Logical_AND = 13, Logical_OR = 14, Logical_XOR = 15,
        Floor_mod = 16, Pow = 17
    };

    opType op;

    size_t inputsNum;
};

template<typename data_t>
void ref_eltwise(const std::vector<Blob::Ptr> &srcs, std::vector<Blob::Ptr> &dsts, eltwise_test_params prm) {
    assert(dsts.size() == 1);
    data_t *dst_data = dsts[0]->buffer().as<data_t*>();

    const data_t *src_data = srcs[0]->buffer().as<data_t*>();

    for (int i = 0; i < srcs[0]->size(); i++) {
        dst_data[i] = src_data[i];
    }

    for (int n = 1; n < srcs.size(); n++) {
        src_data = srcs[n]->buffer().as<data_t*>();

        for (int i = 0; i < srcs[n]->size(); i++) {
            switch (prm.op) {
                case eltwise_test_params::Sum:
                    dst_data[i] += src_data[i];
                    break;

                case eltwise_test_params::Prod:
                    dst_data[i] *= src_data[i];
                    break;

                case eltwise_test_params::Max:
                    dst_data[i] = std::max<data_t>(dst_data[i], src_data[i]);
                    break;

                case eltwise_test_params::Sub:
                    dst_data[i] -= src_data[i];
                    break;

                case eltwise_test_params::Min:
                    dst_data[i] = std::min<data_t>(dst_data[i], src_data[i]);
                    break;

                case eltwise_test_params::Div:
                    dst_data[i] /= src_data[i];
                    break;

                case eltwise_test_params::Squared_diff: {
                    data_t tmp = (dst_data[i] - src_data[i]);
                    dst_data[i] = tmp * tmp;
                    break;
                }

                case eltwise_test_params::Equal:
                    dst_data[i] = dst_data[i] == src_data[i];
                    break;

                case eltwise_test_params::Not_equal:
                    dst_data[i] = dst_data[i] != src_data[i];
                    break;

                case eltwise_test_params::Less:
                    dst_data[i] = dst_data[i] < src_data[i];
                    break;

                case eltwise_test_params::Less_equal:
                    dst_data[i] = dst_data[i] <= src_data[i];
                    break;

                case eltwise_test_params::Greater:
                    dst_data[i] = dst_data[i] > src_data[i];
                    break;

                case eltwise_test_params::Greater_equal:
                    dst_data[i] = dst_data[i] >= src_data[i];
                    break;

                case eltwise_test_params::Logical_AND:
                    dst_data[i] = dst_data[i] && src_data[i];
                    break;

                case eltwise_test_params::Logical_OR:
                    dst_data[i] = dst_data[i] || src_data[i];
                    break;

                case eltwise_test_params::Logical_XOR:
                    dst_data[i] = !dst_data[i] != !src_data[i];
                    break;

                case eltwise_test_params::Floor_mod: {
                    data_t src1 = src_data[i];
                    data_t src2 = dst_data[i];
                    dst_data[i] = src1 - src1 / src2 * src2;
                    break;
                }

                case eltwise_test_params::Pow: {
                    dst_data[i] = std::pow(src_data[i], dst_data[i]);
                    break;
                }
            }
        }
    }
}

class EltwiseOnlyTest : public TestsCommon,
                              public WithParamInterface<eltwise_test_params> {

    std::string model_t = R"V0G0N(
<Net Name="Eltwise_Only" version="2" precision="FP32" batch="1">
    <layers>
        _INPUT_LAYERS_
        <layer name="eltwise" id="0" type="Eltwise" precision="FP32">
            <elementwise_data operation="_OP_"/>
            <input>
                _ELTWISE_INPUT_PORTS_
            </input>
            <output>
                <port id="0">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        _EDGES_
    </edges>
</Net>
)V0G0N";

    std::string getModel(eltwise_test_params p) {
        std::string model = model_t;
        std::string op = p.op == eltwise_test_params::Sum ? "sum" :
                         p.op == eltwise_test_params::Prod ? "mul" :
                         p.op == eltwise_test_params::Max ? "max" :
                         p.op == eltwise_test_params::Sub ? "sub" :
                         p.op == eltwise_test_params::Min ? "min" :
                         p.op == eltwise_test_params::Div ? "div" :
                         p.op == eltwise_test_params::Squared_diff ? "squared_diff" :
                         p.op == eltwise_test_params::Equal ? "equal" :
                         p.op == eltwise_test_params::Not_equal ? "not_equal" :
                         p.op == eltwise_test_params::Less ? "less" :
                         p.op == eltwise_test_params::Less_equal ? "less_equal" :
                         p.op == eltwise_test_params::Greater ? "greater" :
                         p.op == eltwise_test_params::Greater_equal ? "greater_equal" :
                         p.op == eltwise_test_params::Logical_AND ? "logical_and" :
                         p.op == eltwise_test_params::Logical_OR ? "logical_or" :
                         p.op == eltwise_test_params::Logical_XOR ? "logical_xor" :
                         p.op == eltwise_test_params::Floor_mod ? "floor_mod" :
                         p.op == eltwise_test_params::Pow ? "pow" :
                         "sum" /* default */;

        // Generating inputs layers
        auto generateInput = [](size_t inputId) -> std::string {
            std::string inputLayer = R"V0G0N(
        <layer name="data_ID_" type="Input" precision="FP32" id="_ID_">
            <output>
                <port id="_ID_">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>)V0G0N";
            REPLACE_WITH_NUM(inputLayer, "_ID_", inputId);
            return inputLayer;
        };
        std::string tmp;

        for (size_t i = 1; i < p.inputsNum + 1; ++i) {
            tmp += generateInput(i);
        }

        REPLACE_WITH_STR(model, "_INPUT_LAYERS_", tmp);

        // Generating Eltwise inputs
        tmp.clear();
        auto generateEltwiseInputPort = [](size_t inputId) -> std::string {
            std::string inputPort = R"V0G0N(
                <port id="_ID_">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>)V0G0N";
            REPLACE_WITH_NUM(inputPort, "_ID_", inputId);
            return inputPort;
        };

        for (size_t i = p.inputsNum + 1; i < (2 * p.inputsNum) + 1; ++i) {
            tmp += generateEltwiseInputPort(i);
        }

        REPLACE_WITH_STR(model, "_ELTWISE_INPUT_PORTS_", tmp);

        // Generating Edges
        tmp.clear();
        auto generateEdge = [](size_t inputLayerId, size_t eltwiseInputPortId) -> std::string {
            std::string edge = R"V0G0N(
                    <edge from-layer="_INPUT_LAYER_ID_" from-port="_INPUT_LAYER_ID_" to-layer="0" to-port="_ELTWISE_INPUT_PORT_ID_"/>)V0G0N";
            REPLACE_WITH_NUM(edge, "_INPUT_LAYER_ID_", inputLayerId);
            REPLACE_WITH_NUM(edge, "_ELTWISE_INPUT_PORT_ID_", eltwiseInputPortId);
            return edge;
        };

        for (size_t i = 1; i < p.inputsNum + 1; ++i) {
            tmp += generateEdge(i, p.inputsNum + i);
        }

        REPLACE_WITH_STR(model, "_EDGES_", tmp);

        REPLACE_WITH_NUM(model, "_IN_", 1);
        REPLACE_WITH_NUM(model, "_IC_", p.in.c);
        REPLACE_WITH_NUM(model, "_IH_", p.in.h);
        REPLACE_WITH_NUM(model, "_IW_", p.in.w);
        REPLACE_WITH_STR(model, "_OP_", op);
        return model;
    }

 protected:
    virtual void SetUp() {

        try {
            eltwise_test_params p = ::testing::WithParamInterface<eltwise_test_params>::GetParam();
            std::string model = getModel(p);

            Core ie;
            CNNNetwork net = ie.ReadNetwork(model, Blob::CPtr());
            ExecutableNetwork executable_network = ie.LoadNetwork(net, p.device_name);
            InferRequest inferRequest = executable_network.CreateInferRequest();

            std::vector<Blob::Ptr> srcs_vec;

            InputsDataMap in_info_map = net.getInputsInfo();
            for (auto info : in_info_map) {
                Blob::Ptr inputBlob = inferRequest.GetBlob(info.first);
                float* inputData = inputBlob->buffer().as<float*>();
                
                if (p.op != eltwise_test_params::Pow)
                    CommonTestUtils::fill_data_sine(inputBlob->buffer().as<float*>(), inputBlob->size(), 100, 10, 10);
                else
                    CommonTestUtils::fill_data_const(inputBlob, 2);

                srcs_vec.push_back(inputBlob);
            }

            BlobMap dsts_map;
            std::vector<Blob::Ptr> dsts_vec;

            OutputsDataMap out_info_map = net.getOutputsInfo();
            for (auto info : out_info_map) {
                Blob::Ptr outputBlob = inferRequest.GetBlob(info.first);
                dsts_map[info.first] = outputBlob;
                
                Blob::Ptr blob_ref = make_shared_blob<float>({Precision::FP32, info.second->getTensorDesc().getDims(), Layout::NCHW});
                blob_ref->allocate();
                dsts_vec.push_back(blob_ref);
            }

            ref_eltwise<float>(srcs_vec, dsts_vec, p);

            inferRequest.Infer();

            compare(*dsts_map.begin()->second, *dsts_vec[0]);

        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(EltwiseOnlyTest, TestsEltwise) {}

/*** TBD ***/


