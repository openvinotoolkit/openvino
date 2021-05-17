// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>
#include <gtest/gtest.h>
#include <cpp/ie_infer_request.hpp>
#include <blob_factory.hpp>
#include <ie_algorithm.hpp>
#include <precision_utils.h>

#include "plg_test.hpp"
#include "single_layer_common.hpp"

using namespace ::testing;
using namespace InferenceEngine;

struct ti_test_params {
    std::string device_name;
    std::size_t tensorSize;
    InferenceEngine::Precision precision;
};

static void setValuesInBlob(Blob::Ptr blob, float value) {
    auto dims = blob->getTensorDesc().getDims();
    auto output_size = details::product(std::begin(dims), std::end(dims));
    std::vector<float> values_vector(output_size, value);

    if (!blob->is<MemoryBlob>())
        IE_THROW() << "Only MemoryBlob is expected here";

    auto m_blob = blob->as<MemoryBlob>();
    if (m_blob->wmap().as<void*>() == nullptr)
        blob->allocate();

    CopyVectorToBlob(blob, values_vector);
}

/*
         ______________main_ti__________________
  in1 --|~~ iter  -> add -> plus_one -> next1   |
  in2 --|~~ prev1 -> add -> out_iter ~~~~~~~~~~~|-- out1
         ---------------------------------------
*/

class TITestBase: public PlgTest<ti_test_params> {
    std::string model_t = R"V0G0N(
<net batch="1" name="frozen" version="5">
	<layers>
		<layer id="0" name="in1" precision="_PRC_" type="Input">
			<output>
				<port id="0">
                    <dim>_IN_</dim>
                    <dim>_INPUT_SIZE_</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="in2" precision="_PRC_" type="Input">
			<output>
				<port id="0">
                    <dim>_IN_</dim>
                    <dim>_CHUNK_SIZE_</dim>
				</port>
			</output>
		</layer>
        <layer id="2" name="main_ti" type="TensorIterator" precision="_PRC_">
            <input>
                <port id="0">
                    <dim>_IN_</dim>
                    <dim>_INPUT_SIZE_</dim>
                </port>
                <port id="1">
                    <dim>_IN_</dim>
                    <dim>_CHUNK_SIZE_</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>_IN_</dim>
                    <dim>_INPUT_SIZE_</dim>
                </port>
            </output>
            <port_map>
				<input external_port_id="0" internal_layer_id="0" internal_port_id="0" axis="1" stride="_CHUNK_SIZE_"/>
				<input external_port_id="1" internal_layer_id="0" internal_port_id="1"/>
				<output external_port_id="2" internal_layer_id="0" internal_port_id="2" axis="1" stride="_CHUNK_SIZE_"/>
			</port_map>
			<back_edges>
				<edge from-layer="1" from-port="1" to-layer="0" to-port="1"/>
			</back_edges>
            <body>
                <layers>
                    <layer id="0" name="add" precision="_PRC_" type="Eltwise">
                        <data operation="sum"/>
                        <input>
                            <port id="0">
                                <dim>_IN_</dim>
                                <dim>_CHUNK_SIZE_</dim>
                            </port>
                            <port id="1">
                                <dim>_IN_</dim>
                                <dim>_CHUNK_SIZE_</dim>
                            </port>
                        </input>
                        <output>
                            <port id="2">
                                <dim>_IN_</dim>
                                <dim>_CHUNK_SIZE_</dim>
                            </port>
                        </output>
                    </layer>
                    <layer id="1" name="plus_one" precision="_PRC_" type="Power">
                        <data scale="1" shift="1" power="1"/>
                        <input>
                            <port id="0">
                                <dim>_IN_</dim>
                                <dim>_CHUNK_SIZE_</dim>
                            </port>
                        </input>
                        <output>
                            <port id="1">
                                <dim>_IN_</dim>
                                <dim>_CHUNK_SIZE_</dim>
                            </port>
                        </output>
                    </layer>
                </layers>
                <edges>
                    <edge from-layer="0" from-port="2" to-layer="1" to-port="0"/>
                </edges>
            </body>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(const ti_test_params & p) {
        std::string model = model_t;
        std::size_t iteration_count = 3;

        REPLACE_WITH_NUM(model, "_IN_", 1);
        REPLACE_WITH_NUM(model, "_IC_", 3);
        REPLACE_WITH_NUM(model, "_INPUT_SIZE_", iteration_count * p.tensorSize);
        REPLACE_WITH_NUM(model, "_CHUNK_SIZE_", p.tensorSize);
        REPLACE_WITH_STR(model, "_PRC_", p.precision.name());

        return model;
    }

protected:
    void RunTITest(const std::map<std::string, std::string> & config = {}) {

        try {
            ti_test_params p = param();
            std::string model = getModel(p);

            Core ie;
            auto net = ie.ReadNetwork(model, Blob::CPtr());
            auto exec = ie.LoadNetwork(net, device_name, config);
            auto req = exec.CreateInferRequest();
            setValuesInBlob(req.GetBlob("in1"), 1.0f);
            setValuesInBlob(req.GetBlob("in2"), 1.0f);
            req.Infer();

        } catch (const InferenceEngine::Exception &e) {
            FAIL() << e.what();
        }
    }
};

using TITest  = TITestBase;

// disabled due to transition to ngraph transformations
TEST_P(TITest, DISABLED_TestsWitUnusedOut) { RunTITest(); }

/*
  TI body contains const data placeholder

         ______________main_ti__________________
  in1 --|~~ iter  -> add -> plus_one ~~~~~~~~~~~|-- out1
        |  const1 -> add                        |
         ---------------------------------------
*/

class TITest2Base: public PlgTest<ti_test_params> {
    std::string model_t = R"V0G0N(
<net batch="1" name="frozen" version="5">
	<layers>
		<layer id="0" name="in1" precision="_PRC_" type="Input">
			<output>
				<port id="0">
                    <dim>_IN_</dim>
                    <dim>_INPUT_SIZE_</dim>
				</port>
			</output>
		</layer>
        <layer id="1" name="main_ti" type="TensorIterator" precision="_PRC_">
            <input>
                <port id="0">
                    <dim>_IN_</dim>
                    <dim>_INPUT_SIZE_</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>_IN_</dim>
                    <dim>_INPUT_SIZE_</dim>
                </port>
            </output>
            <port_map>
				<input external_port_id="0" internal_layer_id="1" internal_port_id="0" axis="1" stride="_CHUNK_SIZE_"/>
				<output external_port_id="1" internal_layer_id="2" internal_port_id="1" axis="1" stride="_CHUNK_SIZE_"/>
			</port_map>
            <body>
                <layers>
                    <layer id="0" name="const" precision="_PRC_" type="Const">
                        <output>
                            <port id="1">
                                <dim>1</dim>
                                <dim>_CHUNK_SIZE_</dim>
                            </port>
                        </output>
                        <blobs>
                            <custom offset="0" size="_SZ_"/>
                        </blobs>
                    </layer>
                    <layer id="1" name="add" precision="_PRC_" type="Eltwise">
                        <data operation="sum"/>
                        <input>
                            <port id="0">
                                <dim>_IN_</dim>
                                <dim>_CHUNK_SIZE_</dim>
                            </port>
                            <port id="1">
                                <dim>1</dim>
                                <dim>_CHUNK_SIZE_</dim>
                            </port>
                        </input>
                        <output>
                            <port id="2">
                                <dim>_IN_</dim>
                                <dim>_CHUNK_SIZE_</dim>
                            </port>
                        </output>
                    </layer>
                    <layer id="2" name="plus_one" precision="_PRC_" type="Power">
                        <data scale="1" shift="1" power="1"/>
                        <input>
                            <port id="0">
                                <dim>_IN_</dim>
                                <dim>_CHUNK_SIZE_</dim>
                            </port>
                        </input>
                        <output>
                            <port id="1">
                                <dim>_IN_</dim>
                                <dim>_CHUNK_SIZE_</dim>
                            </port>
                        </output>
                    </layer>
                </layers>
                <edges>
                    <edge from-layer="0" from-port="1" to-layer="1" to-port="1"/>
                    <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
                </edges>
            </body>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(const ti_test_params& p) {
        std::string model = model_t;
        std::size_t iteration_count = 3;

        REPLACE_WITH_NUM(model, "_IN_", 1);
        REPLACE_WITH_NUM(model, "_INPUT_SIZE_", iteration_count * p.tensorSize);
        REPLACE_WITH_NUM(model, "_CHUNK_SIZE_", p.tensorSize);
        REPLACE_WITH_STR(model, "_PRC_", p.precision.name());
        REPLACE_WITH_NUM(model, "_SZ_", p.precision.size() * p.tensorSize);

        return model;
    }

protected:
    virtual void RunTITest(const std::map<std::string, std::string> & config = {}) {
        try {
            ti_test_params p = param();
            std::string model = getModel(p);

            auto weights = make_shared_blob<uint8_t>(TensorDesc {Precision::U8, {p.precision.size() * p.tensorSize}, C});
            weights->allocate();

            if (p.precision == Precision::FP32) {
                std::vector<float> weights_vector(p.tensorSize, 1.0f);
                ie_memcpy(weights->buffer().as<float *>(), p.tensorSize * sizeof(float),
                    &weights_vector[0], p.tensorSize * sizeof(float));
            } else if (p.precision == Precision::FP16) {
                //  FP16 case
                std::vector<ie_fp16> weights_vector(p.tensorSize, PrecisionUtils::f32tof16(1.0f));
                ie_memcpy(weights->buffer().as<ie_fp16 *>(), p.tensorSize * sizeof(ie_fp16),
                    &weights_vector[0], p.tensorSize * sizeof(ie_fp16));
            } else {
                ASSERT_TRUE(false);
            }

            Core ie;
            auto net = ie.ReadNetwork(model, weights);
            auto exec = ie.LoadNetwork(net, device_name, config);
            auto req = exec.CreateInferRequest();
            setValuesInBlob(req.GetBlob("in1"), 1.0f);
            req.Infer();

        } catch (const InferenceEngine::Exception &e) {
            FAIL() << e.what();
        }
    }
};

using TITest2  = TITest2Base;

// disabled due to transition to ngraph transformations
TEST_P(TITest2, DISABLED_TestsWitCopy) { RunTITest(); }
