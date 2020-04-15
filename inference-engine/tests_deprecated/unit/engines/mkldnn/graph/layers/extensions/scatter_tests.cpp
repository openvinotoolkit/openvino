// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "mkldnn_graph.h"

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include <mkldnn_extension_utils.h>
#include "tests_common.hpp"
#include <cpp/ie_cnn_net_reader.h>


using namespace ::testing;
using namespace std;
using namespace mkldnn;

struct scatterTF_test_params {
    std::string inIdxPrecision;
    InferenceEngine::SizeVector inDataDim;
    std::vector<float> inData;
    InferenceEngine::SizeVector inIdxDim;
    std::vector<int32_t> inIdx;
    std::vector<float> inUpd;
    int axis;

    std::vector<float> reference;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

class MKLDNNCPUExtScatterTFTests : public TestsCommon, public WithParamInterface<scatterTF_test_params> {
    std::string model_t = R"V0G0N(
<net Name="Scatter_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="InputData" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    _IDATA_
                </port>
            </output>
        </layer>
        <layer name="InputIndexes" type="Input" precision="_IIDXP_" id="2">
            <output>
                <port id="2">
                    _IIDX_
                </port>
            </output>
        </layer>
        <layer name="InputUpdates" type="Input" precision="FP32" id="3">
            <output>
                <port id="3">
                    _IIDX_
                </port>
            </output>
        </layer>
        <layer name="scatter" type="ScatterUpdate" precision="FP32" id="4">
            <data axis="_AX_"/>
            <input>
                <port id="1">
                    _IDATA_
                </port>
                <port id="2" precision="_IIDXP_">
                    _IIDX_
                </port>
                <port id="3">
                    _IIDX_
                </port>
            </input>
            <output>
                <port id="4">
                    _IDATA_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="4" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="4" to-port="2"/>
        <edge from-layer="3" from-port="3" to-layer="4" to-port="3"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(scatterTF_test_params p) {
        std::string model = model_t;
        std::string inIdx;
        std::string inData;

        for (auto& idx : p.inIdxDim) {
            inIdx += "<dim>";
            inIdx += std::to_string(idx) + "</dim>\n";
        }

        for (auto& dct : p.inDataDim) {
            inData += "<dim>";
            inData += std::to_string(dct) + "</dim>\n";
        }

        REPLACE_WITH_STR(model, "_IIDX_", inIdx);
        REPLACE_WITH_STR(model, "_IIDXP_", p.inIdxPrecision);
        REPLACE_WITH_STR(model, "_IDATA_", inData);
        REPLACE_WITH_NUM(model, "_AX_", p.axis);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            scatterTF_test_params p = ::testing::WithParamInterface<scatterTF_test_params>::GetParam();
            std::string model = getModel(p);
            //std::cout << model << std::endl;
            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());

            //  Input Data
            InferenceEngine::Blob::Ptr srcData = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, p.inDataDim, InferenceEngine::TensorDesc::getLayoutByDims(p.inDataDim) });
            srcData->allocate();
            memcpy(srcData->buffer(), &p.inData[0], sizeof(float)*p.inData.size());
            auto * srcDataPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(srcData.get());
            if (srcDataPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            // Input Indexes
            InferenceEngine::Blob::Ptr srcIdx;
            if (p.inIdxPrecision == "I32") {
                srcIdx = InferenceEngine::make_shared_blob<int32_t>({ InferenceEngine::Precision::I32, p.inIdxDim, InferenceEngine::TensorDesc::getLayoutByDims(p.inIdxDim) });
                srcIdx->allocate();
                memcpy(static_cast<int32_t*>(srcIdx->buffer()), &p.inIdx[0], sizeof(int32_t)*p.inIdx.size());
                auto * srcIdxPtr = dynamic_cast<InferenceEngine::TBlob<int32_t>*>(srcIdx.get());
                if (srcIdxPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<int32_t>.";
            } else {
                srcIdx = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, p.inIdxDim, InferenceEngine::TensorDesc::getLayoutByDims(p.inIdxDim) });
                srcIdx->allocate();
                for (size_t i = 0; i < p.inIdx.size(); i++) {
                    static_cast<float*>(srcIdx->buffer())[i] = static_cast<float>(p.inIdx[i]);
                }
                auto * srcIdxPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(srcIdx.get());
                if (srcIdxPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<float>.";
            }

            // Input Updates
            InferenceEngine::Blob::Ptr srcUpd;
            srcUpd = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, p.inIdxDim, InferenceEngine::TensorDesc::getLayoutByDims(p.inIdxDim) });
            srcUpd->allocate();
            memcpy(static_cast<float*>(srcUpd->buffer()), &p.inUpd[0], sizeof(float)*p.inUpd.size());
            auto * srcUpdPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(srcUpd.get());
            if (srcUpdPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            //  Output Data
            InferenceEngine::OutputsDataMap out;
            out = net_reader.getNetwork().getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;
            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();
            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            //  Infer
            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("InputData", srcData));
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("InputIndexes", srcIdx));
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("InputUpdates", srcUpd));
            graph.Infer(srcs, outputBlobs);

            //  Check results
            if (memcmp((*output).data(), &p.reference[0], output->byteSize()) != 0)
                FAIL() << "Wrong result with compare TF reference!";
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

// Disabled these tests as they need to adjust with new specs:
// - new Scatter Update layer: like TF scatter_update
// - new Scatter Elements Update: like ONNX Scatter Elements
// See merge requests:
// DLDT #6005: Specification for the ScatterElementsUpdate layer
// DLDT #6091: Specification for ScatterUpdate operation
TEST_P(MKLDNNCPUExtScatterTFTests, DISABLED_TestsScatter) {}

INSTANTIATE_TEST_CASE_P(
        TestsScatter, MKLDNNCPUExtScatterTFTests,
        ::testing::Values(
// Params: inDataDim, inData, inIdxDim, inIdx, inUpd, axis, reference
        scatterTF_test_params{ "I32", { 3,3 },{ 0,0,0,0,0,0,0,0,0 },{ 2,3 },{ 1,0,2,0,2,1 },{ 1.,1.1,1.2,2,2.1,2.2 }, 0,{ 2,1.1,0,1,0,2.2,0,2.1,1.2 }},
        scatterTF_test_params{ "I32", { 3,3 },{ 0,0,0,0,0,0,0,0,0 },{ 2,3 },{ 1,0,2,0,2,1 },{ 1.,1.1,1.2,2,2.1,2.2 }, 1,{ 1.1,1,1.2,2,2.2,2.1,0,0,0 }},
        scatterTF_test_params{ "I32", { 1,5 },{ 1,2,3,4,5 },{ 1,2 },{ 1,3 },{ 1.1,2.1 }, 1,{ 1,1.1,3,2.1,5 }},
        scatterTF_test_params{"FP32", { 3,3 },{ 0,0,0,0,0,0,0,0,0 },{ 2,3 },{ 1,0,2,0,2,1 },{ 1.,1.1,1.2,2,2.1,2.2 }, 0,{ 2,1.1,0,1,0,2.2,0,2.1,1.2 }},
        scatterTF_test_params{"FP32", { 3,3 },{ 0,0,0,0,0,0,0,0,0 },{ 2,3 },{ 1,0,2,0,2,1 },{ 1.,1.1,1.2,2,2.1,2.2 }, 1,{ 1.1,1,1.2,2,2.2,2.1,0,0,0 }},
        scatterTF_test_params{"FP32", { 1,5 },{ 1,2,3,4,5 },{ 1,2 },{ 1,3 },{ 1.1,2.1 }, 1,{ 1,1.1,3,2.1,5 }}));
