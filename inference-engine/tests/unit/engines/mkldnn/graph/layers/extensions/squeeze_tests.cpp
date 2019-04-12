// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "mkldnn_plugin/mkldnn_graph.h"

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include <mkldnn_plugin/mkldnn_extension_utils.h>
#include <extension/ext_list.hpp>
#include "tests_common.hpp"


using namespace ::testing;
using namespace std;
using namespace mkldnn;

struct squeeze_test_params {
    std::string                 inIdxPrecision;
    InferenceEngine::SizeVector in_shape;
    std::vector<int32_t>        indices_to_squeeze;
    InferenceEngine::SizeVector out_shape;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

void ref_squeeze(
    InferenceEngine::TBlob<float> &src,
    InferenceEngine::SizeVector &out_dims,
    std::vector<int32_t> indices_to_squeeze
) {
    InferenceEngine::SizeVector src_dims = src.getTensorDesc().getDims();

    if (indices_to_squeeze.size() == 0)
        FAIL() << " Index vector should be 1 dimension";

    for (size_t i = 0; i < indices_to_squeeze.size(); i++) {
        int32_t axis = indices_to_squeeze[i];
        if (axis < 0)
            axis += src_dims.size();

        if (axis > src_dims.size())
            FAIL() << " Index to squeeze exceeds data tensor dimension";
        else if (src_dims[axis] != 1)
            FAIL() << " Index to squeeze of data tensor dimension is not 1";
    }

    for (size_t j = 0; j < src_dims.size(); j++) {
        bool found = false;
        for (size_t i = 0; i < indices_to_squeeze.size(); i++) {
            int32_t axis = indices_to_squeeze[i];
            if (axis < 0)
                axis += src_dims.size();
            if (j == static_cast<size_t>(axis)) found = true;
        }
        if(!found) out_dims.push_back(src_dims[j]);
    }
}

class MKLDNNCPUExtSqueezeTests : public TestsCommon, public WithParamInterface<squeeze_test_params> {
    std::string model_t = R"V0G0N(
<net Name="Squeeze_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="input" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    _IN_
                </port>
            </output>
        </layer>
        <layer name="indices_to_squeeze" type="Input" precision="_IIDXP_" id="2">
            <output>
                <port id="2">
                    <dim>_DIM_SIZE_</dim>
                </port>
            </output>
        </layer>
        <layer name="output" id="2" type="Squeeze" precision="FP32">
            <data/>
            <input>
                <port id="1">
                    _IN_
                </port>
                <port id="2">
                    <dim>_DIM_SIZE_</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    _OUT_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="2" to-port="2"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(squeeze_test_params p) {
        std::string model = model_t;
        std::string in_shape;
        std::string out_shape;

        for (size_t i = 0; i < p.in_shape.size(); i++) {
            in_shape += "<dim>";
            in_shape += std::to_string(p.in_shape[i]) + "</dim>\n";
        }
        REPLACE_WITH_STR(model, "_IN_", in_shape);
        REPLACE_WITH_STR(model, "_IIDXP_", p.inIdxPrecision);
        REPLACE_WITH_NUM(model, "_DIM_SIZE_", p.indices_to_squeeze.size());
        if (p.out_shape.size()) {
            for (size_t i = 0; i < p.out_shape.size(); i++) {
                out_shape += "<dim>";
                out_shape += std::to_string(p.out_shape[i]) + "</dim>\n";
            }
        } else {
            out_shape = "<dim>1</dim>\n";
        }
        REPLACE_WITH_STR(model, "_OUT_", out_shape);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            squeeze_test_params p = ::testing::WithParamInterface<squeeze_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            InferenceEngine::Extension cpuExt(make_so_name("cpu_extension"));
            MKLDNNPlugin::MKLDNNExtensionManager::Ptr extMgr(new MKLDNNPlugin::MKLDNNExtensionManager());
            extMgr->AddExtension(InferenceEngine::IExtensionPtr(&cpuExt, [](InferenceEngine::IExtension*){}));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork(), extMgr);

            // Output Data
            InferenceEngine::OutputsDataMap out;
            out = net_reader.getNetwork().getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            // Output Reference
            InferenceEngine::TBlob<float> dst_ref(item.second->getTensorDesc());
            dst_ref.allocate();

            // Input Data
            InferenceEngine::Blob::Ptr src;
            src = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, p.in_shape, InferenceEngine::TensorDesc::getLayoutByDims(p.in_shape) });
            src->allocate();
            fill_data_dbgval(src->buffer(), src->size());
            auto * srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());
            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("input", src));

            InferenceEngine::Blob::Ptr seq_lengthsIdx;
            InferenceEngine::SizeVector seq_lengths_dim(1, p.indices_to_squeeze.size());
            if (p.inIdxPrecision == "I32") {
                seq_lengthsIdx = InferenceEngine::make_shared_blob<int32_t>({ InferenceEngine::Precision::I32, seq_lengths_dim, InferenceEngine::TensorDesc::getLayoutByDims(seq_lengths_dim) });
                seq_lengthsIdx->allocate();
                if (p.indices_to_squeeze.size())
                    memcpy(static_cast<int32_t*>(seq_lengthsIdx->buffer()), &p.indices_to_squeeze[0], sizeof(int32_t)*p.indices_to_squeeze.size());
                auto * seq_lengthsIdxPtr = dynamic_cast<InferenceEngine::TBlob<int32_t>*>(seq_lengthsIdx.get());
                if (seq_lengthsIdxPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<int32_t>.";

                srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("indices_to_squeeze", seq_lengthsIdx));
            } else if (p.inIdxPrecision == "FP32") {
                seq_lengthsIdx = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, seq_lengths_dim, InferenceEngine::TensorDesc::getLayoutByDims(seq_lengths_dim) });
                seq_lengthsIdx->allocate();
                if (p.indices_to_squeeze.size())
                    for (size_t i = 0; i < p.indices_to_squeeze.size(); i++) {
                        static_cast<float *>(seq_lengthsIdx->buffer())[i] = static_cast<float>(p.indices_to_squeeze[i]);
                    }
                auto * seq_lengthsIdxPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(seq_lengthsIdx.get());
                if (seq_lengthsIdxPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<float>.";

                srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("indices_to_squeeze", seq_lengthsIdx));
            }
            else {
                return;
            }

            // Check results
            InferenceEngine::SizeVector out_dims;
            ref_squeeze(*srcPtr, out_dims, p.indices_to_squeeze);
            if (out_dims.size() != p.out_shape.size())
                FAIL() << "Wrong out_shape size!";
            for (size_t i = 0; i < p.out_shape.size(); i++) {
                if (out_dims[i] != p.out_shape[i])
                    FAIL() << "Wrong out_shape dimensions!";
            }

            // Infer
            graph.Infer(srcs, outputBlobs);
            compare(*output, *src);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNCPUExtSqueezeTests, TestsSqueeze) {}

INSTANTIATE_TEST_CASE_P(
    TestsSqueeze, MKLDNNCPUExtSqueezeTests,
            ::testing::Values(
// Params: inIdxPrecision, in_shape, indices_to_squeeze, out_shape
                squeeze_test_params{ "I32",{ 1 },{ 0 },{ } },
                squeeze_test_params{ "I32",{ 1, 3, 1 },{ 0 },{ 3, 1 } },
                squeeze_test_params{ "I32",{ 1, 3, 1 },{ 2 },{ 1, 3 } },
                squeeze_test_params{ "I32",{ 1, 3, 1 },{ 0, 2 },{ 3 } },
                squeeze_test_params{ "I32",{ 1, 3, 1 },{ -1 },{ 1, 3 } },
                squeeze_test_params{ "I32",{ 1, 3, 1, 2 },{ 0, 2 },{ 3, 2 } },
                squeeze_test_params{"FP32",{ 1 },{ 0 },{} },
                squeeze_test_params{"FP32",{ 1, 3, 1 },{ 0 },{ 3, 1 } },
                squeeze_test_params{"FP32",{ 1, 3, 1 },{ 2 },{ 1, 3 } },
                squeeze_test_params{"FP32",{ 1, 3, 1 },{ 0, 2 },{ 3 } },
                squeeze_test_params{"FP32",{ 1, 3, 1 },{ -1 },{ 1, 3 } },
                squeeze_test_params{"FP32",{ 1, 3, 1, 2 },{ 0, 2 },{ 3, 2 } }
            ));
