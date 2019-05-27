// Copyright (C) 2018-2019 Intel Corporation
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

struct unsqueeze_test_params {
    std::string                 inIdxPrecision;
    InferenceEngine::SizeVector in_shape;
    std::vector<int32_t>        indices_to_set;
    InferenceEngine::SizeVector out_shape;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

void ref_unsqueeze(
    InferenceEngine::TBlob<float> &src,
    InferenceEngine::SizeVector &out_dims,
    std::vector<int32_t> indices_to_set
) {
    InferenceEngine::SizeVector src_dims = src.getTensorDesc().getDims();

    if (indices_to_set.size() == 0)
        FAIL() << " Index vector should be 1 dimension";

    size_t i, j, k, max = src_dims.size();
    for (size_t i = 0; i < indices_to_set.size(); i++) {
        if (indices_to_set[i] > max) max = indices_to_set[i];
    }
    max++;

    if ((indices_to_set.size() + src_dims.size()) < max)
        FAIL() << " Indices_to_set for unsqueeze layer is out of tensor dimension";

    max = indices_to_set.size() + src_dims.size();
    for (i = 0, j = 0, k = 0; i < max; i++) {
        if (k < indices_to_set.size() && i == indices_to_set[k]) {
            out_dims.push_back(1);
            k++;
        } else {
            out_dims.push_back(src_dims[j++]);
        }
    }
}

class MKLDNNCPUExtUnsqueezeTests : public TestsCommon, public WithParamInterface<unsqueeze_test_params> {
    std::string model_t = R"V0G0N(
<net Name="Unsqueeze_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="input" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    _IN_
                </port>
            </output>
        </layer>
        <layer name="indices_to_set" type="Input" precision="_IIDXP_" id="2">
            <output>
                <port id="2">
                    <dim>_DIM_SIZE_</dim>
                </port>
            </output>
        </layer>
        <layer name="output" id="2" type="Unsqueeze" precision="FP32">
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

    std::string getModel(unsqueeze_test_params p) {
        std::string model = model_t;
        std::string in_shape;
        std::string out_shape;

        for (size_t i = 0; i < p.in_shape.size(); i++) {
            in_shape += "<dim>";
            in_shape += std::to_string(p.in_shape[i]) + "</dim>\n";
        }
        REPLACE_WITH_STR(model, "_IN_", in_shape);
        REPLACE_WITH_STR(model, "_IIDXP_", p.inIdxPrecision);
        REPLACE_WITH_NUM(model, "_DIM_SIZE_", p.indices_to_set.size());
        for (size_t i = 0; i < p.out_shape.size(); i++) {
            out_shape += "<dim>";
            out_shape += std::to_string(p.out_shape[i]) + "</dim>\n";
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
            unsqueeze_test_params p = ::testing::WithParamInterface<unsqueeze_test_params>::GetParam();
            std::string model = getModel(p);
            ////std::cout << model;
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
            InferenceEngine::SizeVector seq_lengths_dim(1, p.indices_to_set.size());
            if (p.inIdxPrecision == "I32") {
                seq_lengthsIdx = InferenceEngine::make_shared_blob<int32_t>({ InferenceEngine::Precision::I32, seq_lengths_dim, InferenceEngine::TensorDesc::getLayoutByDims(seq_lengths_dim) });
                seq_lengthsIdx->allocate();
                if (p.indices_to_set.size())
                    memcpy(static_cast<int32_t*>(seq_lengthsIdx->buffer()), &p.indices_to_set[0], sizeof(int32_t)*p.indices_to_set.size());
                auto * seq_lengthsIdxPtr = dynamic_cast<InferenceEngine::TBlob<int32_t>*>(seq_lengthsIdx.get());
                if (seq_lengthsIdxPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<int32_t>.";

                srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("indices_to_set", seq_lengthsIdx));
            } else if (p.inIdxPrecision == "FP32") {
                seq_lengthsIdx = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, seq_lengths_dim, InferenceEngine::TensorDesc::getLayoutByDims(seq_lengths_dim) });
                seq_lengthsIdx->allocate();
                if (p.indices_to_set.size())
                    for (size_t i = 0; i < p.indices_to_set.size(); i++) {
                        static_cast<float *>(seq_lengthsIdx->buffer())[i] = static_cast<float>(p.indices_to_set[i]);
                    }
                auto * seq_lengthsIdxPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(seq_lengthsIdx.get());
                if (seq_lengthsIdxPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<float>.";

                srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("indices_to_set", seq_lengthsIdx));
            }
            else {
                return;
            }

            // Check results
            InferenceEngine::SizeVector out_dims;
            ref_unsqueeze(*srcPtr, out_dims, p.indices_to_set);
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

TEST_P(MKLDNNCPUExtUnsqueezeTests, TestsUnsqueeze) {}

INSTANTIATE_TEST_CASE_P(
    TestsUnsqueeze, MKLDNNCPUExtUnsqueezeTests,
            ::testing::Values(
// Params: inIdxPrecision, in_shape, indices_to_set, out_shape
                unsqueeze_test_params{ "I32",{ 3 },{ 0 },{ 1, 3 } },
                unsqueeze_test_params{ "I32",{ 3 },{ 0, 1, 2 },{ 1, 1, 1, 3 } },
                unsqueeze_test_params{ "I32",{ 3 },{ 0, 2, 3 },{ 1, 3, 1, 1 } },
                unsqueeze_test_params{ "I32",{ 2, 3 },{ 0, 3 },{ 1, 2, 3, 1 } },
                unsqueeze_test_params{ "I32",{ 2, 3 },{ 1 },{ 2, 1, 3 } },
                unsqueeze_test_params{"FP32",{ 3 },{ 0 },{ 1, 3 } },
                unsqueeze_test_params{"FP32",{ 3 },{ 0, 1, 2 },{ 1, 1, 1, 3 } },
                unsqueeze_test_params{"FP32",{ 3 },{ 0, 2, 3 },{ 1, 3, 1, 1 } },
                unsqueeze_test_params{"FP32",{ 2, 3 },{ 0, 3 },{ 1, 2, 3, 1 } },
                unsqueeze_test_params{"FP32",{ 2, 3 },{ 1 },{ 2, 1, 3 } }
            ));
