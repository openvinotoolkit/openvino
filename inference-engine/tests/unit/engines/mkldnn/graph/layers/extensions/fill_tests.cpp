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


using namespace ::testing;
using namespace std;
using namespace mkldnn;

struct fill_test_params {
    std::string                 precision;
    InferenceEngine::SizeVector out_shape;
    float                       value;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

class MKLDNNCPUExtFillTests : public TestsCommon, public WithParamInterface<fill_test_params> {
    std::string model_t = R"V0G0N(
<net Name="Fill_net" version="2" precision="_IIDXP_" batch="1">
    <layers>
        <layer name="dims" type="Input" precision="I32" id="1">
            <output>
                <port id="1">
                    <dim>_DIM_SIZE_</dim>
                </port>
            </output>
        </layer>
        <layer name="value" type="Input" precision="_IIDXP_" id="2">
            <output>
                <port id="2">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="output" id="2" type="Fill" precision="_IIDXP_">
            <data/>
            <input>
                <port id="1">
                    <dim>_DIM_SIZE_</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
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

    std::string getModel(fill_test_params p) {
        std::string model = model_t;
        std::string out_shape;

        REPLACE_WITH_STR(model, "_IIDXP_", p.precision);
        for (size_t i = 0; i < p.out_shape.size(); i++) {
            out_shape += "<dim>";
            out_shape += std::to_string(p.out_shape[i]) + "</dim>\n";
        }
        REPLACE_WITH_STR(model, "_OUT_", out_shape);
        REPLACE_WITH_NUM(model, "_DIM_SIZE_", p.out_shape.size());

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            fill_test_params p = ::testing::WithParamInterface<fill_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());

            // Output Data
            InferenceEngine::OutputsDataMap out;
            out = net_reader.getNetwork().getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            // Input Data
            InferenceEngine::Blob::Ptr dims;
            InferenceEngine::SizeVector vector_dim(1, p.out_shape.size());
            dims = InferenceEngine::make_shared_blob<int32_t>({ InferenceEngine::Precision::I32, vector_dim, InferenceEngine::TensorDesc::getLayoutByDims(vector_dim) });
            dims->allocate();
            for (size_t i = 0; i < p.out_shape.size(); i++) {
                static_cast<int32_t*>(dims->buffer())[i] = static_cast<int32_t>(p.out_shape[i]);
            }
            auto * srcPtr = dynamic_cast<InferenceEngine::TBlob<int32_t>*>(dims.get());
            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<int32_t>.";

            InferenceEngine::BlobMap srcs;
            InferenceEngine::Blob::Ptr value_scalar;
            InferenceEngine::SizeVector value_scalar_dim(1, 1);
            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();
            if (p.precision == "I32") {
                value_scalar = InferenceEngine::make_shared_blob<int32_t>({ InferenceEngine::Precision::I32, value_scalar_dim, InferenceEngine::TensorDesc::getLayoutByDims(value_scalar_dim) });
                value_scalar->allocate();
                static_cast<int32_t*>(value_scalar->buffer())[0] = static_cast<int32_t>(p.value);
                auto * value_scalarPtr = dynamic_cast<InferenceEngine::TBlob<int32_t>*>(value_scalar.get());
                if (value_scalarPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<int32_t>.";

                srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("dims", dims));
                srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("value", value_scalar));

                // Output Blob
                InferenceEngine::TBlob<int32_t>::Ptr output;
                output = InferenceEngine::make_shared_blob<int32_t>(item.second->getTensorDesc());
                output->allocate();
                outputBlobs[item.first] = output;

                // Output Reference
                InferenceEngine::TBlob<int32_t> dst_ref(item.second->getTensorDesc());
                dst_ref.allocate();
                std::fill_n(static_cast<int32_t*>(dst_ref.data()), dst_ref.size(), static_cast<int32_t>(p.value));

                // Infer
                graph.Infer(srcs, outputBlobs);
                for (int i = 0; i < dst_ref.size(); i++) {
                    if(dst_ref.data()[i] != (*output).data()[i])
                        FAIL() << "The difference between res_ptr[i] and ref_ptr[i]";
                }
            } else if (p.precision == "FP32") {
                value_scalar = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, value_scalar_dim, InferenceEngine::TensorDesc::getLayoutByDims(value_scalar_dim) });
                value_scalar->allocate();
                static_cast<float*>(value_scalar->buffer())[0] = p.value;
                auto * value_scalarPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(value_scalar.get());
                if (value_scalarPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<float>.";

                srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("dims", dims));
                srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("value", value_scalar));

                // Output Blob
                InferenceEngine::TBlob<float>::Ptr output;
                output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
                output->allocate();
                outputBlobs[item.first] = output;

                // Output Reference
                InferenceEngine::TBlob<float> dst_ref(item.second->getTensorDesc());
                dst_ref.allocate();
                std::fill_n(static_cast<float*>(dst_ref.data()), dst_ref.size(), p.value);

                // Infer
                graph.Infer(srcs, outputBlobs);
                compare(*output, dst_ref);
            } else {
                return;
            }
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNCPUExtFillTests, TestsFill) {}

INSTANTIATE_TEST_CASE_P(
    TestsFill, MKLDNNCPUExtFillTests,
            ::testing::Values(
// Params: precision, value, out_shape
                fill_test_params{ "I32", { 1 }, 1.f },
                fill_test_params{ "I32", { 1, 3, 1 }, 1.f },
                fill_test_params{ "I32", { 2, 3, 6 }, -1.f },
                fill_test_params{"FP32", { 2, 3, 6 }, -1.f },
                fill_test_params{"FP32", { 1 }, 1.f },
                fill_test_params{"FP32", { 1, 3, 1, 2 }, .5f },
                fill_test_params{"FP32", { 4, 3, 2, 5, 4, 2 }, .25f }
            ));
