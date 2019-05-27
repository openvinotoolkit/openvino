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

struct range_test_params {
    std::string                 precision;
    float                       start;
    float                       limit;
    float                       delta;
    InferenceEngine::SizeVector out_shape;

    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

template <typename data_t>
void ref_range(
    float start,
    float limit,
    float delta,
    InferenceEngine::TBlob<data_t> &dst
) {
    data_t* dst_data = dst.data();
    size_t work_amount_dst = std::floor(std::abs((limit - start) / delta));
    if (work_amount_dst != dst.size())
        FAIL() << "Range indexes exceeds data tensor dimension";

    data_t dst_value = static_cast<data_t>(start);
    for (size_t iwork = 0; iwork < work_amount_dst; ++iwork, dst_value += static_cast<data_t>(delta)) {
        dst_data[iwork] = dst_value;
    }
}

class MKLDNNCPUExtRangeTests : public TestsCommon, public WithParamInterface<range_test_params> {
    std::string model_t = R"V0G0N(
<net Name="Range_net" version="2" precision="_IIDXP_" batch="1">
    <layers>
        <layer name="start" type="Input" precision="_IIDXP_" id="1">
            <output>
                <port id="1">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="limit" type="Input" precision="_IIDXP_" id="2">
            <output>
                <port id="2">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="delta" type="Input" precision="_IIDXP_" id="3">
            <output>
                <port id="3">
                    <dim>1</dim>
                </port>
            </output>
        </layer>
        <layer name="output" id="2" type="Range" precision="_IIDXP_">
            <data/>
            <input>
                <port id="1">
                    <dim>1</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                </port>
                <port id="3">
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
        <edge from-layer="3" from-port="3" to-layer="2" to-port="3"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(range_test_params p) {
        std::string model = model_t;
        std::string out_shape;

        REPLACE_WITH_STR(model, "_IIDXP_", p.precision);
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
            range_test_params p = ::testing::WithParamInterface<range_test_params>::GetParam();
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

            // Input Data
            InferenceEngine::Blob::Ptr start_scalar;
            InferenceEngine::Blob::Ptr limit_scalar;
            InferenceEngine::Blob::Ptr delta_scalar;
            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();
            InferenceEngine::SizeVector scalar_dim(1, 1);
            InferenceEngine::BlobMap srcs;
            InferenceEngine::SizeVector out_dims;
            if (p.precision == "I32") {
                start_scalar = InferenceEngine::make_shared_blob<int32_t>({ InferenceEngine::Precision::I32, scalar_dim, InferenceEngine::TensorDesc::getLayoutByDims(scalar_dim) });
                start_scalar->allocate();
                static_cast<int32_t*>(start_scalar->buffer())[0] = static_cast<int32_t>(p.start);
                auto * start_scalarPtr = dynamic_cast<InferenceEngine::TBlob<int32_t>*>(start_scalar.get());
                if (start_scalarPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<int32_t>.";

                limit_scalar = InferenceEngine::make_shared_blob<int32_t>({ InferenceEngine::Precision::I32, scalar_dim, InferenceEngine::TensorDesc::getLayoutByDims(scalar_dim) });
                limit_scalar->allocate();
                static_cast<int32_t*>(limit_scalar->buffer())[0] = static_cast<int32_t>(p.limit);
                auto * limit_scalarPtr = dynamic_cast<InferenceEngine::TBlob<int32_t>*>(limit_scalar.get());
                if (limit_scalarPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<int32_t>.";

                delta_scalar = InferenceEngine::make_shared_blob<int32_t>({ InferenceEngine::Precision::I32, scalar_dim, InferenceEngine::TensorDesc::getLayoutByDims(scalar_dim) });
                delta_scalar->allocate();
                static_cast<int32_t*>(delta_scalar->buffer())[0] = static_cast<int32_t>(p.delta);
                auto * delta_scalarPtr = dynamic_cast<InferenceEngine::TBlob<int32_t>*>(delta_scalar.get());
                if (delta_scalarPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<int32_t>.";

                srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("start", start_scalar));
                srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("limit", limit_scalar));
                srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("delta", delta_scalar));

                // Output Blob
                InferenceEngine::TBlob<int32_t>::Ptr output;
                output = InferenceEngine::make_shared_blob<int32_t>(item.second->getTensorDesc());
                output->allocate();
                outputBlobs[item.first] = output;

                // Output Reference
                InferenceEngine::TBlob<int32_t> dst_ref(item.second->getTensorDesc());
                dst_ref.allocate();
                ref_range(p.start, p.limit, p.delta, dst_ref);

                // Infer
                graph.Infer(srcs, outputBlobs);
                for (int i = 0; i < dst_ref.size(); i++) {
                    if (dst_ref.data()[i] != (*output).data()[i])
                        FAIL() << "The difference between res_ptr[i] and ref_ptr[i]";
                }
            } else if (p.precision == "FP32") {
                start_scalar = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, scalar_dim, InferenceEngine::TensorDesc::getLayoutByDims(scalar_dim) });
                start_scalar->allocate();
                static_cast<float*>(start_scalar->buffer())[0] = p.start;
                auto * start_scalarPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(start_scalar.get());
                if (start_scalarPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<float>.";

                limit_scalar = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, scalar_dim, InferenceEngine::TensorDesc::getLayoutByDims(scalar_dim) });
                limit_scalar->allocate();
                static_cast<float*>(limit_scalar->buffer())[0] = p.limit;
                auto * limit_scalarPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(limit_scalar.get());
                if (limit_scalarPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<float>.";

                delta_scalar = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, scalar_dim, InferenceEngine::TensorDesc::getLayoutByDims(scalar_dim) });
                delta_scalar->allocate();
                static_cast<float*>(delta_scalar->buffer())[0] = p.delta;
                auto * delta_scalarPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(delta_scalar.get());
                if (delta_scalarPtr == nullptr)
                    FAIL() << "Cannot cast blob to TBlob<float>.";

                srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("start", start_scalar));
                srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("limit", limit_scalar));
                srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("delta", delta_scalar));

                // Output Blob
                InferenceEngine::Blob::Ptr output;
                output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
                output->allocate();
                outputBlobs[item.first] = output;

                // Output Reference
                InferenceEngine::TBlob<float> dst_ref(item.second->getTensorDesc());
                dst_ref.allocate();
                ref_range(p.start, p.limit, p.delta, dst_ref);

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

TEST_P(MKLDNNCPUExtRangeTests, TestsRange) {}

INSTANTIATE_TEST_CASE_P(
    TestsRange, MKLDNNCPUExtRangeTests,
            ::testing::Values(
// Params: precision, start, limit, delta, out_shape
                range_test_params{ "I32", 3.f, 18.f, 3.f, { 5 } },
                range_test_params{ "I32", 3.f, 1.f, -1.f, { 2 } },
                range_test_params{ "I32", 3.f, -3.f, -1.f, { 6 } },
                range_test_params{ "I32", 0.f, 5.f, 1.f, { 5 } },
                range_test_params{"FP32", 3.f, 18.f, 3.f, { 5 } },
                range_test_params{"FP32", 3.f, 1.f, -.5f, { 4 } },
                range_test_params{"FP32", 3.f, -1.f, -.5f, { 8 } },
                range_test_params{"FP32", 0.f, 5.f, 1.f, { 5 } }
            ));
