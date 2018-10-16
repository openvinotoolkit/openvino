// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "mkldnn_plugin/mkldnn_graph.h"
#include "mock_mkldnn_primitive.hpp"

#include "test_graph.hpp"

#include <mock_mkldnn_extension.hpp>
#include <mkldnn/mkldnn_extension_ptr.hpp>
#include <mock_error_listener.hpp>
#include <mkldnn_plugin/mkldnn_extension_mngr.h>
#include "tests_common.hpp"

using namespace ::testing;
using namespace std;
using namespace mkldnn;

class FakeGenericPrimitive : public InferenceEngine::MKLDNNPlugin::IMKLDNNGenericPrimitive {
public:
    virtual std::vector<InferenceEngine::MKLDNNPlugin::MKLDNNGenericFormats> GetSupportedFormats() noexcept {
        return {{{InferenceEngine::MKLDNNPlugin::MemoryFormat::nChw8c,
                         InferenceEngine::MKLDNNPlugin::MemoryFormat::nChw8c},
                        {InferenceEngine::MKLDNNPlugin::MemoryFormat::nChw8c},
                        InferenceEngine::MKLDNNPlugin::MemoryFormat::oIhw8i,
                        InferenceEngine::MKLDNNPlugin::MemoryFormat::x},
                {{InferenceEngine::MKLDNNPlugin::MemoryFormat::nchw,
                         InferenceEngine::MKLDNNPlugin::MemoryFormat::nChw8c},
                        {InferenceEngine::MKLDNNPlugin::MemoryFormat::nhwc},
                        InferenceEngine::MKLDNNPlugin::MemoryFormat::oihw}};
    };
    virtual void Execute() noexcept {
        std::cerr << "This is face primitive";
    };
};

class DoublePrimitive : public InferenceEngine::MKLDNNPlugin::IMKLDNNGenericPrimitive {
public:
    virtual std::vector<InferenceEngine::MKLDNNPlugin::MKLDNNGenericFormats> GetSupportedFormats() noexcept {
        return {{{InferenceEngine::MKLDNNPlugin::MemoryFormat::any},
                        {InferenceEngine::MKLDNNPlugin::MemoryFormat::any}}};
    };

    virtual void Execute() noexcept {
        const float *src_data = static_cast<float*>(inputs[0].data);
        float *dst_data = static_cast<float*>(outputs[0].data);

        size_t data_size = 0;
        for(size_t i = 0; i < inputs[0].dims.size(); i++) {
            if (!i) {
                data_size = inputs[0].dims[i];
            } else {
                data_size *= inputs[0].dims[i];
            }
        }

        for (size_t i = 0; i < data_size; i++) {
            dst_data[i] = src_data[i]*2;
        }
    }
};

class TwoDifferentOutputs : public InferenceEngine::MKLDNNPlugin::IMKLDNNGenericPrimitive {
public:
    virtual std::vector<InferenceEngine::MKLDNNPlugin::MKLDNNGenericFormats> GetSupportedFormats() noexcept {
        return {{{InferenceEngine::MKLDNNPlugin::MemoryFormat::any},
                        {InferenceEngine::MKLDNNPlugin::MemoryFormat::any,
                                InferenceEngine::MKLDNNPlugin::MemoryFormat::any}}};
    };

    virtual void Execute() noexcept {
        const float *src_data = static_cast<float*>(inputs[0].data);
        float *dst_data0 = static_cast<float*>(outputs[0].data);
        float *dst_data1 = static_cast<float*>(outputs[0].data);

        size_t out_data_size0 = 0;
        for(size_t i = 0; i < outputs[0].dims.size(); i++) {
            if (!i) {
                out_data_size0 = outputs[0].dims[i];
            } else {
                out_data_size0 *= outputs[0].dims[i];
            }
        }

        size_t out_data_size1 = 0;
        for(size_t i = 0; i < outputs[1].dims.size(); i++) {
            if (!i) {
                out_data_size1 = outputs[1].dims[i];
            } else {
                out_data_size1 *= outputs[1].dims[i];
            }
        }

        for (size_t i = 0; i < out_data_size0; i++) {
            dst_data0[i] = (*(src_data++))*2;
        }

        for (size_t i = 0; i < out_data_size1; i++) {
            dst_data1[i] = (*(src_data++))*3;
        }
    };
};

class FakeExtension : public InferenceEngine::MKLDNNPlugin::IMKLDNNExtension {
public:
    /**
     * @brief return extension version information
     * @param versionInfo pointer to version info, will be set by plugin
     */
    void GetVersion(const InferenceEngine::Version *&versionInfo) const noexcept override {}

    /**
     * @brief logging is used to track what is going on inside
     * @param listener - logging sink
     */
    void SetLogCallback(InferenceEngine::IErrorListener &listener) noexcept override {}

    /**
     * @brief creates generic layer and returns a pointer to an instance
     * @param primitive - pointer to newly created layer
     * @param layer - layer parameters (source for name, type, precision, attr, weights...)
     * @param utility -  pointer to MKLDNN reorder helper*
     * @param resp - Optional: a pointer to an already allocated object to contain extra information of a failure (if occurred)
     * @return Enumeration of the resulted action: OK (0) for success.
     */
    InferenceEngine::StatusCode CreateGenericPrimitive(InferenceEngine::MKLDNNPlugin::IMKLDNNGenericPrimitive *&primitive,
                                                       const InferenceEngine::CNNLayerPtr &layer,
                                                       InferenceEngine::ResponseDesc *resp) const noexcept override {
        if (layer->type == "CustomConvolution") {
            primitive = new FakeGenericPrimitive();
            return InferenceEngine::StatusCode::OK;
        }
        if (layer->type == "DoubleLayer") {
            primitive = new DoublePrimitive();
            return InferenceEngine::StatusCode::OK;
        }
        if (layer->type == "TwoDifferentOutputs") {
            primitive = new TwoDifferentOutputs();
            return InferenceEngine::StatusCode::OK;
        }
        return InferenceEngine::StatusCode::NOT_FOUND;
    }

    /**
     * @brief could be used to cleanup resources
     */
    void Unload() noexcept override {
    }

    void Release() noexcept override {
        delete this;
    }
};


class MKLDNNGraphGenericTestsOLD: public TestsCommon {
protected:
    virtual void SetUp() {
        TestsCommon::SetUp();
        extension.reset(new FakeExtension());
        extMgr.reset(new MKLDNNPlugin::MKLDNNExtensionManager());
        extMgr->AddExtension(extension);
    }
    std::shared_ptr<InferenceEngine::MKLDNNPlugin::IMKLDNNExtension> extension;
    MKLDNNPlugin::MKLDNNExtensionManager::Ptr extMgr;
};

TEST_F(MKLDNNGraphGenericTestsOLD, canGetPrimitiveDescriptorsList) {
    std::shared_ptr<MKLDNNPlugin::MKLDNNNode> node;
    InferenceEngine::DataPtr dataPtr;
    dataPtr.reset(new InferenceEngine::Data("test", {1, 3, 4, 5}, InferenceEngine::Precision::FP32, InferenceEngine::Layout::NCHW));
    InferenceEngine::CNNLayerPtr layerPtr;
    layerPtr.reset(new InferenceEngine::CNNLayer({"name", "CustomConvolution", InferenceEngine::Precision::FP32}));
    layerPtr->outData.push_back(dataPtr);
    mkldnn::engine eng(mkldnn::engine(mkldnn::engine::kind::cpu, 0));
    node.reset(MKLDNNPlugin::MKLDNNNode::CreateNode(layerPtr, eng, extMgr));
    ASSERT_EQ(MKLDNNPlugin::Type::Generic, node->getType());

    ASSERT_NO_THROW(node->getSupportedDescriptors());
}

template <typename data_t>
void ref_double(const InferenceEngine::TBlob<data_t> &src, InferenceEngine::TBlob<data_t> &dst) {
    const data_t *src_data = src.readOnly();
    data_t *dst_data = dst.data();

#pragma omp parallel for
    for (int i=0; i < src.size(); i++)
        dst_data[i] = src_data[i]*2;
}

template <typename data_t>
void ref_twoDifferent(const InferenceEngine::TBlob<data_t> &src, InferenceEngine::TBlob<data_t> &dst1, InferenceEngine::TBlob<data_t> &dst2) {
    const data_t *src_data = src.readOnly();
    data_t *dst_data1 = dst1.data();
    data_t *dst_data2 = dst2.data();

#pragma omp parallel for
    for (int i=0; i < dst1.size(); i++)
        dst_data1[i] = (*(src_data++))*2;

#pragma omp parallel for
    for (int i=0; i < dst2.size(); i++)
        dst_data2[i] = (*(src_data++))*3;
}

TEST_F(MKLDNNGraphGenericTestsOLD, ExecuteGenericPrimitive) {
    std::string model = R"V0G0N(
        <Net Name="DoubleLayer_Only" version="2" precision="FP32" batch="1">
            <layers>
                <layer name="in1" type="Input" precision="FP32" id="0">
                    <output>
                        <port id="0">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </output>
                </layer>
                <layer name="double_layer" id="1" type="DoubleLayer" precision="FP32">
                    <input>
                        <port id="1">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </input>
                    <output>
                        <port id="2">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
            </edges>
        </Net>
        )V0G0N";

    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(net_reader.getNetwork(), extMgr);

    InferenceEngine::SizeVector dims_src = {1, 3, 5, 5};

    InferenceEngine::Blob::Ptr src =
           InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, InferenceEngine::NCHW, dims_src);
    src->allocate();
    fill_data(src->buffer(), src->size());

    InferenceEngine::TBlob<float>* srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

    if (srcPtr == nullptr)
        FAIL() << "Cannot cast blob to TBlob<float>.";

    InferenceEngine::BlobMap srcs;
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src));

    InferenceEngine::OutputsDataMap out;
    out = net_reader.getNetwork().getOutputsInfo();
    InferenceEngine::BlobMap outputBlobs;

    std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

    InferenceEngine::TBlob<float>::Ptr output;
    output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    output->allocate();
    outputBlobs[item.first] = output;

    graph.Infer(srcs, outputBlobs);

    InferenceEngine::TBlob<float> dst_ref(item.second->getTensorDesc());
    dst_ref.allocate();

    ref_double(*srcPtr, dst_ref);

    compare(*output, dst_ref);
}

TEST_F(MKLDNNGraphGenericTestsOLD, DISABLED_ExecuteGenericPrimitiveWithTwoOutputs) {
    std::string model = R"V0G0N(
        <Net Name="DoubleLayer_Only" version="2" precision="FP32" batch="1">
            <layers>
                <layer name="in1" type="Input" precision="FP32" id="0">
                    <output>
                        <port id="0">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </output>
                </layer>
                <layer name="two_diff_layer" id="1" type="TwoDifferentOutputs" precision="FP32">
                    <input>
                        <port id="1">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </input>
                    <output>
                        <port id="2">
                            <dim>1</dim>
                            <dim>1</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                        <port id="3">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </output>
                </layer>
                <layer name="con" id="2" type="Concat" precision="FP32">
                    <concat_data axis="_AXIS_"/>
                    <input>
                        <port id="4">
                            <dim>1</dim>
                            <dim>1</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                        <port id="5">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </input>
                    <output>
                        <port id="6">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
                <edge from-layer="1" from-port="2" to-layer="2" to-port="4"/>
                <edge from-layer="1" from-port="3" to-layer="2" to-port="5"/>
            </edges>
        </Net>
        )V0G0N";

    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(net_reader.getNetwork(), extMgr);

    InferenceEngine::SizeVector dims_src = {1, 3, 5, 5};

    InferenceEngine::Blob::Ptr src =
            InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, InferenceEngine::NCHW, dims_src);
    src->allocate();
    fill_data(src->buffer(), src->size());

    InferenceEngine::TBlob<float>* srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

    if (srcPtr == nullptr)
        FAIL() << "Cannot cast blob to TBlob<float>.";

    InferenceEngine::BlobMap srcs;
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src));

    InferenceEngine::OutputsDataMap out;
    out = net_reader.getNetwork().getOutputsInfo();
    InferenceEngine::BlobMap outputBlobs;

    auto it = out.begin();

    std::pair<std::string, InferenceEngine::DataPtr> item = *it;
    InferenceEngine::DataPtr data1 = item.second;

    InferenceEngine::TensorDesc outputDesc1 = item.second->getTensorDesc();
    InferenceEngine::TBlob<float>::Ptr output1;
    output1 = InferenceEngine::make_shared_blob<float>(outputDesc1);
    output1->allocate();
    outputBlobs[item.first] = output1;


    item = *(it++);
    InferenceEngine::DataPtr data2 = item.second;
    InferenceEngine::TensorDesc outputDesc2 = item.second->getTensorDesc();
    InferenceEngine::TBlob<float>::Ptr output2;
    output2 = InferenceEngine::make_shared_blob<float>(outputDesc2);
    output2->allocate();
    outputBlobs[item.first] = output2;

    graph.Infer(srcs, outputBlobs);

    InferenceEngine::TBlob<float> dst_ref1(outputDesc1);
    dst_ref1.allocate();

    InferenceEngine::TBlob<float> dst_ref2(outputDesc2);
    dst_ref2.allocate();

    ref_twoDifferent(*srcPtr, dst_ref1, dst_ref2);

    compare(*output1, dst_ref1);
    compare(*output2, dst_ref2);
}
