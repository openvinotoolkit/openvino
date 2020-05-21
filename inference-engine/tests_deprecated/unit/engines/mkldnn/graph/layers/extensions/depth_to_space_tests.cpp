// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include "tests_common.hpp"
#include <ie_core.hpp>


using namespace ::testing;
using namespace std;
using namespace mkldnn;

struct depth_to_space_test_params {
    InferenceEngine::SizeVector in_shape;
    size_t block_size;
    InferenceEngine::SizeVector out_shape;

    std::vector<float> reference;
    std::vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

void ref_depth_to_space(
    InferenceEngine::TBlob<float> &src,
    InferenceEngine::TBlob<float> &dst,
    size_t block_size
) {
    size_t i;
    const float *src_data = src.data();
    InferenceEngine::SizeVector src_dims = src.getTensorDesc().getDims();
    InferenceEngine::SizeVector srcStrides = src.getTensorDesc().getBlockingDesc().getStrides();
    float* dst_data = dst.data();
    InferenceEngine::SizeVector dst_dims = dst.getTensorDesc().getDims();
    InferenceEngine::SizeVector dstStrides = dst.getTensorDesc().getBlockingDesc().getStrides();

    if (src_dims.size() < 3)
        FAIL() << " Incorrect number of input dimensions!";

    if (dst_dims.size() < 2)
        FAIL() << " Incorrect number of output dimensions!";

    if (block_size == 0)
        FAIL() << " Incorrect block_size parameter is zero!";

    if (src_dims[src_dims.size() - 3] % (block_size * block_size))
        FAIL() << " block_size parameter is incompatible with input tensor Color dimension size!";

    if (dst_dims.size() > 2 && src_dims[src_dims.size() - 3] != (dst_dims[dst_dims.size() - 3] * block_size * block_size))
        FAIL() << " Input/Output tensor Color dimension is incompatible with block_size!";

    if (dst_dims[dst_dims.size() - 2] != (src_dims[src_dims.size() - 2] * block_size))
        FAIL() << " Input/Output tensor Height dimension is incompatible with block_size!";

    if (dst_dims[dst_dims.size() - 1] != (src_dims[src_dims.size() - 1] * block_size))
        FAIL() << " Input/Output tensor Width dimension is incompatible with block_size!";

    size_t X = 1;
    for (i = 0; i < (src_dims.size() - 3); i++)
        X *= src_dims[i];

    size_t C = src_dims[src_dims.size() - 3];
    size_t H = src_dims[src_dims.size() - 2];
    size_t W = src_dims[src_dims.size() - 1];

    for (size_t x = 0, k = 0; x < X; ++x) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t c = 0; c < C; c += block_size) {
                for (size_t w = 0; w < W; ++w) {
                    for (size_t b = 0; b < block_size; ++b) {
                        size_t idx = x * C*H*W + (c + b) * H*W + h * W + w;
                        dst_data[k++] = src_data[idx];
                    }
                }
            }
        }
    }
}

void ref_space_to_depth(
    InferenceEngine::TBlob<float> &src,
    InferenceEngine::TBlob<float> &dst,
    size_t block_size
) {
    size_t i;
    const float *src_data = src.data();
    InferenceEngine::SizeVector src_dims = src.getTensorDesc().getDims();
    InferenceEngine::SizeVector srcStrides = src.getTensorDesc().getBlockingDesc().getStrides();
    float* dst_data = dst.data();
    InferenceEngine::SizeVector dst_dims = dst.getTensorDesc().getDims();
    InferenceEngine::SizeVector dstStrides = dst.getTensorDesc().getBlockingDesc().getStrides();

    if (dst_dims.size() < 3)
        FAIL() << " Incorrect number of output dimensions!";

    if (src_dims.size() < 2)
        FAIL() << " Incorrect number of input dimensions!";

    if (block_size == 0)
        FAIL() << " Incorrect block_size parameter is zero!";

    if (dst_dims[dst_dims.size() - 3] % (block_size * block_size))
        FAIL() << " block_size parameter is incompatible with input tensor Color dimension size!";

    if (src_dims.size() > 2 && dst_dims[dst_dims.size() - 3] != (src_dims[dst_dims.size() - 3] * block_size * block_size))
        FAIL() << " Input/Output tensor Color dimension is incompatible with block_size!";

    if (src_dims[src_dims.size() - 2] != (dst_dims[dst_dims.size() - 2] * block_size))
        FAIL() << " Input/Output tensor Height dimension is incompatible with block_size!";

    if (src_dims[src_dims.size() - 1] != (dst_dims[dst_dims.size() - 1] * block_size))
        FAIL() << " Input/Output tensor Width dimension is incompatible with block_size!";

    size_t X = 1;
    for (i = 0; i < (dst_dims.size() - 3); i++)
        X *= dst_dims[i];

    size_t C = dst_dims[dst_dims.size() - 3];
    size_t H = dst_dims[dst_dims.size() - 2];
    size_t W = dst_dims[dst_dims.size() - 1];

    for (size_t x = 0, k = 0; x < X; ++x) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t c = 0; c < C; c += block_size) {
                for (size_t w = 0; w < W; ++w) {
                    for (size_t b = 0; b < block_size; ++b) {
                        size_t idx = x * C*H*W + (c + b) * H*W + h * W + w;
                        dst_data[idx] = src_data[k++];
                    }
                }
            }
        }
    }
}

class MKLDNNCPUExtDepthToSpaceTests : public TestsCommon, public WithParamInterface<depth_to_space_test_params> {
    std::string model_t = R"V0G0N(
<net Name="DepthToSpace_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="input" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    _IN_
                </port>
            </output>s
        </layer>
        <layer name="output" id="2" type="DepthToSpace" precision="FP32">
            <data block_size="_BS_"/>
            <input>
                <port id="1">
                    _IN_
                </port>
           </input>
            <output>
                <port id="2">
                    _OUT_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(depth_to_space_test_params p) {
        std::string model = model_t;
        std::string in_shape, out_shape;

        for (size_t i = 0; i < p.in_shape.size(); i++) {
            in_shape += "<dim>";
            in_shape += std::to_string(p.in_shape[i]) + "</dim>\n";
        }
        for (size_t i = 0; i < p.out_shape.size(); i++) {
            out_shape += "<dim>";
            out_shape += std::to_string(p.out_shape[i]) + "</dim>\n";
        }
        REPLACE_WITH_STR(model, "_IN_", in_shape);
        REPLACE_WITH_STR(model, "_OUT_", out_shape);
        REPLACE_WITH_NUM(model, "_BS_", p.block_size);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            depth_to_space_test_params p = ::testing::WithParamInterface<depth_to_space_test_params>::GetParam();
            std::string model = getModel(p);

                        InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(network);

            // Output Data
            InferenceEngine::OutputsDataMap out;
            out = network.getOutputsInfo();
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

            // Check results
            InferenceEngine::SizeVector out_dims;
            ref_depth_to_space(*srcPtr, dst_ref, p.block_size);

            //  Check results
            if(p.reference.size())
                if (memcmp(dst_ref.data(), &p.reference[0], p.reference.size() * sizeof(float)) != 0)
                    FAIL() << "Wrong result with compare TF reference!";

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("input", src));

            // Infer
            graph.Infer(srcs, outputBlobs);
            compare(*output, dst_ref);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

class MKLDNNCPUExtSpaceToDepthTests : public TestsCommon, public WithParamInterface<depth_to_space_test_params> {
    std::string model_t = R"V0G0N(
<net Name="SpaceToDepth_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="input" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    _IN_
                </port>
            </output>s
        </layer>
        <layer name="output" id="2" type="SpaceToDepth" precision="FP32">
            <data block_size="_BS_"/>
            <input>
                <port id="1">
                    _IN_
                </port>
           </input>
            <output>
                <port id="2">
                    _OUT_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(depth_to_space_test_params p) {
        std::string model = model_t;
        std::string in_shape, out_shape;

        for (size_t i = 0; i < p.out_shape.size(); i++) {
            in_shape += "<dim>";
            in_shape += std::to_string(p.out_shape[i]) + "</dim>\n";
        }
        for (size_t i = 0; i < p.in_shape.size(); i++) {
            out_shape += "<dim>";
            out_shape += std::to_string(p.in_shape[i]) + "</dim>\n";
        }
        REPLACE_WITH_STR(model, "_IN_", in_shape);
        REPLACE_WITH_STR(model, "_OUT_", out_shape);
        REPLACE_WITH_NUM(model, "_BS_", p.block_size);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            depth_to_space_test_params p = ::testing::WithParamInterface<depth_to_space_test_params>::GetParam();
            std::string model = getModel(p);
            //std::cout << model;
                        InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(network);

            // Output Data
            InferenceEngine::OutputsDataMap out;
            out = network.getOutputsInfo();
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
            src = InferenceEngine::make_shared_blob<float>({ InferenceEngine::Precision::FP32, p.out_shape, InferenceEngine::TensorDesc::getLayoutByDims(p.out_shape) });
            src->allocate();
            if (p.reference.size())
                memcpy(static_cast<float*>(src->buffer()), &p.reference[0], sizeof(float)*p.reference.size());
            auto * srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());
            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            // Check results
            InferenceEngine::SizeVector out_dims;
            ref_space_to_depth(*srcPtr, dst_ref, p.block_size);

            //  Check results
            if (p.reference.size()) {
            //    fill_data_dbgval(src->buffer(), src->size());
            //    if (memcmp(dst_ref.data(), &p.reference[0], p.reference.size() * sizeof(float)) != 0)
            //        FAIL() << "Wrong result with compare TF reference!";
            }

            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("input", src));

            // Infer
            graph.Infer(srcs, outputBlobs);
            compare(*output, dst_ref);
        }
        catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};



class MKLDNNCPUExtDepthToSpaceToDepthTests : public TestsCommon, public WithParamInterface<depth_to_space_test_params> {
    std::string model_t = R"V0G0N(
<net Name="DepthToSpaceToDepth_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="input" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    _IN_
                </port>
            </output>s
        </layer>
        <layer name="intermediate" id="2" type="DepthToSpace" precision="FP32">
            <data block_size="_BS_"/>
            <input>
                <port id="1">
                    _IN_
                </port>
           </input>
            <output>
                <port id="2">
                    _OUT_
                </port>
            </output>
        </layer>
        <layer name="output" id="3" type="SpaceToDepth" precision="FP32">
            <data block_size="_BS_"/>
            <input>
                <port id="1">
                    _OUT_
                </port>
           </input>
            <output>
                <port id="2">
                    _IN_
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="1"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(depth_to_space_test_params p) {
        std::string model = model_t;
        std::string in_shape, out_shape;

        for (size_t i = 0; i < p.in_shape.size(); i++) {
            in_shape += "<dim>";
            in_shape += std::to_string(p.in_shape[i]) + "</dim>\n";
        }
        for (size_t i = 0; i < p.out_shape.size(); i++) {
            out_shape += "<dim>";
            out_shape += std::to_string(p.out_shape[i]) + "</dim>\n";
        }
        REPLACE_WITH_STR(model, "_IN_", in_shape);
        REPLACE_WITH_STR(model, "_OUT_", out_shape);
        REPLACE_WITH_NUM(model, "_BS_", p.block_size);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            depth_to_space_test_params p = ::testing::WithParamInterface<depth_to_space_test_params>::GetParam();
            std::string model = getModel(p);

                        InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(network);

            // Output Data
            InferenceEngine::OutputsDataMap out;
            out = network.getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

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

            // Infer
            graph.Infer(srcs, outputBlobs);
            compare(*output, *src);
        }
        catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNCPUExtDepthToSpaceTests, TestsDepthToSpace) {}
//  Test data vectors
static std::vector<float> test0 = { 0.f, 6.f, 1.f, 7.f, 2.f, 8.f, 12.f, 18.f, 13.f, 19.f, 14.f, 20.f, 3.f, 9.f, 4.f, 10.f, 5.f, 11.f, 15.f, 21.f, 16.f, 22.f, 17.f, 23.f};
INSTANTIATE_TEST_CASE_P(
    TestsDepthToSpace, MKLDNNCPUExtDepthToSpaceTests,
            ::testing::Values(
// Params: in_shape, block_size, out_shape, reference
                depth_to_space_test_params{ { 1, 4, 2, 3 }, 2, { 1, 1, 4, 6 }, test0 },
                depth_to_space_test_params{ { 4, 2, 3 }, 2, { 1, 1, 4, 6 }, test0 },
                depth_to_space_test_params{ { 1, 4, 2, 3 }, 2, { 4, 6 }, test0 },
                depth_to_space_test_params{ { 4, 2, 3 }, 2, { 4, 6 }, test0 },
                depth_to_space_test_params{ { 5, 4, 2, 3 }, 2, { 5, 1, 4, 6 }, test0 },
                depth_to_space_test_params{ { 2, 3, 5, 4, 2, 3 }, 2, { 2, 3, 5, 1, 4, 6 }, test0 }
));


TEST_P(MKLDNNCPUExtDepthToSpaceToDepthTests, TestsDepthToSpaceToDepth) {}
INSTANTIATE_TEST_CASE_P(
    TestsDepthToSpaceToDepth, MKLDNNCPUExtDepthToSpaceToDepthTests,
    ::testing::Values(
        // Params: in_shape, block_size, out_shape, reference
        depth_to_space_test_params{ { 1, 9, 2, 3 }, 3,{ 1, 1, 6, 9 },{} },
        depth_to_space_test_params{ { 16, 2, 3 }, 4,{ 1, 1, 8, 12 },{} },
        depth_to_space_test_params{ { 1, 25, 4, 3 }, 5,{ 20, 15 },{} },
        depth_to_space_test_params{ { 72, 10, 3 }, 6,{ 2, 60, 18 },{} },
        depth_to_space_test_params{ { 5, 8, 2, 3 }, 2,{ 5, 2, 4, 6 },{} },
        depth_to_space_test_params{ { 2, 3, 5, 16, 2, 3 }, 2,{ 2, 3, 5, 4, 4, 6 },{} }
));
