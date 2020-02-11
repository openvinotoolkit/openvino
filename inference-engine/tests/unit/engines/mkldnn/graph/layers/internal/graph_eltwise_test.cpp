// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "mkldnn_graph.h"

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include <mkldnn_extension_utils.h>
#include <cnn_network_impl.hpp>
#include "tests_common.hpp"

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

class MKLDNNGraphEltwise3InputsTests: public TestsCommon,
                                     public WithParamInterface<eltwise_test_params> {
    std::string model_t = R"V0G0N(
<net name="EltwiseOnly" version="3" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">__SRC_DIMS_1__
                </port>
            </output>
        </layer>
        <layer name="in2" type="Input" precision="FP32" id="2">
            <output>
                <port id="2">__SRC_DIMS_2__
                </port>
            </output>
        </layer>
        <layer name="in3" type="Input" precision="FP32" id="3">
            <output>
                <port id="3">__SRC_DIMS_3__
                </port>
            </output>
        </layer>
        <layer name="con" id="4" type="Eltwise" precision="FP32">
            <data operation="_OP_" _COEFF_/>
            <input>
                <port id="1">__SRC_DIMS_1__
                </port>
                <port id="2">__SRC_DIMS_2__
                </port>
                <port id="3">__SRC_DIMS_3__
                </port>
            </input>
            <output>
                <port id="4">__SRC_DIMS__
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

protected:
    std::string getModel(eltwise_test_params p) {
        std::string model = model_t;
        std::string op = select_op(p.op);

        std::string src_dims1;
        for (auto &dim : p.dims1) {
            src_dims1 += "\n                    <dim>";
            src_dims1 += std::to_string(dim) + "</dim>";
        }
        REPLACE_WITH_STR(model, "__SRC_DIMS_1__", src_dims1);

        std::string src_dims2;
        for (auto &dim : p.dims2) {
            src_dims2 += "\n                    <dim>";
            src_dims2 += std::to_string(dim) + "</dim>";
        }
        REPLACE_WITH_STR(model, "__SRC_DIMS_2__", src_dims2);

        std::string src_dims3;
        for (auto &dim : p.dims3) {
            src_dims3 += "\n                    <dim>";
            src_dims3 += std::to_string(dim) + "</dim>";
        }
        REPLACE_WITH_STR(model, "__SRC_DIMS_3__", src_dims3);

        std::string src_dims;
        std::vector<size_t> dims = p.dims1;
        for (int i = 0; i < dims.size(); i++) {
            dims[i] = std::max(p.dims1[i], p.dims2[i]);
            dims[i] = std::max(dims[i], p.dims3[i]);
        }
        for (auto &dim : dims) {
            src_dims += "\n                    <dim>";
            src_dims += std::to_string(dim) + "</dim>";
        }
        REPLACE_WITH_STR(model, "__SRC_DIMS__", src_dims);

        std::string scale;
        if (!p.scales.empty()) {
            scale = std::string("coeff=\"") + to_string_c_locale(p.scales) + std::string("\"");
        }
        REPLACE_WITH_STR(model, "_OP_", op);
        REPLACE_WITH_STR(model, "_COEFF_", scale);

        return model;
    }

    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            eltwise_test_params p = ::testing::WithParamInterface<eltwise_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());

            auto& nodes = graph.getNodes();
            for (int i = 0; i < nodes.size(); i++) {
                if (nodes[i]->getType() == MKLDNNPlugin::Eltwise) {
                    ASSERT_EQ(p.num_prim_desc, nodes[i]->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(nodes[i]->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, nodes[i]->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType, nodes[i]->getSelectedPrimitiveDescriptor()->getImplementationType());
                }
            }
            InferenceEngine::SizeVector dims_src1 = p.dims1;
            InferenceEngine::Layout layout1 = InferenceEngine::ANY;
            switch (p.dims1.size()) {
                case 4:
                    layout1 = InferenceEngine::NCHW;
                    break;
                case 5:
                    layout1 = InferenceEngine::NCDHW;
                    break;
            }
            InferenceEngine::SizeVector dims_src2 = p.dims2;
            InferenceEngine::Layout layout2 = InferenceEngine::ANY;
            switch (p.dims2.size()) {
                case 4:
                    layout2 = InferenceEngine::NCHW;
                    break;
                case 5:
                    layout2 = InferenceEngine::NCDHW;
                    break;
            }
            InferenceEngine::SizeVector dims_src3 = p.dims3;
            InferenceEngine::Layout layout3 = InferenceEngine::ANY;
            switch (p.dims3.size()) {
                case 4:
                    layout3 = InferenceEngine::NCHW;
                    break;
                case 5:
                    layout3 = InferenceEngine::NCDHW;
                    break;
            }

            InferenceEngine::Blob::Ptr src1 = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src1, layout1});
            src1->allocate();

            InferenceEngine::TBlob<float>* srcPtr1 = dynamic_cast<InferenceEngine::TBlob<float>*>(src1.get());

            if (srcPtr1 == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";
            fill_data_sine(src1->buffer(), src1->size(), 0.1, 0.9, 1);
            InferenceEngine::Blob::Ptr src2 = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src2, layout2});
            src2->allocate();

            InferenceEngine::TBlob<float>* srcPtr2 = dynamic_cast<InferenceEngine::TBlob<float>*>(src2.get());

            if (srcPtr2 == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";
            fill_data_sine(src2->buffer(), src2->size(), 0.1, 0.9, 2);
            InferenceEngine::Blob::Ptr src3 = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src3, layout3});
            src3->allocate();

            InferenceEngine::TBlob<float>* srcPtr3 = dynamic_cast<InferenceEngine::TBlob<float>*>(src3.get());

            if (srcPtr3 == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";
            fill_data_sine(src3->buffer(), src3->size(), 0.1, 0.9, 3);
            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src1));
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in2", src2));
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in3", src3));

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

            std::vector<InferenceEngine::TBlob<float>> src_vec = {*srcPtr1, *srcPtr2, *srcPtr3};

            ref_eltwise(src_vec, dst_ref, p);

            compare(*output, dst_ref, 0.0005f);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphEltwise3InputsTests, TestsEltwise) {}


INSTANTIATE_TEST_CASE_P(
        TestsEltwise, MKLDNNGraphEltwise3InputsTests,
        ::testing::Values(
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{1, 3, 3, 3}, eltwise_test_params::opType::Sum, "", 3, MKLDNNPlugin::impl_desc_type::ref, {
                        [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::ref, impl.getImplementationType());
                            ASSERT_EQ(3, impl.getConfig().inConfs.size());
                            ASSERT_EQ(1, impl.getConfig().outConfs.size());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(1).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(2).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                        }
                } },
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{1, 3, 3, 3}, eltwise_test_params::opType::Sum, "1.0,1.0,1.0", 3, MKLDNNPlugin::impl_desc_type::ref, {
                        [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::ref, impl.getImplementationType());
                            ASSERT_EQ(3, impl.getConfig().inConfs.size());
                            ASSERT_EQ(1, impl.getConfig().outConfs.size());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(1).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(2).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                        }
                } },
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{1, 3, 3, 3}, eltwise_test_params::opType::Sum, "1.5,0.5,-2.0", 3, MKLDNNPlugin::impl_desc_type::ref, {
                        [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::ref, impl.getImplementationType());
                            ASSERT_EQ(3, impl.getConfig().inConfs.size());
                            ASSERT_EQ(1, impl.getConfig().outConfs.size());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(1).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(2).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                        }
                } },
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{1, 3, 3, 3}, eltwise_test_params::opType::Prod, "", 3, MKLDNNPlugin::impl_desc_type::ref, {
                        [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::ref, impl.getImplementationType());
                            ASSERT_EQ(3, impl.getConfig().inConfs.size());
                            ASSERT_EQ(1, impl.getConfig().outConfs.size());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(1).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(2).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                        }
                } },
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{1, 3, 3, 3}, eltwise_test_params::opType::Max, "", 3, MKLDNNPlugin::impl_desc_type::ref, {
                        [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::ref, impl.getImplementationType());
                            ASSERT_EQ(3, impl.getConfig().inConfs.size());
                            ASSERT_EQ(1, impl.getConfig().outConfs.size());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(1).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().inConfs.at(2).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                        }
                } },
                eltwise_test_params{{1, 32, 16, 16, 16},{1, 32, 16, 16, 16},{1, 32, 16, 16, 16}, eltwise_test_params::opType::Sum, "", 3, MKLDNNPlugin::impl_desc_type::ref, {
                        [](MKLDNNPlugin::PrimitiveDescInfo impl) {
                            ASSERT_EQ(MKLDNNPlugin::impl_desc_type::ref, impl.getImplementationType());
                            ASSERT_EQ(3, impl.getConfig().inConfs.size());
                            ASSERT_EQ(1, impl.getConfig().outConfs.size());
                            ASSERT_EQ(InferenceEngine::Layout::NCDHW, impl.getConfig().inConfs.at(0).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCDHW, impl.getConfig().inConfs.at(1).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCDHW, impl.getConfig().inConfs.at(2).desc.getLayout());
                            ASSERT_EQ(InferenceEngine::Layout::NCDHW, impl.getConfig().outConfs.at(0).desc.getLayout());
                        }
                } },
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{1, 3, 3, 3}, eltwise_test_params::opType::Min, "", 3, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{1, 3, 3, 3}, eltwise_test_params::opType::Sub, "", 3, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{1, 3, 3, 3}, eltwise_test_params::opType::Div, "", 3, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{1, 3, 3, 3}, eltwise_test_params::opType::Logical_AND, "", 3, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{1, 3, 3, 3}, eltwise_test_params::opType::Logical_OR, "", 3, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{1, 3, 3, 3}, eltwise_test_params::opType::Logical_XOR, "", 3, MKLDNNPlugin::impl_desc_type::ref}
        ));
        
class MKLDNNGraphEltwise2InputsTests: public TestsCommon,
                                     public WithParamInterface<eltwise_test_params> {
    std::string model_t = R"V0G0N(
<net name="EltwiseOnly" version="2" precision="FP32">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">__SRC_DIMS_1__
                </port>
            </output>
        </layer>
        <layer name="in2" type="Input" precision="FP32" id="2">
            <output>
                <port id="2">__SRC_DIMS_2__
                </port>
            </output>
        </layer>
        <layer name="con" id="3" type="Eltwise" precision="FP32">
            <data operation="_OP_" _COEFF_/>
            <input>
                <port id="1">__SRC_DIMS_1__
                </port>
                <port id="2">__SRC_DIMS_2__
                </port>
            </input>
            <output>
                <port id="3">__SRC_DIMS__
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="2"/>
    </edges>
</net>
)V0G0N";

protected:
    std::string getModel(eltwise_test_params p) {
        std::string model = model_t;
        std::string op = select_op(p.op);

        std::string src_dims1 = "";
        for (auto &dim : p.dims1) {
            src_dims1 += "\n                    <dim>";
            src_dims1 += std::to_string(dim) + "</dim>";
        }
        REPLACE_WITH_STR(model, "__SRC_DIMS_1__", src_dims1);

        std::string src_dims2 = "";
        for (auto &dim : p.dims2) {
            src_dims2 += "\n                    <dim>";
            src_dims2 += std::to_string(dim) + "</dim>";
        }
        REPLACE_WITH_STR(model, "__SRC_DIMS_2__", src_dims2);

        std::string src_dims;
        std::vector<size_t> dims = (p.dims1.size() >= p.dims2.size()) ? p.dims1 : p.dims2;
        int i = dims.size() - 1, j = p.dims1.size() - 1, k = p.dims2.size() - 1;
        for (; j >= 0 && k >= 0; i--, j--, k-- ) {
            dims[i] = std::max(p.dims1[j], p.dims2[k]);
        }

        for (auto &dim : dims) {
            src_dims += "\n                    <dim>";
            src_dims += std::to_string(dim) + "</dim>";
        }
        REPLACE_WITH_STR(model, "__SRC_DIMS__", src_dims);

        std::string scale;
        if (!p.scales.empty()) {
            scale = std::string("coeff=\"") + to_string_c_locale(p.scales) + std::string("\"");
        }
        REPLACE_WITH_STR(model, "_OP_", op);
        REPLACE_WITH_STR(model, "_COEFF_", scale);

        return model;
    }

    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            eltwise_test_params p = ::testing::WithParamInterface<eltwise_test_params>::GetParam();
            std::string model = getModel(p);

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork());

            auto& nodes = graph.getNodes();
            for (int i = 0; i < nodes.size(); i++) {
                if (nodes[i]->getType() == MKLDNNPlugin::Eltwise) {
                    ASSERT_EQ(p.num_prim_desc, nodes[i]->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(nodes[i]->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, nodes[i]->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType, nodes[i]->getSelectedPrimitiveDescriptor()->getImplementationType());
                }
            }
            InferenceEngine::SizeVector dims_src1 = p.dims1;
            InferenceEngine::Blob::Ptr src1 = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src1, InferenceEngine::TensorDesc::getLayoutByDims(p.dims1) });
            src1->allocate();

            InferenceEngine::TBlob<float>* srcPtr1 = dynamic_cast<InferenceEngine::TBlob<float>*>(src1.get());

            if (srcPtr1 == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            fill_data_sine(src1->buffer(), src1->size(), 0.1, 0.9, 1);

            InferenceEngine::SizeVector dims_src2 = p.dims2;
            InferenceEngine::Blob::Ptr src2 = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src2, InferenceEngine::TensorDesc::getLayoutByDims(p.dims2) });
            src2->allocate();

            InferenceEngine::TBlob<float>* srcPtr2 = dynamic_cast<InferenceEngine::TBlob<float>*>(src2.get());

            if (srcPtr2 == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";
            
            fill_data_sine(src2->buffer(), src2->size(), 0.1, 0.9, 2);
            
            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src1));
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in2", src2));

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

            std::vector<InferenceEngine::TBlob<float>> src_vec = {*srcPtr1, *srcPtr2};

            ref_eltwise(src_vec, dst_ref, p);

            compare(*output, dst_ref, 0.0005f);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }

};

TEST_P(MKLDNNGraphEltwise2InputsTests, TestsEltwise) {}

INSTANTIATE_TEST_CASE_P(
        TestsEltwise, MKLDNNGraphEltwise2InputsTests,
        ::testing::Values(
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{}, eltwise_test_params::opType::Sum, "", 3, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{}, eltwise_test_params::opType::Prod, "", 3, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{}, eltwise_test_params::opType::Max, "", 3, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{}, eltwise_test_params::opType::Min, "", 3, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{}, eltwise_test_params::opType::Sub, "", 3, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{}, eltwise_test_params::opType::Div, "", 3, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{}, eltwise_test_params::opType::Squared_diff, "", 3, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{}, eltwise_test_params::opType::Logical_AND, "", 3, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{}, eltwise_test_params::opType::Logical_OR, "", 3, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{}, eltwise_test_params::opType::Logical_XOR, "", 3, MKLDNNPlugin::impl_desc_type::ref}
        ));

INSTANTIATE_TEST_CASE_P(
        TestsBroadcasting, MKLDNNGraphEltwise2InputsTests,
        ::testing::Values(
                eltwise_test_params{{1, 3, 1, 3},{1, 1, 3, 3},{}, eltwise_test_params::opType::Sum, "", 1, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 1, 3},{1, 1, 3, 3},{}, eltwise_test_params::opType::Prod, "", 1, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 1, 3},{1, 1, 3, 3},{}, eltwise_test_params::opType::Max, "", 1, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 1, 3},{1, 1, 3, 3},{}, eltwise_test_params::opType::Min, "", 1, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 1, 3},{1, 1, 3, 3},{}, eltwise_test_params::opType::Sub, "", 1, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 1, 3},{1, 1, 3, 3},{}, eltwise_test_params::opType::Div, "", 1, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 1, 3},{1, 1, 3, 3},{}, eltwise_test_params::opType::Squared_diff, "", 1, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 1, 3},{1, 1, 3, 3},{}, eltwise_test_params::opType::Logical_AND, "", 1, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 1, 3},{1, 1, 3, 3},{}, eltwise_test_params::opType::Logical_OR, "", 1, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 1, 3},{1, 1, 3, 3},{}, eltwise_test_params::opType::Logical_XOR, "", 1, MKLDNNPlugin::impl_desc_type::ref},
                //  batch broadcasting
                eltwise_test_params{{1, 3, 224},{224, 3, 1},{}, eltwise_test_params::opType::Sum, "", 2, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{2, 3, 1, 2},{1, 3, 2, 1},{}, eltwise_test_params::opType::Sub, "", 1, MKLDNNPlugin::impl_desc_type::ref}

        ));

INSTANTIATE_TEST_CASE_P(
        TestsDiffDims, MKLDNNGraphEltwise2InputsTests,
        ::testing::Values(
                eltwise_test_params{{},{1, 3},{}, eltwise_test_params::opType::Sum, "", 1, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3},{},{}, eltwise_test_params::opType::Sum, "", 1, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3},{3},{}, eltwise_test_params::opType::Sum, "", 1, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{},{1, 3, 3},{}, eltwise_test_params::opType::Sum, "", 2, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3},{},{}, eltwise_test_params::opType::Sum, "", 2, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3},{3},{}, eltwise_test_params::opType::Sum, "", 2, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3},{1, 3, 3},{}, eltwise_test_params::opType::Sum, "", 2, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3},{1, 3},{}, eltwise_test_params::opType::Sum, "", 2, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{},{1, 3, 3, 3},{}, eltwise_test_params::opType::Sum, "", 1, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3},{},{}, eltwise_test_params::opType::Sum, "", 1, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3},{1, 3, 3, 3},{}, eltwise_test_params::opType::Sum, "", 1, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3},{1, 3},{}, eltwise_test_params::opType::Sum, "", 1, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3},{1, 3, 3, 3},{}, eltwise_test_params::opType::Sum, "", 1, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3},{}, eltwise_test_params::opType::Sum, "", 1, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{},{1, 3, 3, 3, 3},{}, eltwise_test_params::opType::Sum, "", 1, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3, 3},{},{}, eltwise_test_params::opType::Sum, "", 1, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3},{1, 3, 3, 3, 3},{}, eltwise_test_params::opType::Sum, "", 1, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3, 3},{1, 3},{}, eltwise_test_params::opType::Sum, "", 1, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3},{1, 3, 3, 3, 3},{}, eltwise_test_params::opType::Sum, "", 1, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3, 3},{1, 3, 3},{}, eltwise_test_params::opType::Sum, "", 1, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3, 3},{}, eltwise_test_params::opType::Sum, "", 1, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3, 3},{1, 3, 3, 3},{}, eltwise_test_params::opType::Sum, "", 1, MKLDNNPlugin::impl_desc_type::ref}
        ));

class MKLDNNGraphEltwiseDynBatchTests: public MKLDNNGraphEltwise3InputsTests {
protected:
    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            eltwise_test_params p = ::testing::WithParamInterface<eltwise_test_params>::GetParam();
            std::string model = getModel(p);
            size_t MB = p.dims1[0];
            if (MB < 2)
                MB = 2;

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));
            InferenceEngine::CNNNetwork network = net_reader.getNetwork();
            auto implNet = dynamic_cast<InferenceEngine::details::CNNNetworkImpl *>(&((InferenceEngine::ICNNNetwork&)network));
            ASSERT_NE(nullptr, implNet) << "Failed to cast ICNNNetwork to CNNNetworkImpl";
            InferenceEngine::ResponseDesc resp;
            InferenceEngine::StatusCode sts  = implNet->setBatchSizeReshape(MB, &resp);
            ASSERT_EQ((int)InferenceEngine::StatusCode::OK, sts) << resp.msg;

            MKLDNNGraphTestClass graph;
            graph.setProperty({{InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED, InferenceEngine::PluginConfigParams::YES}});
            graph.CreateGraph(net_reader.getNetwork());

            InferenceEngine::SizeVector dims_src1 = p.dims1;
            InferenceEngine::Layout layout1 = InferenceEngine::ANY;
            switch (p.dims1.size()) {
                case 4:
                    layout1 = InferenceEngine::NCHW;
                    break;
                case 5:
                    layout1 = InferenceEngine::NCDHW;
                    break;
            }
            InferenceEngine::SizeVector dims_src2 = p.dims2;
            InferenceEngine::Layout layout2 = InferenceEngine::ANY;
            switch (p.dims2.size()) {
                case 4:
                    layout2 = InferenceEngine::NCHW;
                    break;
                case 5:
                    layout2 = InferenceEngine::NCDHW;
                    break;
            }
            InferenceEngine::SizeVector dims_src3 = p.dims3;
            InferenceEngine::Layout layout3 = InferenceEngine::ANY;
            switch (p.dims3.size()) {
                case 4:
                    layout3 = InferenceEngine::NCHW;
                    break;
                case 5:
                    layout3 = InferenceEngine::NCDHW;
                    break;
            }

            InferenceEngine::Blob::Ptr src1 = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src1, layout1});
            src1->allocate();

            InferenceEngine::TBlob<float>* srcPtr1 = dynamic_cast<InferenceEngine::TBlob<float>*>(src1.get());

            if (srcPtr1 == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            fill_data(src1->buffer(), src1->size());
            InferenceEngine::Blob::Ptr src2 = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src2, layout2});
            src2->allocate();

            InferenceEngine::TBlob<float>* srcPtr2 = dynamic_cast<InferenceEngine::TBlob<float>*>(src2.get());

            if (srcPtr2 == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";
            fill_data(src2->buffer(), src2->size());
            InferenceEngine::Blob::Ptr src3 = InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src3, layout3});
            src3->allocate();

            InferenceEngine::TBlob<float>* srcPtr3 = dynamic_cast<InferenceEngine::TBlob<float>*>(src3.get());

            if (srcPtr3 == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";
            fill_data(src3->buffer(), src3->size());
            InferenceEngine::BlobMap srcs;
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src1));
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in2", src2));
            srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in3", src3));

            InferenceEngine::OutputsDataMap out;
            out = net_reader.getNetwork().getOutputsInfo();
            InferenceEngine::BlobMap outputBlobs;

            std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;


            auto checkDepthwise = [](const MKLDNNPlugin::MKLDNNNodePtr& node) {
                return node->getType() == MKLDNNPlugin::Eltwise;
            };

            graph.checkDynBatch(srcs, outputBlobs, MB, MB, checkDepthwise);
            graph.checkDynBatch(srcs, outputBlobs, 1, MB, checkDepthwise);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphEltwiseDynBatchTests, TestsDynBatchEltwise) {}

INSTANTIATE_TEST_CASE_P(
        TestsDynBatchEltwise, MKLDNNGraphEltwiseDynBatchTests,
        ::testing::Values(
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{1, 3, 3, 3}, eltwise_test_params::opType::Sum, "", 3, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{1, 3, 3, 3}, eltwise_test_params::opType::Sum, "1.0,1.0,1.0", 3, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{1, 3, 3, 3}, eltwise_test_params::opType::Sum, "1.5,0.5,-2.0", 3, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{1, 3, 3, 3}, eltwise_test_params::opType::Prod, "", 3, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{1, 3, 3, 3}, eltwise_test_params::opType::Max, "", 3, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{1, 3, 3, 3}, eltwise_test_params::opType::Sub, "", 3, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{1, 3, 3, 3}, eltwise_test_params::opType::Min, "", 3, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{1, 3, 3, 3}, eltwise_test_params::opType::Div, "", 3, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{1, 3, 3, 3}, eltwise_test_params::opType::Pow, "", 3, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{1, 3, 3, 3}, eltwise_test_params::opType::Logical_AND, "", 3, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{1, 3, 3, 3}, eltwise_test_params::opType::Logical_OR, "", 3, MKLDNNPlugin::impl_desc_type::ref},
                eltwise_test_params{{1, 3, 3, 3},{1, 3, 3, 3},{1, 3, 3, 3}, eltwise_test_params::opType::Logical_XOR, "", 3, MKLDNNPlugin::impl_desc_type::ref}
                ));

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

            InferenceEngine::CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            MKLDNNGraphTestClass graph;
            ASSERT_NO_THROW(graph.CreateGraph(net_reader.getNetwork()));

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
            precisions_test_2params{ {  "U8", "FP32"}, 5, 1 },
            precisions_test_2params{ {"FP32",   "U8"}, 5, 1 },
            precisions_test_2params{ {  "U8",   "U8"}, 6, 2 }
        ));
