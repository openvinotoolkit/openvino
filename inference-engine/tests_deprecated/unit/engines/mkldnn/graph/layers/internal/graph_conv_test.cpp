// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_graph.hpp"

#include "single_layer_common.hpp"
#include <legacy/cnn_network_impl.hpp>
#include "tests_common.hpp"
#include <ie_core.hpp>
#include <ie_system_conf.h>

using namespace InferenceEngine;
using namespace ::testing;
using namespace std;
using namespace mkldnn;

struct conv_test_params {
    // Formats: NCHW, NCDHW
    vector<size_t> dims;
    // Formats: WH, WHD
    vector<size_t> kernel;
    vector<size_t> strides;
    vector<size_t> pads_begin;
    vector<size_t> pads_end;

    size_t out_c;
    size_t grp_c;
    string auto_pad;

    size_t num_prim_desc;

    int selectedType;
    bool defaultPrimitivesPriority;
    vector<MKLDNNPlugin::impl_desc_type> preferTypes;

    vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

template <typename data_t>
void ref_conv(const TBlob<data_t> &src, const data_t *weights, const size_t weightsSize,
                TBlob<data_t> &dst, struct conv_test_params prm) {
    SizeVector src_dims = src.getTensorDesc().getDims();
    auto dims_size = src_dims.size();

    size_t KW = prm.kernel[X_AXIS];
    size_t KH = prm.kernel[Y_AXIS];
    size_t KD = dims_size == 5 ? prm.kernel[Z_AXIS] : 1u;
    size_t GC = prm.grp_c;

    size_t IC = src_dims[1];
    size_t ID = dims_size == 5 ? src_dims[dims_size - 3] : 1u;
    size_t IH = src_dims[dims_size - 2];
    size_t IW = src_dims[dims_size - 1];

    size_t OW = (IW + prm.pads_end[X_AXIS] + prm.pads_begin[X_AXIS] - prm.kernel[X_AXIS]) / prm.strides[X_AXIS] + 1u;
    size_t OH = (IH + prm.pads_end[Y_AXIS] + prm.pads_begin[Y_AXIS] - prm.kernel[Y_AXIS]) / prm.strides[Y_AXIS] + 1u;
    size_t OD = dims_size == 5 ? (ID + 2u * prm.pads_begin[Z_AXIS] - prm.kernel[Z_AXIS]) / prm.strides[Z_AXIS] + 1u : 1u;
    size_t OC = prm.out_c;


    const data_t *src_data = src.readOnly();
    const data_t *weights_data = weights;
    const data_t *bias_data = weights_data + KW * KH * KD * OC * IC / GC;
    data_t *dst_data = dst.data();

    IE_ASSERT(KW * KH * KD * OC * IC / GC + OC == weightsSize);
    SizeVector dst_dims = dst.getTensorDesc().getDims();
    auto dst_dims_size = dst_dims.size();
    IE_ASSERT(OW == dst_dims[dst_dims_size - 1]);
    IE_ASSERT(OH == dst_dims[dst_dims_size - 2]);

    size_t SC1 = OH * OW;
    size_t SC2 = SC1 * OD;
    size_t SC3 = OC / GC;
    size_t SC4 = SC2 * SC3;

    size_t IC1 = IH * IW;
    size_t IC2 = IC1 * ID;
    size_t IC3 = IC / GC;
    size_t IC4 = IC2 * IC3;

    size_t KC1 = KH * KW;
    size_t KC2 = KC1 * KD;
    size_t KC3 = IC3 * KC2;
    size_t KC4 = SC3 * KC3;

    for (uint32_t g = 0; g < GC; g++) {
        size_t gc = g * SC4;
        size_t goc = g * SC3;
        size_t gic = g * IC4;
        size_t gkc = g * KC4;
        for (uint32_t oc = 0; oc < OC / GC; oc++) {
            size_t cc = gc + oc * SC2;
            size_t gooc = goc + oc;
            size_t gkoc = gkc + oc * KC3;
            for (uint32_t od = 0; od < OD; od++) {
                size_t dc = cc + od * SC1;
                for (uint32_t oh = 0; oh < OH; oh++) {
                    size_t hc = dc + oh * OW;
                    for (uint32_t ow = 0; ow < OW; ow++) {
                        size_t oidx = hc + ow;

                        dst_data[oidx] = bias_data[gooc];

                        for (size_t ic = 0; ic < IC / GC; ic++) {
                            size_t icc = gkoc + ic * KC2;
                            size_t kicc = gic + ic * IC2;
                            for (size_t kd = 0; kd < KD; kd++) {
                                int32_t id = dims_size == 5 ? od * prm.strides[Z_AXIS] - prm.pads_begin[Z_AXIS] + kd : 0;
                                if (id < 0 || id >= (int32_t)ID) continue;
                                size_t kidc = kicc + id * IC1;
                                size_t kdc = icc + kd * KC1;
                                for (size_t kh = 0; kh < KH; kh++) {
                                    int32_t ih = oh * prm.strides[Y_AXIS] - prm.pads_begin[Y_AXIS] + kh;
                                    if (ih < 0 || ih >= (int32_t)IH) continue;
                                    size_t kihc = kidc + ih * IW;
                                    size_t khc = kdc + kh * KW;
                                    for (size_t kw = 0; kw < KW; kw++) {
                                        int32_t iw = ow * prm.strides[X_AXIS] - prm.pads_begin[X_AXIS] + kw;
                                        if (iw < 0 || iw >= (int32_t)IW) continue;

                                        size_t iidx = kihc + iw;
                                        size_t widx = khc + kw;

                                        dst_data[oidx] += src_data[iidx] * weights_data[widx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

class MKLDNNGraphConvolutionTests: public TestsCommon,
                                   public WithParamInterface<conv_test_params> {
    std::string model_t_5D = R"V0G0N(
<net name="Convolution_Only" version="4" precision="FP32" batch="1">
    <layers>
        <layer name="in1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">__SRC_DIMS__
                </port>
            </output>
        </layer>
        <layer name="conv1" id="1" type="Convolution" precision="FP32">
            <convolution _AP_ kernel="_K_"
                         pads_begin="_PB_"  pads_end="_PE_"
                         strides="_KS_"
                         output="_OC_"  group="_GC_" _PRIM_PRIORITY_/>

            <weights offset="0" size="_S1_" />
            <biases offset="_S1_" size="_S2_" />

            <input>
                <port id="1">__SRC_DIMS__
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>_IN_</dim>
                    <dim>_OC_</dim>__DST_DIMS__
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
</net>
)V0G0N";

protected:
    std::string getModel(conv_test_params p) {
        std::string model = model_t_5D;
        std::string s_dims;
        for (auto& dim : p.dims) {
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(dim) + "</dim>";
        }
	REPLACE_WITH_STR(model, "__SRC_DIMS__", s_dims);

        s_dims = "";
        int k_len = p.kernel.size();
        for (size_t i = 2; i < p.dims.size(); i++) {
            size_t inx = k_len - i + 1;
            size_t dim = (p.dims[i] + p.pads_end[inx] + p.pads_begin[inx] - p.kernel[inx]) / p.strides[inx] + 1lu;
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(dim) + "</dim>";
        }
	REPLACE_WITH_STR(model, "__DST_DIMS__", s_dims);

        REPLACE_WITH_NUM(model, "_IN_", p.dims[0]);

        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_K_", p.kernel);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_KS_", p.strides);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_PB_", p.pads_begin);
        REPLACE_WITH_NUM_VECTOR_REVERSE(model, "_PE_", p.pads_end);
        string auto_pad;
        if (!p.auto_pad.empty()) auto_pad = string("auto_pad=") + string("\"") + p.auto_pad + string("\"");
        REPLACE_WITH_STR(model, "_AP_", auto_pad);

        REPLACE_WITH_NUM(model, "_GC_", p.grp_c);
        REPLACE_WITH_NUM(model, "_OC_", p.out_c);

        size_t w_data_size = 1;
        for (auto ker : p.kernel) {
            w_data_size *= ker;
        }

        w_data_size = (w_data_size * p.out_c * p.dims[1] / p.grp_c) * sizeof(float);
        size_t b_data_size = p.out_c * sizeof(float);

        REPLACE_WITH_NUM(model, "_S1_", w_data_size);
        REPLACE_WITH_NUM(model, "_S2_", b_data_size);

        std::string primitivesPriorityStr;
        if (!p.defaultPrimitivesPriority) {
            std::string impls;
            for (const auto& preferType : p.preferTypes) {
                if (!impls.empty())
                    impls += ",";
                impls += "cpu:" + MKLDNNGraphTestClass::getStrPrimitiveDescriptorType(preferType);
            }
            primitivesPriorityStr = "PrimitivesPriority=\"" + impls + "\"";
        }
        REPLACE_WITH_STR(model, "_PRIM_PRIORITY_", primitivesPriorityStr);

        return model;
    }

    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            conv_test_params p = ::testing::WithParamInterface<conv_test_params>::GetParam();
            std::string model = getModel(p);

            size_t blob_size = p.out_c * p.dims[1] / p.grp_c;
            for (auto k : p.kernel) {
                blob_size *= k;
            }
            blob_size = (blob_size + p.out_c) * sizeof(float);
            TBlob<uint8_t> *weights = new TBlob<uint8_t>
                    ({ Precision::U8, {blob_size}, C });
            weights->allocate();

            fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

            TBlob<uint8_t>::Ptr weights_ptr = TBlob<uint8_t>::Ptr(weights);

            InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, weights_ptr));

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(network);

            auto& nodes = graph.getNodes();
            nodes = graph.getNodes();
            bool isWino = false;
            for (auto &node : nodes) {
                if (node->getType() == MKLDNNPlugin::Convolution) {
                    ASSERT_LE(p.num_prim_desc, node->getSupportedPrimitiveDescriptors().size());
                    for (const auto prim : node->getSupportedPrimitiveDescriptors()) {
                        if (p.defaultPrimitivesPriority) {
                            if (prim.getImplementationType() & MKLDNNPlugin::impl_desc_type::gemm)
                                FAIL() << "There should be no gemm implementation in supportedPrimitiveDescriptors";
                        }
                        std::cout << MKLDNNGraphTestClass::getStrPrimitiveDescriptorType(prim.getImplementationType()) << " ";
                    }
                    std::cout << std::endl;
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(node->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, node->getSelectedPrimitiveDescriptor());
                    if (InferenceEngine::with_cpu_x86_avx512f() &&
                            InferenceEngine::with_cpu_x86_avx512_core()
                            && !p.preferTypes.empty()
                            && p.preferTypes[0] == MKLDNNPlugin::impl_desc_type::jit_avx512_winograd) {
                        isWino = true;
                        ASSERT_EQ(p.preferTypes[0], node->getSelectedPrimitiveDescriptor()->getImplementationType());
                    } else {
                        ASSERT_EQ(p.selectedType,
                                  node->getSelectedPrimitiveDescriptor()->getImplementationType() & p.selectedType);
                    }
                }
            }

            Layout layout = ANY;
            switch (p.dims.size()) {
                case 4:
                    layout = NCHW;
                    break;
                case 5:
                    layout = NCDHW;
                    break;
            }

            Blob::Ptr src = make_shared_blob<float>
                    ({ Precision::FP32, p.dims, layout });
            src->allocate();
            fill_data(src->buffer(), src->size());

            auto * srcPtr = dynamic_cast<TBlob<float>*>(src.get());

            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            BlobMap srcs;
            srcs.insert(std::pair<std::string, Blob::Ptr>("in1", src));

            OutputsDataMap out;
            out = network.getOutputsInfo();
            BlobMap outputBlobs;

            std::pair<std::string, DataPtr> item = *out.begin();

            TBlob<float>::Ptr output;
            output = make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            graph.Infer(srcs, outputBlobs);

            TBlob<float> dst_ref(item.second->getTensorDesc());
            dst_ref.allocate();
            ref_conv(*srcPtr, (const float *)weights->buffer(), weights->size() / sizeof(float), dst_ref, p);
            compare(*output, dst_ref, 0.0002f);
        } catch (const details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphConvolutionTests, TestsConvolution) {}


INSTANTIATE_TEST_CASE_P(
    TestConvolution, MKLDNNGraphConvolutionTests,
    ::testing::Values(
        /*0*/   conv_test_params{{1, 9, 16, 32}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, 17, 1, "same_upper", 6,
                                 MKLDNNPlugin::impl_desc_type::jit | MKLDNNPlugin::impl_desc_type::_1x1, false },
                conv_test_params{{1, 9, 32, 16}, {2, 4}, {1, 1}, {1, 1}, {0, 2}, 17, 1, "", 4, MKLDNNPlugin::impl_desc_type::jit, false },
                conv_test_params{{1, 9, 32, 16}, {2, 4}, {2, 1}, {0, 0}, {0, 0}, 17, 1, "", 4, MKLDNNPlugin::impl_desc_type::jit, false },
                conv_test_params{{1, 3, 40, 40}, {3, 3}, {1, 2}, {0, 0}, {0, 0}, 20, 1, "", 4, MKLDNNPlugin::impl_desc_type::jit, false },
                conv_test_params{{1, 1, 40, 40}, {3, 3}, {1, 2}, {0, 0}, {0, 0}, 20, 1, "", 4, MKLDNNPlugin::impl_desc_type::jit, false },
                conv_test_params{{1, 1, 32, 16}, {2, 4}, {2, 1}, {0, 0}, {0, 0}, 17, 1, "", 4, MKLDNNPlugin::impl_desc_type::jit, false },
                conv_test_params{{1, 9, 32, 16}, {2, 4}, {1, 1}, {0, 0}, {0, 0}, 17, 1, "", 4, MKLDNNPlugin::impl_desc_type::jit, false },
                conv_test_params{{1, 4, 54, 96}, {3, 3}, {1, 1}, {1, 1}, {0, 0}, 64, 1, "", 3, MKLDNNPlugin::impl_desc_type::jit, false },
                // 5D
        /*8*/   conv_test_params{{1, 3, 15, 20, 20}, {3, 3, 3}, {2, 2, 2}, {0, 0, 0}, {0, 0, 0}, 64, 1, "", 2, MKLDNNPlugin::impl_desc_type::jit, false },
                conv_test_params{{1, 24, 15, 20, 20}, {3, 3, 3}, {2, 2, 2}, {0, 0, 0}, {0, 0, 0}, 64, 1, "", 2, MKLDNNPlugin::impl_desc_type::jit, false },
                conv_test_params{{1, 32, 15, 20, 20}, {3, 3, 3}, {2, 2, 2}, {0, 0, 0}, {0, 0, 0}, 64, 1, "", 2, MKLDNNPlugin::impl_desc_type::jit, false },
                conv_test_params{{1, 3, 15, 25, 20}, {3, 3, 3}, {2, 2, 2}, {0, 0, 0}, {0, 0, 0}, 64, 1, "", 2, MKLDNNPlugin::impl_desc_type::jit, false },
                conv_test_params{{1, 24, 15, 25, 20}, {3, 3, 3}, {2, 2, 2}, {0, 0, 0}, {0, 0, 0}, 64, 1, "", 2, MKLDNNPlugin::impl_desc_type::jit, false },
        /*13*/  conv_test_params{{1, 32, 15, 25, 20}, {3, 3, 3}, {2, 2, 2}, {0, 0, 0}, {0, 0, 0}, 64, 1, "", 2, MKLDNNPlugin::impl_desc_type::jit, false },
                conv_test_params{{1, 16, 30, 30, 10}, {5, 5, 5}, {1, 1, 1}, {2, 2, 2}, {2, 2, 2}, 16, 1, "", 2, MKLDNNPlugin::impl_desc_type::jit, false },
                conv_test_params{{1, 16, 30, 30, 10}, {5, 5, 5}, {1, 1, 1}, {2, 2, 2}, {2, 2, 2}, 16, 1, "", 2, MKLDNNPlugin::impl_desc_type::jit, false} ));

#ifdef USE_MKL
INSTANTIATE_TEST_CASE_P(
    MKLTestConvolution, MKLDNNGraphConvolutionTests,
    ::testing::Values(
                conv_test_params{{1, 9, 16, 32},
                                 {1, 1}, {1, 1}, {0, 0}, {0, 0}, 17, 1, "", 6, MKLDNNPlugin::impl_desc_type::gemm, false,
                                 {MKLDNNPlugin::impl_desc_type::gemm_any,
                                  MKLDNNPlugin::impl_desc_type::gemm_blas,
                                  MKLDNNPlugin::impl_desc_type::gemm_avx512,
                                  MKLDNNPlugin::impl_desc_type::gemm_avx2,
                                  MKLDNNPlugin::impl_desc_type::gemm_sse42} },
                conv_test_params{{1, 5, 15, 20, 20},
                                 {3, 3, 3}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, 64, 1, "", 2, MKLDNNPlugin::impl_desc_type::gemm_blas, false,
                                 {MKLDNNPlugin::impl_desc_type::gemm_blas} },
                conv_test_params{{1, 5, 15, 20, 20},
                                 {3, 3, 3}, {3, 2, 1}, {0, 0, 0}, {0, 0, 0}, 64, 1, "", 2, MKLDNNPlugin::impl_desc_type::gemm_blas, false,
                                 {MKLDNNPlugin::impl_desc_type::gemm_blas} },
                // conv_test_params{{1, 5, 15, 20, 20},
                //                  {3, 3, 3}, {1, 1, 1}, {2, 2, 2}, {1, 1, 1}, 64, 1, "", 2, MKLDNNPlugin::impl_desc_type::gemm_blas, false,
                //                  {MKLDNNPlugin::impl_desc_type::gemm_blas}  },
                conv_test_params{{1, 16, 30, 30, 10},
                                 {5, 5, 5}, {1, 1, 1}, {2, 2, 2}, {2, 2, 2}, 16, 1, "", 2, MKLDNNPlugin::impl_desc_type::gemm_blas, false,
                                 {MKLDNNPlugin::impl_desc_type::gemm_blas} },
                conv_test_params{{1, 4, 16, 16, 16},
                                 {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, 8, 1, "", 2, MKLDNNPlugin::impl_desc_type::gemm_blas, false,
                                 {MKLDNNPlugin::impl_desc_type::gemm_blas} } ));
#endif

INSTANTIATE_TEST_CASE_P(
    TestConvolutionDefaultPrimitivesPriority, MKLDNNGraphConvolutionTests,
    ::testing::Values(
        /*0*/   conv_test_params{{1, 9, 16, 32}, {1, 1}, {1, 1}, {0, 0}, {0, 0}, 17, 1, "same_upper", 6,
                                 MKLDNNPlugin::impl_desc_type::jit | MKLDNNPlugin::impl_desc_type::_1x1, true },
                conv_test_params{{1, 9, 32, 16}, {2, 4}, {1, 1}, {1, 1}, {0, 2}, 17, 1, "", 3, MKLDNNPlugin::impl_desc_type::jit, true },
                conv_test_params{{1, 9, 32, 16}, {2, 4}, {2, 1}, {0, 0}, {0, 0}, 17, 1, "", 3, MKLDNNPlugin::impl_desc_type::jit, true },
                conv_test_params{{1, 3, 40, 40}, {3, 3}, {1, 2}, {0, 0}, {0, 0}, 20, 1, "", 3, MKLDNNPlugin::impl_desc_type::jit, true },
                conv_test_params{{1, 1, 40, 40}, {3, 3}, {1, 2}, {0, 0}, {0, 0}, 20, 1, "", 3, MKLDNNPlugin::impl_desc_type::jit, true },
                conv_test_params{{1, 1, 32, 16}, {2, 4}, {2, 1}, {0, 0}, {0, 0}, 17, 1, "", 3, MKLDNNPlugin::impl_desc_type::jit, true },
        // 5D
        /*6*/   conv_test_params{{1, 3, 15, 25, 20}, {3, 3, 3}, {2, 2, 2}, {0, 0, 0}, {0, 0, 0}, 64, 1, "", 2, MKLDNNPlugin::impl_desc_type::jit, true },
                conv_test_params{{1, 24, 15, 25, 20}, {3, 3, 3}, {2, 2, 2}, {0, 0, 0}, {0, 0, 0}, 64, 1, "", 2, MKLDNNPlugin::impl_desc_type::jit, true },
                conv_test_params{{1, 32, 15, 25, 20}, {3, 3, 3}, {2, 2, 2}, {0, 0, 0}, {0, 0, 0}, 64, 1, "", 2, MKLDNNPlugin::impl_desc_type::jit, true },
                conv_test_params{{1, 16, 30, 30, 10}, {5, 5, 5}, {1, 1, 1}, {2, 2, 2}, {2, 2, 2}, 16, 1, "", 2, MKLDNNPlugin::impl_desc_type::jit, true } ));


class MKLDNNGraphDynBatchConvolutionTests: public MKLDNNGraphConvolutionTests {
protected:
    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            conv_test_params p = ::testing::WithParamInterface<conv_test_params>::GetParam();
            std::string model = getModel(p);
            std::vector<size_t> dims = p.dims;
            if (dims[0] < 2)
                dims[0] = 2;

            size_t blob_size = p.out_c * dims[1] / p.grp_c;
            for (auto k : p.kernel) {
                blob_size *= k;
            }
            blob_size = (blob_size + p.out_c) * sizeof(float);
            TBlob<uint8_t> *weights = new TBlob<uint8_t>({ Precision::U8, {blob_size}, Layout::C });
            weights->allocate();
            fill_data((float *) weights->buffer(), weights->size() / sizeof(float));
            TBlob<uint8_t>::Ptr weights_ptr = TBlob<uint8_t>::Ptr(weights);

            InferenceEngine::Core core;
            InferenceEngine::CNNNetwork network;
            ASSERT_NO_THROW(network = core.ReadNetwork(model, weights_ptr));

            auto implNet = dynamic_cast<details::CNNNetworkImpl *>(&((ICNNNetwork&)network));
            ASSERT_NE(nullptr, implNet) << "Failed to cast ICNNNetwork to CNNNetworkImpl";
            ResponseDesc resp;
            StatusCode sts  = implNet->setBatchSizeReshape(dims[0], &resp);
            ASSERT_EQ((int)StatusCode::OK, sts) << resp.msg;

            MKLDNNGraphTestClass graph;
            graph.CreateGraph(network);

            Layout layout = ANY;
            switch (dims.size()) {
                case 4:
                    layout = NCHW;
                    break;
                case 5:
                    layout = NCDHW;
                    break;
            }

            Blob::Ptr src = make_shared_blob<float>({ Precision::FP32, dims, layout });
            src->allocate();
            fill_data(src->buffer(), src->size());

            auto * srcPtr = dynamic_cast<TBlob<float>*>(src.get());

            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            BlobMap srcs;
            srcs.insert(std::pair<std::string, Blob::Ptr>("in1", src));

            OutputsDataMap out;
            out = network.getOutputsInfo();
            BlobMap outputBlobs;

            std::pair<std::string, DataPtr> item = *out.begin();

            TBlob<float>::Ptr output;
            output = make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            auto checkConvolution = [](const MKLDNNPlugin::MKLDNNNodePtr& node) {
                return node->getType() == MKLDNNPlugin::Convolution;
            };

            graph.checkDynBatch(srcs, outputBlobs, dims[0], dims[0], checkConvolution, MKLDNNGraphTestClass::CheckDynBatchType::Child);
            graph.checkDynBatch(srcs, outputBlobs, 1, dims[0], checkConvolution, MKLDNNGraphTestClass::CheckDynBatchType::Child);
        } catch (const details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNGraphDynBatchConvolutionTests, TestsDynBatchConvolution) {}

// TODO: rewrite to ngraph to have reshape functionality
INSTANTIATE_TEST_CASE_P(
    DISABLED_TestDynBatchConvolution, MKLDNNGraphDynBatchConvolutionTests,
    ::testing::Values(
                conv_test_params{{1, 8, 16, 32},
                                 {1, 1}, {1, 1}, {0, 0}, {0, 0}, 17, 1, "same_upper", 7, MKLDNNPlugin::impl_desc_type::jit | MKLDNNPlugin::impl_desc_type::_1x1,
                                 false, {MKLDNNPlugin::impl_desc_type::jit_avx512_winograd}},
                conv_test_params{{1, 9, 32, 16},
                                 {2, 4}, {1, 1}, {0, 0}, {0, 0}, 17, 1, "", 5, MKLDNNPlugin::impl_desc_type::jit,
                                 false, {MKLDNNPlugin::impl_desc_type::jit_avx512_winograd} },
                conv_test_params{{1, 9, 32, 16},
                                 {2, 4}, {2, 1}, {0, 0}, {0, 0}, 17, 1, "", 5, MKLDNNPlugin::impl_desc_type::jit,
                                 false, {MKLDNNPlugin::impl_desc_type::jit_avx512_winograd} },
                conv_test_params{{1, 3, 40, 40},
                                 {3, 3}, {1, 2}, {0, 0}, {0, 0}, 20, 1, "", 5, MKLDNNPlugin::impl_desc_type::jit,
                                 false, {MKLDNNPlugin::impl_desc_type::jit_avx512_winograd} },
                conv_test_params{{1, 1, 40, 40},
                                 {3, 3}, {1, 2}, {0, 0}, {0, 0}, 20, 1, "", 5, MKLDNNPlugin::impl_desc_type::jit,
                                 false, {MKLDNNPlugin::impl_desc_type::jit_avx512_winograd} },
                conv_test_params{{1, 1, 32, 16},
                                 {2, 4}, {2, 1}, {0, 0}, {0, 0}, 17, 1, "", 5, MKLDNNPlugin::impl_desc_type::jit,
                                 false, {MKLDNNPlugin::impl_desc_type::jit_avx512_winograd} },
                conv_test_params{{1, 9, 32, 16},
                                 {2, 4}, {1, 1}, {0, 0}, {0, 0}, 17, 1, "", 5, MKLDNNPlugin::impl_desc_type::ref_any,
                                 false, {MKLDNNPlugin::impl_desc_type::ref_any} } ));

#ifdef USE_MKL
INSTANTIATE_TEST_CASE_P(
    MKLTestDynBatchConvolution, MKLDNNGraphDynBatchConvolutionTests,
    ::testing::Values(
                conv_test_params{{1, 9, 16, 32},
                                 {1, 1}, {1, 1}, {0, 0}, {0, 0}, 17, 1, "", 7, MKLDNNPlugin::impl_desc_type::gemm, false,
                                 {MKLDNNPlugin::impl_desc_type::gemm_any,
                                  MKLDNNPlugin::impl_desc_type::gemm_blas,
                                  MKLDNNPlugin::impl_desc_type::gemm_avx512,
                                  MKLDNNPlugin::impl_desc_type::gemm_avx2,
                                  MKLDNNPlugin::impl_desc_type::gemm_sse42}
                }));
#endif

