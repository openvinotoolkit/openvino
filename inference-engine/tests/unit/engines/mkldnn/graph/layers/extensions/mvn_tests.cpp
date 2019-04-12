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
#include "ir_gen_helper.hpp"

using namespace InferenceEngine;
using namespace ::testing;
using namespace std;
using namespace mkldnn;
using namespace single_layer_tests;


struct mvn_test_params {
    // Formats: NCHW, NCDHW
    vector<size_t> dims;

    int across_channels;
    int normalize_variance;
    float eps;

    size_t num_prim_desc;
    bool isBlockedFormat;
    int selectedType;

    vector<std::function<void(MKLDNNPlugin::PrimitiveDescInfo)>> comp;
};

extern InferenceEngine::IExtensionPtr make_FakeExtensions();

template <typename data_t>
void ref_mvn(const TBlob<data_t> &src, TBlob<data_t> &dst, mvn_test_params prm) {
    const data_t *src_data = src.readOnly();
    data_t *dst_data = dst.data();
    size_t dims_size = prm.dims.size();

    size_t N = prm.dims[0];
    size_t C = prm.dims[1];
    size_t D = dims_size > 4 ? prm.dims[dims_size - 3lu] : 1lu;
    size_t H = dims_size > 3 ? prm.dims[dims_size - 2lu] : 1lu;
    size_t W = prm.dims[dims_size - 1lu];

    float eps = prm.eps;

    size_t C1 = H * W;
    size_t C2 = C1 * D;
    size_t C3 = C2 * C;

    for (size_t b = 0lu; b < N; b++) {
        size_t cb = b * C3;
        // Calculate mean value
        if (prm.across_channels) {
            double mean = 0.0;
            for (size_t c = 0lu; c < C; c++) {
                size_t cc = cb + c * C2;
                for (size_t d = 0lu; d < D; d++) {
                    size_t cd = cc + d * C1;
                    for (size_t h = 0lu; h < H; h++) {
                        size_t ch = cd + h * W;
                        for (size_t w = 0lu; w < W; w++) {
                            mean += src_data[ch + w];
                        }
                    }
                }
            }
            mean /= (double)C3;
            for (size_t c = 0lu; c < C; c++) {
                size_t cc = cb + c * C2;
                for (size_t d = 0lu; d < D; d++) {
                    size_t cd = cc + d * C1;
                    for (size_t h = 0lu; h < H; h++) {
                        size_t ch = cd + h * W;
                        for (size_t w = 0lu; w < W; w++) {
                            size_t index = ch + w;
                            dst_data[index] = src_data[index] - mean;
                        }
                    }
                }
            }
        } else {
            for (size_t c = 0lu; c < C; c++) {
                size_t cc = cb + c * C2;
                double mean = 0.0;
                for (size_t d = 0lu; d < D; d++) {
                    size_t cd = cc + d * C1;
                    for (size_t h = 0lu; h < H; h++) {
                        size_t ch = cd + h * W;
                        for (size_t w = 0lu; w < W; w++) {
                            mean += src_data[ch + w];
                        }
                    }
                }

                mean /= (double)C2;

                for (size_t d = 0lu; d < D; d++) {
                    size_t cd = cc + d * C1;
                    for (size_t h = 0lu; h < H; h++) {
                        size_t ch = cd + h * W;
                        for (size_t w = 0lu; w < W; w++) {
                            size_t index = ch + w;
                            dst_data[index] = src_data[index] - mean;
                        }
                    }
                }
            }
        }
    }

    if (prm.normalize_variance) {
        for (size_t b = 0; b < N; b++) {
            size_t cb = b * C3;
            // Calculate variances value
            if (prm.across_channels) {
                double variance = 0.f;
                for (size_t c = 0lu; c < C; c++) {
                    size_t cc = cb + c * C2;
                    for (size_t d = 0lu; d < D; d++) {
                        size_t cd = cc + d * C1;
                        for (size_t h = 0lu; h < H; h++) {
                            size_t ch = cd + h * W;
                            for (size_t w = 0lu; w < W; w++) {
                                variance += std::pow(dst_data[ch + w], 2);
                            }
                        }
                    }
                }
                variance /= C3;
                variance += eps;
                variance = std::pow(variance, 0.5f);
                for (size_t c = 0lu; c < C; c++) {
                    size_t cc = cb + c * C2;
                    for (size_t d = 0lu; d < D; d++) {
                        size_t cd = cc + d * C1;
                        for (size_t h = 0lu; h < H; h++) {
                            size_t ch = cd + h * W;
                            for (size_t w = 0lu; w < W; w++) {
                                dst_data[ch + w] /= variance;
                            }
                        }
                    }
                }
            } else {
                for (size_t c = 0lu; c < C; c++) {
                    size_t cc = cb + c * C2;
                    double variance = 0.0;
                    for (size_t d = 0lu; d < D; d++) {
                        size_t cd = cc + d * C1;
                        for (size_t h = 0lu; h < H; h++) {
                            size_t ch = cd + h * W;
                            for (size_t w = 0lu; w < W; w++) {
                                variance += std::pow(dst_data[ch + w], 2);
                            }
                        }
                    }
                    variance /= C2;
                    variance += eps;
                    variance = std::pow(variance, 0.5f);
                    for (size_t d = 0lu; d < D; d++) {
                        size_t cd = cc + d * C1;
                        for (size_t h = 0lu; h < H; h++) {
                            size_t ch = cd + h * W;
                            for (size_t w = 0lu; w < W; w++) {
                                dst_data[ch + w] /= variance;
                            }
                        }
                    }
                }
            }
        }
    }
}

class MKLDNNCPUExtMVNTests: public TestsCommon, public WithParamInterface<mvn_test_params> {
    std::string layers_t = R"V0G0N(
        <layer name="fakeLayer" id="1" type="_FL_" precision="FP32">
            <input>
                <port id="1">
                    __SRC_DIMS__
                </port>
            </input>
            <output>
                <port id="2">
                    __SRC_DIMS__
                </port>
            </output>
        </layer>
        <layer name="mvn" id="2" type="MVN" precision="FP32">
            <data across_channels="_AC_" normalize_variance="_NV_" eps="_EPS_"/>
            <input>
                <port id="3">
                    __SRC_DIMS__
                </port>
            </input>
            <output>
                <port id="4">
                    __SRC_DIMS__
                </port>
            </output>
        </layer>
)V0G0N";

    std::string edges_t = R"V0G0N(
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="3"/>
)V0G0N";

    std::string getModel(mvn_test_params p) {
        std::string model = layers_t;
        if (p.isBlockedFormat)
            REPLACE_WITH_STR(model, "_FL_", "FakeLayerBLK");
        else
            REPLACE_WITH_STR(model, "_FL_", "FakeLayerPLN");

        std::string s_dims;
        for (auto& dim : p.dims) {
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(dim) + "</dim>";
        }
	REPLACE_WITH_STR(model, "__SRC_DIMS__", s_dims);

        REPLACE_WITH_NUM(model, "_AC_", p.across_channels);
        REPLACE_WITH_NUM(model, "_NV_", p.normalize_variance);
        REPLACE_WITH_NUM(model, "_EPS_", p.eps);

        model = IRTemplateGenerator::getIRTemplate("MVN_Only", p.dims, "FP32", model, edges_t);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            mvn_test_params p = ::testing::WithParamInterface<mvn_test_params>::GetParam();
            std::string model = getModel(p);

            CNNNetReader net_reader;
            ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

            InferenceEngine::Extension cpuExt(make_so_name("cpu_extension"));
            MKLDNNPlugin::MKLDNNExtensionManager::Ptr extMgr(new MKLDNNPlugin::MKLDNNExtensionManager());
            extMgr->AddExtension(InferenceEngine::IExtensionPtr(&cpuExt, [](InferenceEngine::IExtension*){}));
            extMgr->AddExtension(make_FakeExtensions());


            MKLDNNGraphTestClass graph;
            graph.CreateGraph(net_reader.getNetwork(), extMgr);

            auto& nodes = graph.getNodes();
            nodes = graph.getNodes();

            for (auto &node : nodes) {
                if (node->getName() == "mvn") {
                    ASSERT_EQ(p.num_prim_desc, node->getSupportedPrimitiveDescriptors().size());
                    for (size_t j = 0; j < p.num_prim_desc && j < p.comp.size(); j++) {
                        p.comp.at(j)(node->getSupportedPrimitiveDescriptors().at(j));
                    }
                    ASSERT_NE(nullptr, node->getSelectedPrimitiveDescriptor());
                    ASSERT_EQ(p.selectedType,
                              node->getSelectedPrimitiveDescriptor()->getImplementationType() & p.selectedType);
                }
            }
            if (p.isBlockedFormat)
                ASSERT_EQ(6, nodes.size());
            else
                ASSERT_EQ(5, nodes.size()); // TODO: should be 4 (redudant reorder in case of both layers are inplace)

            SizeVector dims_src = p.dims;

            Layout layout = ANY;
            switch (p.dims.size()) {
                case 4:
                    layout = NCHW;
                    break;
                case 5:
                    layout = NCDHW;
                    break;
            }

            Blob::Ptr src = make_shared_blob<float, const SizeVector>(Precision::FP32, layout, dims_src);
            src->allocate();
            fill_data(src->buffer(), src->size());

            auto * srcPtr = dynamic_cast<TBlob<float>*>(src.get());

            if (srcPtr == nullptr)
                FAIL() << "Cannot cast blob to TBlob<float>.";

            BlobMap srcs;
            srcs.insert(std::pair<std::string, Blob::Ptr>("in1", src));

            OutputsDataMap out;
            out = net_reader.getNetwork().getOutputsInfo();
            BlobMap outputBlobs;

            std::pair<std::string, DataPtr> item = *out.begin();

            TBlob<float>::Ptr output;
            output = make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            outputBlobs[item.first] = output;

            graph.Infer(srcs, outputBlobs);

            TBlob<float> dst_ref(item.second->getTensorDesc());
            dst_ref.allocate();
            ref_mvn(*srcPtr, dst_ref, p);
            compare(*output, dst_ref, 0.0001f);
        } catch (const details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(MKLDNNCPUExtMVNTests, TestsMVN) {}

INSTANTIATE_TEST_CASE_P(
        TestsMVN, MKLDNNCPUExtMVNTests,
        ::testing::Values(
        /*0*/   mvn_test_params{{2, 64, 15, 15}, 0, 0, 0.00001, 2, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2,  2, 33, 65}, 0, 0, 0.00001, 2, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 15, 15}, 0, 1, 0.00001, 2, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2,  2, 33, 65}, 0, 1, 0.00001, 2, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 15, 15}, 1, 0, 0.00001, 2, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2,  2, 33, 65}, 1, 0, 0.00001, 2, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 15, 15}, 1, 1, 0.00001, 2, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2,  2, 33, 65}, 1, 1, 0.00001, 2, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 15, 15}, 0, 0, 0.00001, 2, true, MKLDNNPlugin::impl_desc_type::unknown },
        /*9*/   mvn_test_params{{2,  2, 33, 65}, 0, 0, 0.00001, 2, true, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 15, 15}, 0, 1, 0.00001, 2, true, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2,  2, 33, 65}, 0, 1, 0.00001, 2, true, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 15, 15}, 1, 0, 0.00001, 2, true, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2,  2, 33, 65}, 1, 0, 0.00001, 2, true, MKLDNNPlugin::impl_desc_type::unknown },
        /*14*/  mvn_test_params{{2,640, 15, 15}, 1, 1, 0.00001, 2, true, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2,  2, 33, 65}, 1, 1, 0.00001, 2, true, MKLDNNPlugin::impl_desc_type::unknown },

                // 5D
        /*16*/  mvn_test_params{{2, 64, 24, 32, 40}, 0, 0, 0.00001f, 2, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 24, 32, 40}, 0, 1, 0.00001f, 2, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 24, 32, 40}, 1, 0, 0.00001f, 2, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 24, 32, 40}, 1, 1, 0.00001f, 2, false, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 24, 32, 40}, 0, 0, 0.00001f, 2, true, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 24, 32, 40}, 0, 1, 0.00001f, 2, true, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{2, 64, 24, 32, 40}, 1, 0, 0.00001f, 2, true, MKLDNNPlugin::impl_desc_type::unknown },
        /*23*/  mvn_test_params{{2, 64, 24, 32, 40}, 1, 1, 0.00001f, 2, true, MKLDNNPlugin::impl_desc_type::unknown },
                mvn_test_params{{1, 64, 32, 32, 32}, 0, 1, 0.001f, 2, true, MKLDNNPlugin::impl_desc_type::unknown }
            ));
