// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <cmath>

#include "tests_common.hpp"
#include "single_layer_common.hpp"
#include <ie_core.hpp>

using namespace ::testing;
using namespace InferenceEngine;
using namespace std;

struct one_hot_base_params {
    std::vector<size_t> in;
    std::vector<size_t> out;
    int axis;
    unsigned int depth;
    float on, off;
};

struct one_hot_test_params : one_hot_base_params {
    std::string device_name;

    one_hot_test_params(std::string name, one_hot_base_params params) :
            one_hot_base_params(params), device_name(name) {}
};

class OneHotOnlyTestShared: public TestsCommon,
                        public WithParamInterface<one_hot_test_params> {

    std::string model_t = R"V0G0N(
<net name="OneHot_Only" version="2" precision="FP32" batch="1">
    <layers>
        <layer id="1" name="input" precision="FP32" type="Input">
            <output>
                <port id="0">
                    _IN_
                </port>
            </output>
        </layer>
        <layer id="2" name="OneHot1" type="OneHot" precision="FP32">

            <data depth="_DEPTH_" axis="_AXIS_"/>

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
    <edges>l
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(one_hot_test_params p) {
        std::string model = model_t;

        std::string in_shape;
        std::string out_shape;

        for (size_t i = 0; i < p.in.size(); i++) {
            in_shape += "<dim>";
            in_shape += std::to_string(p.in[i]) + "</dim>\n";
        }
        for (size_t i = 0; i < p.out.size(); i++) {
            out_shape += "<dim>";
            out_shape += std::to_string(p.out[i]) + "</dim>\n";
        }


        REPLACE_WITH_STR(model, "_IN_", in_shape);
        REPLACE_WITH_STR(model, "_OUT_", out_shape);

        REPLACE_WITH_NUM(model, "_AXIS_", p.axis);
        REPLACE_WITH_NUM(model, "_DEPTH_", p.depth);

        return model;
    }

    void ref_one_hot_4d(Blob &src, Blob &dst, one_hot_test_params p)
    {
        float *src_ptr = src.buffer().as<float*>();
        std::size_t src_size = src.size();
        float *dst_ptr = dst.buffer().as<float*>();
        std::size_t dst_size = dst.size();

        int out_n = (p.out.size() >= 1) ? p.out[0] : 1;
        int out_c = (p.out.size() >= 2) ? p.out[1] : 1;
        int out_d = (p.out.size() == 5) ? p.out[2] : 1;
        int out_h = (p.out.size() >= 3 && p.out.size() < 5) ? p.out[2] : (p.out.size() == 5) ? p.out[3] : 1;
        int out_w = (p.out.size() >= 4 && p.out.size() < 5) ? p.out[3] : (p.out.size() == 5) ? p.out[4] : 1;

        int hot_axis = (p.axis == - 1) ? p.in.size() : p.axis;

        for (int ob = 0; ob < out_n; ob++) {
            for (int oc = 0; oc < out_c; oc++) {
                for (int od = 0; od < out_d; od++) {
                    for (int oh = 0; oh < out_h; oh++) {
                        for (int ow = 0; ow < out_w; ow++) {
                            std::size_t dst_offset = ow + out_w * oh + out_w * out_h * od + out_w * out_h * out_d * oc + out_w * out_h * out_d * out_c * ob;
                            std::size_t src_offset = 0;

                            std::vector<int> out_dims = {ob, oc, oh, ow};
                            if (p.out.size() == 5)
                                out_dims.insert(out_dims.begin() + 2, od);
                            std::vector<int> in_dims(out_dims.begin(), out_dims.end());
                            in_dims.erase(in_dims.begin() + hot_axis);

                            for (int i = 0; i < p.in.size(); i++) {
                                int mul = 1;
                                if (i == p.in.size() - 1) {
                                    src_offset += in_dims[i];
                                    break;
                                }
                                for (int j = i; j < p.in.size(); j++) {
                                    if (j == i)
                                        mul *= in_dims[j];
                                    else
                                        mul *= p.in[j];
                                }
                                src_offset += mul;
                            }

                            if (out_dims[hot_axis] == src_ptr[src_offset])
                                dst_ptr[dst_offset] = p.on;
                            else
                                dst_ptr[dst_offset] = p.off;
                        }
                    }
                }
            }
        }
    }
protected:
    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            one_hot_test_params p = ::testing::WithParamInterface<one_hot_test_params>::GetParam();
            std::string model = getModel(p);

            Core ie;
            CNNNetwork net = ie.ReadNetwork(model, Blob::CPtr());
            ExecutableNetwork executable_network = ie.LoadNetwork(net, p.device_name);
            InferRequest inferRequest = executable_network.CreateInferRequest();

            // Output Data
            OutputsDataMap out = net.getOutputsInfo();
            std::pair<std::string, DataPtr> item = *out.begin();

            TBlob<float>::Ptr output;
            output = make_shared_blob<float>(item.second->getTensorDesc());
            output->allocate();
            inferRequest.SetBlob(item.first, output);

            // Output Reference
            TBlob<float> dst_ref(item.second->getTensorDesc());
            dst_ref.allocate();

            Blob::Ptr src;
            src = make_shared_blob<float>({ Precision::FP32, p.in, TensorDesc::getLayoutByDims(p.in) });
            src->allocate();
            float* s = src->buffer().as<float*>();
            for (int i = 0; i < src->size(); ++i)
                s[i] = -1;
            s[0] = 1;
            s[1] = 1;
            inferRequest.SetBlob("input", src);

            inferRequest.Infer();

            // Check results
            ref_one_hot_4d(*src, dst_ref, p);

            compare(*output, dst_ref);
        } catch (const details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

#define case_2d_0 one_hot_base_params({{3}, {3, 6},-1, 6, 1.0f, 0.0f })
#define case_2d_1 one_hot_base_params({{3}, {6, 3}, 0, 6, 1.0f, 0.0f })
#define case_2d_2 one_hot_base_params({{3}, {3, 6}, 1, 6, 1.0f, 0.0f })
#define case_3d_0 one_hot_base_params({{3, 2}, {3, 2, 4},-1, 4, 1.0f, 0.0f })
#define case_3d_1 one_hot_base_params({{3, 2}, {4, 3, 2}, 0, 4, 1.0f, 0.0f })
#define case_3d_2 one_hot_base_params({{3, 2}, {3, 4, 2}, 1, 4, 1.0f, 0.0f })
#define case_4d_0 one_hot_base_params({ {1, 3, 2}, {1, 3, 2, 4},-1, 4, 1.0f, 0.0f })
#define case_4d_1 one_hot_base_params({ {1, 3, 2}, {4, 1, 3, 2}, 0, 4, 1.0f, 0.0f })
#define case_4d_2 one_hot_base_params({ {1, 3, 2}, {1, 4, 3, 2}, 1, 4, 1.0f, 0.0f })
#define case_4d_3 one_hot_base_params({ {1, 3, 2}, {1, 3, 4, 2}, 2, 4, 1.0f, 0.0f })
#define case_5d_0 one_hot_base_params({ {1, 3, 2, 3}, {4, 1, 3, 2, 3}, 0, 4, 1.0f, 0.0f })
#define case_5d_1 one_hot_base_params({ {1, 3, 2, 3}, {1, 4, 3, 2, 3}, 1, 4, 1.0f, 0.0f })
#define case_5d_2 one_hot_base_params({ {1, 3, 2, 3}, {1, 3, 4, 2, 3}, 2, 4, 1.0f, 0.0f })
#define case_5d_3 one_hot_base_params({ {1, 3, 2, 3}, {1, 3, 2, 4, 3}, 3, 4, 1.0f, 0.0f })
#define case_5d_4 one_hot_base_params({ {1, 3, 2, 3}, {1, 3, 2, 3, 4}, 4, 4, 1.0f, 0.0f })

TEST_P(OneHotOnlyTestShared, TestsOneHot) {}
