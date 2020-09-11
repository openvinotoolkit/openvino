// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_core.hpp>
#include <cmath>

#include "tests_common.hpp"
#include "single_layer_common.hpp"

using namespace ::testing;
using namespace InferenceEngine;
using namespace std;

static inline int count(std::vector<size_t> dims, size_t start_ind, size_t end_ind) {
    size_t count = 1;
    for (size_t i = start_ind; i < end_ind; i++)
        count *= dims[i];
    return static_cast<int>(count);
}

static inline int count(std::vector<size_t> dims, size_t start_ind = 0) {
    return count(dims, start_ind, dims.size());
}

struct argMaxMinTF_test_params {
    std::string device_name;
    std::string layer_type;

    InferenceEngine::SizeVector in_dim;
    std::vector<float> in;

    int has_axis;
    int out_max_val;
    size_t top_k;
    int axis;

    InferenceEngine::SizeVector ref_dim;
    std::vector<float> ref;
};


static void ref_argmax(float *src_data, float* dst_data, argMaxMinTF_test_params p) {
    int dim, axis_dist;
    if (p.has_axis) {
        int axis_ = (p.axis < 0) ? p.axis + static_cast<int>(p.in_dim.size()) : p.axis;
        dim = static_cast<int>(p.in_dim[axis_]);
        axis_dist = count(p.in_dim, axis_) / dim;
    } else {
        dim = count(p.in_dim, 1);
        axis_dist = 1;
    }

    int num = count(p.in_dim) / dim;
    std::vector<std::pair<float, int> > src_vector(dim);

    for (int i = 0; i < num; ++i) {
        for (int j = 0; j < dim; ++j) {
            src_vector[j] = std::make_pair(
                    src_data[(i / axis_dist * dim + j) * axis_dist + i % axis_dist], j);
        }

        if (p.layer_type == "ArgMax") {
            for (int j = 0; j < p.top_k; j++) {
                for (int k = src_vector.size() - 1; k > j; k--) {
                    if (src_vector[k].first > src_vector[k - 1].first) {
                        std::pair<float, int> tmp = src_vector[k];
                        src_vector[k] = src_vector[k - 1];
                        src_vector[k - 1] = tmp;
                    }
                }
            }
        } else {
            for (int j = 0; j < p.top_k; j++) {
                for (int k = src_vector.size() - 1; k > j; k--) {
                    if (src_vector[k].first < src_vector[k - 1].first) {
                        std::pair<float, int> tmp = src_vector[k];
                        src_vector[k] = src_vector[k - 1];
                        src_vector[k - 1] = tmp;
                    }
                }
            }
        }
        for (int j = 0; j < p.top_k; ++j) {
            if (p.out_max_val) {
                if (p.has_axis) {
                    // Produces max_val per axis
                    dst_data[(i / axis_dist * p.top_k + j) * axis_dist + i % axis_dist] = src_vector[j].first;
                } else {
                    // Produces max_ind and max_val
                    dst_data[2 * i * p.top_k + j] = src_vector[j].second;
                    dst_data[2 * i * p.top_k + p.top_k + j] = src_vector[j].first;
                }
            } else {
                // Produces max_ind per axis
                dst_data[(i / axis_dist * p.top_k + j) * axis_dist + i % axis_dist] = src_vector[j].second;
            }
        }
    }
}

class ArgMaxMinTFTests : public TestsCommon, public WithParamInterface<argMaxMinTF_test_params> {
    std::string model_t = R"V0G0N(
<net Name="ArgMin_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="input" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    _IDIM_
                </port>
            </output>
        </layer>
        <layer name="ArgMinTest" id="2" type="_LAYER_TYPE_" precision="FP32">
            <data top_k="_TOP_K_" out_max_val="_OUT_MAX_VAL_" _AXIS_/>
            <input>
                <port id="1">
                    _IDIM_
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

    std::string getModel(argMaxMinTF_test_params p) {
        std::string model = model_t;
        std::string inDim;
        std::string out;

        for (auto& dim : p.in_dim) {
            inDim += "<dim>";
            inDim += std::to_string(dim) + "</dim>\n";
        }

        for (auto& dst : p.ref_dim) {
            out += "<dim>";
            out += std::to_string(dst) + "</dim>\n";
        }

        REPLACE_WITH_STR(model, "_LAYER_TYPE_", p.layer_type);
        REPLACE_WITH_STR(model, "_IDIM_", inDim);
        REPLACE_WITH_NUM(model, "_TOP_K_", p.top_k);
        REPLACE_WITH_NUM(model, "_OUT_MAX_VAL_", p.out_max_val);

        std::string axis;
        if (p.has_axis)
            axis += "axis=\"" + std::to_string(p.axis) + "\"";

        REPLACE_WITH_STR(model, "_AXIS_", axis);
        REPLACE_WITH_STR(model, "_OUT_", out);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            TestsCommon::SetUp();
            argMaxMinTF_test_params p = ::testing::WithParamInterface<argMaxMinTF_test_params>::GetParam();
            std::string model = getModel(p);

            Core ie;
            CNNNetwork net = ie.ReadNetwork(model, Blob::CPtr());
            InputsDataMap in_info_map = net.getInputsInfo();
            OutputsDataMap out_info_map = net.getOutputsInfo();

            ExecutableNetwork executable_network = ie.LoadNetwork(net, p.device_name);
            InferRequest inferRequest = executable_network.CreateInferRequest();

            Blob::Ptr inputBlob = inferRequest.GetBlob(in_info_map.begin()->first);
            float* inputData = inputBlob->buffer().as<float*>();
            memcpy(inputData, &p.in[0], sizeof(float)*p.in.size());

            TBlob<float> dst_ref(out_info_map.begin()->second->getTensorDesc());
            dst_ref.allocate();
            ref_argmax(inputData, dst_ref.data(), p);

            inferRequest.Infer();

            Blob::Ptr outputBlob = inferRequest.GetBlob(out_info_map.begin()->first);
            //  Check results
            compare(outputBlob->buffer().as<float*>(), dst_ref.buffer().as<float*>(), outputBlob->size());
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(ArgMaxMinTFTests, TestsArgMaxMin) {}
