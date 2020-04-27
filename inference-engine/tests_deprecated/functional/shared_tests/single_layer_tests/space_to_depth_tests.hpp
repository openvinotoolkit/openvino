// Copyright (C) 2020 Intel Corporation
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


struct space_to_depth_test_params {
    std::string device_name;
    std::string inPrecision;
    InferenceEngine::SizeVector in_dim;
    std::string depth_mode;
    size_t block_size;
    InferenceEngine::SizeVector ref_dim;
};

template<typename data_t>
void ref_spaceToDepth(const std::vector<Blob::Ptr> &srcs, std::vector<Blob::Ptr> &dsts, space_to_depth_test_params& prm) {
    assert(dsts.size() == 1);

    data_t *dst_data = dsts[0]->buffer().as<data_t*>();
    const data_t *src_data = srcs[0]->buffer().as<data_t*>();

    size_t feature_in = prm.in_dim[1];
    size_t y_in = prm.in_dim[2];
    size_t x_in = prm.in_dim[3];

    size_t batch_out = prm.ref_dim[0];
    size_t feature_out = prm.ref_dim[1];
    size_t y_out = prm.ref_dim[2];
    size_t x_out = prm.ref_dim[3];

    if (prm.depth_mode != "depth_first" && prm.depth_mode != "blocks_first")
        FAIL() << " Invalid mode for spaceToDepth: must be \"blocks_first\" or \"depth_first\" only";

    if (prm.block_size < 1)
        FAIL() << " Invalid block size number: must be greater than or equal to 1";

    if (y_in % prm.block_size != 0 || x_in % prm.block_size != 0)
        FAIL () << " Invalid sizes of spatials x, y: must be divisible by block size";

    for (size_t batch = 0; batch < batch_out; ++batch) {
        for (size_t feature = 0; feature < feature_out; ++feature) {
            size_t offset_in, f_in;
            if (prm.depth_mode == "blocks_first") {
                offset_in = feature / feature_in;
                f_in = feature % feature_in;
            } else {
                offset_in = feature % (prm.block_size * prm.block_size);
                f_in = feature / (prm.block_size * prm.block_size);
            }
            for (size_t y = 0; y < y_out; ++y) {
                size_t input_y = (y * prm.block_size) + (offset_in / prm.block_size);
                for (size_t x = 0; x < x_out; ++x) {
                    size_t input_x = (x * prm.block_size) + (offset_in % prm.block_size);
                    size_t f_in_offset = (input_y * x_in) + input_x;
                    size_t input_index = (batch * feature_in * y_in * x_in) + (f_in * y_in * x_in) + f_in_offset;
                    size_t output_index = (batch * feature_out * y_out * x_out) + (feature * y_out * x_out) + (y * x_out) + x;
                    dst_data[output_index] = src_data[input_index];
                }
            }
        }
    }
}

class SpaceToDepthTests : public TestsCommon, public WithParamInterface<space_to_depth_test_params> {
    std::string model_t = R"V0G0N(
<net Name="Space2depth_net" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="Input0" type="Input" precision="_IPRS_" id="1">
            <output>
                <port id="1">
                    _IDIM_
                </port>
            </output>
        </layer>
        <layer name="SpaceToDepth" id="3" type="SpaceToDepth" precision="FP32">
            <data block_size="_BS_" depth_mode="_DM_"/>
            <input>
                <port id="1">
                    _IDIM_
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
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
    </edges>
</net>
)V0G0N";

    std::string getModel(space_to_depth_test_params p) {
        std::string model = model_t;
        std::string inIdx;
        std::string inDict;
        std::string out;

        for (auto& dct : p.in_dim) {
            inDict += "<dim>";
            inDict += std::to_string(dct) + "</dim>\n";
        }

        for (auto& dst : p.ref_dim) {
            out += "<dim>";
            out += std::to_string(dst) + "</dim>\n";
        }

        REPLACE_WITH_STR(model, "_IPRS_", p.inPrecision);
        REPLACE_WITH_STR(model, "_IDIM_", inDict);
        REPLACE_WITH_STR(model, "_DM_", p.depth_mode);
        REPLACE_WITH_NUM(model, "_BS_", p.block_size);
        REPLACE_WITH_STR(model, "_OUT_", out);

        return model;
    }

protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        try {
            space_to_depth_test_params p = ::testing::WithParamInterface<space_to_depth_test_params>::GetParam();
            std::string model = getModel(p);

            Core ie;
            CNNNetwork net = ie.ReadNetwork(model, Blob::CPtr());
            ExecutableNetwork executable_network = ie.LoadNetwork(net, p.device_name);
            InferRequest inferRequest = executable_network.CreateInferRequest();

            std::vector<Blob::Ptr> srcs_vec;
            std::vector<Blob::Ptr> dsts_vec;
            std::vector<Blob::Ptr> out_vec;

            InputsDataMap in_info_map = net.getInputsInfo();
            for (auto info : in_info_map) {
                Blob::Ptr blob = make_shared_blob<float>({Precision::FP32, info.second->getTensorDesc().getDims(), NCHW});
                blob->allocate();
                fill_data_dbgval(blob->buffer().as<float*>(), blob->size());
                inferRequest.SetBlob(info.first, blob);
                srcs_vec.push_back(blob);
            }

            OutputsDataMap out_info_map = net.getOutputsInfo();
            for (auto info : out_info_map) {
                Blob::Ptr blob = make_shared_blob<float>({Precision::FP32, info.second->getTensorDesc().getDims(), NCHW});
                blob->allocate();
                inferRequest.SetBlob(info.first, blob);
                out_vec.push_back(blob);

                Blob::Ptr blob_ref = make_shared_blob<float>({Precision::FP32, info.second->getTensorDesc().getDims(), NCHW});
                blob_ref->allocate();
                dsts_vec.push_back(blob_ref);
            }

            ref_spaceToDepth<float>(srcs_vec, dsts_vec, p);

            inferRequest.Infer();

            compare(*out_vec[0], *dsts_vec[0]);
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(SpaceToDepthTests, TestsSpaceToDepth) {}
