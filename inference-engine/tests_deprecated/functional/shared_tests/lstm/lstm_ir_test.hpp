// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plg_test.hpp"

#include <string>
#include <vector>
#include <ngraph_functions/subgraph_builders.hpp>

// library taken from https://github.com/llohse/libnpy
#include "npy.hpp"

using namespace ::testing;
using namespace InferenceEngine;

struct ModelInfo {
    std::string dir, xml, bin;
};

class LSTM_IR_Test : public PlgTest<ModelInfo> {
protected:
    virtual void SetUp() {
        PlgTest::SetUp();
        auto p = param();
    }
};

TEST_P(LSTM_IR_Test, canParseLSTM) {
    auto fn_ptr = ngraph::builder::subgraph::makeTIwithLSTMcell();
    CNNNetwork net(fn_ptr);

    Core ie;
    auto exec_net = ie.LoadNetwork(net, device_name);
    auto inf_req = exec_net.CreateInferRequest();

    auto _load_from_npy = [&](std::string name) {
        std::replace(name.begin(), name.end(), '\\', '_');
        std::replace(name.begin(), name.end(), '/', '_');
        auto file_path = name + ".npy";

        std::ifstream npy_file(file_path);
        std::vector<unsigned long> npy_shape;
        std::vector<float> npy_data;
        if (npy_file.good())
            npy::LoadArrayFromNumpy(file_path, npy_shape, npy_data);

        return npy_data;
    };

    auto _save_to_npy = [&](std::string name,
                            const std::vector<unsigned long>& npy_shape,
                            const std::vector<float>& npy_data) {
        std::replace(name.begin(), name.end(), '\\', '_');
        std::replace(name.begin(), name.end(), '/', '_');
        auto file_path = name + ".npy";

        npy::SaveArrayAsNumpy(file_path, false, (unsigned int)(npy_shape.size()), npy_shape.data(), npy_data);
    };

    for (auto &info: net.getInputsInfo()) {
        auto blob = inf_req.GetBlob(info.first);
        auto npy = _load_from_npy(info.first);

        if (!npy.empty())
            std::copy_n(npy.data(), npy.size(), blob->buffer().as<float*>());
    }

    inf_req.Infer();

    for (auto &info : net.getOutputsInfo()) {
        auto blob = inf_req.GetBlob(info.first);
        auto npy = _load_from_npy(info.first);

        if (!npy.empty())
            TestsCommon::compare(blob->buffer().as<float*>(), npy.data(), npy.size());

        /* auto dims = blob->dims();

        std::vector<unsigned long> shape;
        for (auto d : dims) shape.push_back(d);

        std::vector<float> npy_data(blob->buffer().as<float*>(), blob->buffer().as<float*>() + blob->size());
        _save_to_npy(plugin_name + "_" + info.first, shape, npy_data); */
    }
}

static std::vector<ModelInfo> workload = {
/*  Directory             |       XML name              |   Bin name    */
{"Basic_LSTM_S/FP32", "Basic_LSTM_S"},
};
