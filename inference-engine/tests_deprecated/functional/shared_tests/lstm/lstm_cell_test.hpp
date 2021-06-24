// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plg_test.hpp"
#include "xml_net_builder.hpp"

#include <cmath>
#include <vector>
#include <string>

using namespace InferenceEngine;
using std::map;
using std::pair;
using std::vector;
using std::string;

static const size_t DataSize = 10;  // Data size
static const size_t T = 1;   // Sequence length (for single LSTM cell is 1)
static const size_t StateSize = 5;   // State size
static const size_t G = 4;   // Number of gate

class LSTMCellNet : CommonTestUtils::V2NetBuilder {
public:
    LSTMCellNet(size_t N, size_t S, size_t D): CommonTestUtils::V2NetBuilder(buildNetworkWithOneInput("LSTM_Cell_Net"
            , { N,D }, "FP32")) {
        const size_t wSz = S*G*(D+S);
        const size_t bSz = S*G;
        const size_t wSz_b = wSz * sizeof(float);
        const size_t bSz_b = bSz * sizeof(float);

        weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, SizeVector{ (wSz_b + bSz_b) }, Layout::C));
        weights->allocate();

        auto ptr = weights->buffer().as<float*>();
        w_blob = make_shared_blob<float>(TensorDesc(Precision::FP32, SizeVector{wSz}, 
            TensorDesc::getLayoutByDims(SizeVector{wSz})), ptr);
        b_blob = make_shared_blob<float>(TensorDesc(Precision::FP32, SizeVector{bSz}, 
            TensorDesc::getLayoutByDims(SizeVector{bSz})), ptr + wSz);

        // input layer #1 and #2
        addInputLayer("FP32", { N,S });
        addInputLayer("FP32", { N,S });

        // layer #3
        map<string, string> lstm_p {{"hidden_size", std::to_string(S)}};
        addLayer("LSTMCell", "FP32", &lstm_p,
                { {{ N,D }, { N,S }, { N,S }},
                  {{ N,S }, { N,S }} },
                wSz_b, bSz_b);

        vector<pair<string, string>> edges = {
                {"0,0", "3,3"},
                {"1,1", "3,4"},
                {"2,2", "3,5"}
        };
        model = finish(&edges);
    }

    using Filler = std::function<void(Blob::Ptr)>;

    CNNNetwork net(Filler w_filler = nullptr, Filler b_filler = nullptr) {
        w_filler(w_blob);
        b_filler(b_blob);

        Core ie;
        return ie.ReadNetwork(model, weights);
    }

private:
    string model;
    TBlob<uint8_t>::Ptr weights;
    Blob::Ptr w_blob;
    Blob::Ptr b_blob;
};

static inline bool cmp_near(float res, float ref, float eq_threshold) {
    constexpr float eps = 1e-5;

    if (eq_threshold != eps) {
        return std::abs(res-ref) < eq_threshold;
    }

    auto ref_abs = std::abs(ref);
    if (ref_abs > eps)
        return std::abs(res-ref)/ref_abs < eps;
    else
        return std::abs(res-ref) < eps;
}

/**********************************************/
/***     Test Body       **********************/
/**********************************************/

struct lstm_cell_param {
    size_t N;    // Batch size
    size_t S;    // State size
    size_t D;    // Data  size
};

class LSTMCellTestBase : public PlgTest<lstm_cell_param> {
 public:
    void runSingleLSTMTest(const std::map<std::string, std::string> & config = {},
                           float eq_threshold = 1e-5) {
        auto p = param();
        const size_t N = p.N;
        const size_t S = p.S;
        const size_t D = p.D;

        /* Broadcast value through tensors */
        const float H0 = 0.3, C0 = 0.77;

        const float Wf = 0.1, Bf = 0.35;
        const float Wi = 0.2, Bi = 0.25;
        const float Wc = 0.5, Bc = 0.15;
        const float Wo = 0.7, Bo = 0.05;

        using Vals = float[T+1];
        Vals f, i, c, o, C, H, X;

        auto _f = [](float x) { return 1/(1 + std::exp(-x)); };  // sigmoid
        auto _h = [](float x) { return std::tanh(x); };          // tanh

        H[0] = H0; C[0] = C0;

        for (int t = 1; t < T+1; t++) {  // t=0 - initial state. So time index starts from 1.
            X[t] = t;
            f[t] = _f(Wf*(H[t-1] + X[t]) + Bf);
            i[t] = _f(Wi*(H[t-1] + X[t]) + Bi);
            c[t] = _h(Wc*(H[t-1] + X[t]) + Bc);
            o[t] = _f(Wo*(H[t-1] + X[t]) + Bo);

            C[t] = f[t] * C[t-1] + i[t] * c[t];
            H[t] = o[t] * _h(C[t]);
        }

        /********  Weight and Input blob filler *****************/

        auto w_filler = [=](Blob::Ptr blob) {
            assert(blob->size() == G*S*(S+D));
            auto ptr = blob->buffer().as<float*>();

            float W[] = {Wf, Wi, Wc, Wo};
            for (int g = 0; g < G; g++)
                for (int s = 0; s < S; s++) {
                    for (int i = 0; i < D; i++) *ptr++ = W[g] / D;
                    for (int i = 0; i < S; i++) *ptr++ = W[g] / S;
                }
        };

        auto b_filler = [=](Blob::Ptr blob) {
            assert(blob->size() == G*S);
            auto ptr = blob->buffer().as<float*>();

            float B[] = {Bf, Bi, Bc, Bo};
            for (int g = 0; g < G; g++)
                for (int s = 0; s < S; s++) *ptr++ = B[g];
        };

        auto stat_filler = [=](Blob::Ptr blob, float val) {
            assert(blob->size() == N*S);
            auto ptr = blob->buffer().as<float*>();

            for (size_t n = 0; n < N; n++)
                for (size_t s = 0; s < S; s++) *ptr++ = val;
        };

        auto data_filler = [&](Blob::Ptr blob) {
            assert(blob->size() == N*T*D);
            auto ptr = blob->buffer().as<float*>();

            for (size_t n = 0; n < N; n++)
                for (size_t d = 0; d < D; d++) *ptr++ = X[1];
        };

        /*****  Output blob checkers  ************************/
        auto stat_checker = [=](Blob::Ptr blob, float val) {
            assert(blob->size() == N*S);
            auto ptr = blob->buffer().as<float*>();

            bool passed = true;
            float maxDiff = 0;
            for (size_t n = 0; n < N; n++)
                for (size_t s = 0; s < S; s++) {

                    if (!cmp_near(*ptr, val, eq_threshold)) {
                        printf("float eq %zux%zu fail: %f : %f\n", n, s, *ptr, val);
                        passed = false;
                    }
                    maxDiff = std::max(std::abs(*ptr - val), maxDiff);
                    ptr++;
                }
            if (eq_threshold != 1e-5) {
                printf("max diff= %.6f\n", maxDiff);
            }
            return passed;
        };

        /************ Test Body  *****************************/

        LSTMCellNet topology(N, S, D);
        auto net = topology.net(w_filler, b_filler);

        Core ie;
        auto execNet = ie.LoadNetwork(net, device_name, config);
        auto req = execNet.CreateInferRequest();

        auto in_data    = req.GetBlob("Input0");
        auto in_h_state = req.GetBlob("Input1");
        auto in_c_state = req.GetBlob("Input2");

        data_filler(in_data);
        stat_filler(in_h_state, H0);
        stat_filler(in_c_state, C0);

        req.Infer();

        auto out_h_state = req.GetBlob("LSTMCell3.0");
        auto out_c_state = req.GetBlob("LSTMCell3.1");

        EXPECT_TRUE(stat_checker(out_h_state, H[T]));
        EXPECT_TRUE(stat_checker(out_c_state, C[T]));
    }
};

using LSTMCellTest  = LSTMCellTestBase;

TEST_P(LSTMCellTest, SingleLSTM) {
    runSingleLSTMTest();
}


static const lstm_cell_param workload[] = {{1, StateSize, DataSize}, {2, StateSize, DataSize}};
