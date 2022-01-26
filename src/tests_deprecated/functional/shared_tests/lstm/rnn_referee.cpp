// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rnn_referee.hpp"

#include <cmath>
#include <vector>
#include <string>

using namespace InferenceEngine;
using namespace std::placeholders;
using std::vector;

class RNN_ReferBase : public RNN_Referee {
protected:
    RNN_ReferBase(float clip, size_t D, size_t S, size_t G, size_t Gb, size_t ST_N)
            : clip(clip), D(D), S(S), G(G), Gb(Gb), state_num(ST_N) {}

    const size_t D, S, G, Gb;
    const size_t state_num;
    const float clip;

    vector<float> W, B;

    vector<Filler> _d_filler;
    vector<Checker> _d_checker;

    const vector<Filler>& getDataFillers()  override { return _d_filler;  }
    const vector<Checker>& getDataChecker() override { return _d_checker; }
    size_t wSize()                          override { return G*S*(S+D);  }
    size_t bSize()                          override { return Gb*S;        }
    size_t stateNum()                       override { return state_num;  }

    using Act = std::function<float(const float)>;

    static float _clip (const float x, const float clip) {
        return std::min(std::max(x, -clip), clip);
    }

    static Act clip_before(Act act, const float clip) {
        return [=] (const float x) {
            return act(_clip(x, clip));
        };
    }

    Act act(ActivationDesc act) {
        float alpha = act.alpha;
        Act res;
        if (act.alg == "sigmoid")
            res = [=] (const float x) { return 1 / (1 + std::exp(-x)); };
        else if (act.alg == "tanh")
            res = [=] (const float x) { return std::tanh(x); };
        else if (act.alg == "relu")
            res = [=] (const float x) { return (x > 0) ? x : alpha*x; };
        else
            IE_THROW() << "Unknown activation type " << act.alg;
        return res;
    }


public:
    void wFiller(Blob::Ptr blob) override {
        IE_ASSERT(blob->size() == wSize());
        auto ptr = blob->buffer().as<float*>();

        for (int g = 0; g < G; g++)
        for (int s = 0; s < S; s++) {
            for (int i = 0; i < D; i++) *ptr++ = W[g] / D;
            for (int i = 0; i < S; i++) *ptr++ = W[g] / S;
        }
    }

    void bFiller(Blob::Ptr blob) override {
        IE_ASSERT(blob->size() == bSize());
        auto ptr = blob->buffer().as<float*>();

        for (int g = 0; g < Gb; g++)
        for (int s = 0; s < S; s++) *ptr++ = B[g];
    }
};

#define Vals(_name) std::vector<float> _name(T+1)

class LSTMCell_Refer : public RNN_ReferBase {
public:
    LSTMCell_Refer(CellDesc cell, size_t N, size_t T, size_t D, size_t S) : RNN_ReferBase(cell.clip, D, S, 4, 4, 2) {
        // Some random values in range [0,1]
        const float H0 = 0.3, C0 = 0.77;

        const float Wf = 0.1, Bf = 0.35;
        const float Wi = 0.2, Bi = 0.25;
        const float Wc = 0.5, Bc = 0.15;
        const float Wo = 0.7, Bo = 0.05;

        auto _f = act(cell.acts[0]);
        auto _g = act(cell.acts[1]);
        auto _h = act(cell.acts[2]);

        if (clip > 0.0f) {
            _f = clip_before(_f, clip);
            _g = clip_before(_g, clip);
        }

        Vals(f); Vals(i); Vals(c); Vals(o);
        Vals(X); Vals(H); Vals(C);

        H[0] = H0;
        C[0] = C0;

        for (int t = 1; t < T+1; t++) {
            X[t] = t;
            f[t] = _f(Wf*(H[t-1] + X[t]) + Bf);
            i[t] = _f(Wi*(H[t-1] + X[t]) + Bi);
            c[t] = _g(Wc*(H[t-1] + X[t]) + Bc);
            o[t] = _f(Wo*(H[t-1] + X[t]) + Bo);

            C[t] = f[t] * C[t-1] + i[t] * c[t];
            H[t] = o[t] * _h(C[t]);
        }

        W = {Wf, Wi, Wc, Wo};
        B = {Bf, Bi, Bc, Bo};

        X.erase(X.begin());  // remove first element (unused zero element)
        H.erase(H.begin());
        C.erase(C.begin());

        _d_filler.resize(3);
        _d_filler[0] = std::bind(vector_filler, _1, SizeVector {N,T,D}, X, 1);
        _d_filler[1] = std::bind(scalar_filler, _1, SizeVector {N,S}, H0);
        _d_filler[2] = std::bind(scalar_filler, _1, SizeVector {N,S}, C0);

        _d_checker.resize(3);
        _d_checker[0] = std::bind(vector_checker, _1, SizeVector {N,T,S}, H, 1);
        _d_checker[1] = std::bind(scalar_checker, _1, SizeVector {N,S}  , H[T-1]);
        _d_checker[2] = std::bind(scalar_checker, _1, SizeVector {N,S}  , C[T-1]);
    }
};

class GRUCell_Refer : public RNN_ReferBase {
public:
    GRUCell_Refer(CellDesc cell, size_t N, size_t T, size_t D, size_t S) : RNN_ReferBase(cell.clip, D, S, 3, 3, 1) {
        // Some random values in range [0,1]
        const float H0 = 0.3;

        const float Wz = 0.1, Bz = 0.35;
        const float Wr = 0.2, Br = 0.25;
        const float Wh = 0.5, Bh = 0.15;

        auto _f = act(cell.acts[0]);
        auto _g = act(cell.acts[1]);

        if (clip > 0.0f) {
            _f = clip_before(_f, clip);
            _g = clip_before(_g, clip);
        }

        Vals(z); Vals(r); Vals(h);
        Vals(X); Vals(H);

        H[0] = H0;

        for (int t = 1; t < T+1; t++) {
            X[t] = t;
            z[t] = _f(Wz*(H[t-1] + X[t]) + Bz);
            r[t] = _f(Wr*(H[t-1] + X[t]) + Br);
            h[t] = _g(Wh*(H[t-1]*r[t] + X[t]) + Bh);
            H[t] = (1 - z[t])*h[t] + z[t]*H[t-1];
        }

        W = {Wz, Wr, Wh};
        B = {Bz, Br, Bh};

        X.erase(X.begin());
        H.erase(H.begin());

        _d_filler.resize(2);
        _d_filler[0] = std::bind(vector_filler, _1, SizeVector {N,T,D}, X, 1);
        _d_filler[1] = std::bind(scalar_filler, _1, SizeVector {N,S}  , H0);

        _d_checker.resize(2);
        _d_checker[0] = std::bind(vector_checker, _1, SizeVector {N,T,S}, H, 1);
        _d_checker[1] = std::bind(scalar_checker, _1, SizeVector {N,S}  , H[T-1]);
    }
};


class GRUlbrCell_Refer : public RNN_ReferBase {
public:
    GRUlbrCell_Refer(CellDesc cell, size_t N, size_t T, size_t D, size_t S) : RNN_ReferBase(cell.clip, D, S, 3, 4, 1) {
        // Some random values in range [0,1]
        const float H0 = 0.3;

        const float Wz = 0.1, Bz = 0.35;
        const float Wr = 0.2, Br = 0.25;
        const float Wh = 0.5, Bh = 0.15, Bhr = 0.33;

        auto _f = act(cell.acts[0]);
        auto _g = act(cell.acts[1]);

        if (clip > 0.0f) {
            _f = clip_before(_f, clip);
            _g = clip_before(_g, clip);
        }

        Vals(z); Vals(r); Vals(h);
        Vals(X); Vals(H);

        H[0] = H0;

        for (int t = 1; t < T+1; t++) {
            X[t] = 0.1 * t;
            z[t] = _f(Wz*(H[t-1] + X[t]) + Bz);
            r[t] = _f(Wr*(H[t-1] + X[t]) + Br);
            h[t] = _g(Wh*X[t] + r[t]*(Wh*H[t-1] + Bhr) + Bh);
            H[t] = (1 - z[t])*h[t] + z[t]*H[t-1];
        }

        W = {Wz, Wr, Wh};
        B = {Bz, Br, Bh, Bhr};

        X.erase(X.begin());
        H.erase(H.begin());

        _d_filler.resize(2);
        _d_filler[0] = std::bind(vector_filler, _1, SizeVector {N,T,D}, X, 1);
        _d_filler[1] = std::bind(scalar_filler, _1, SizeVector {N,S}  , H0);

        _d_checker.resize(2);
        _d_checker[0] = std::bind(vector_checker, _1, SizeVector {N,T,S}, H, 1);
        _d_checker[1] = std::bind(scalar_checker, _1, SizeVector {N,S}  , H[T-1]);
    }
};

class RNNCell_Refer : public RNN_ReferBase {
public:
    RNNCell_Refer(CellDesc cell, size_t N, size_t T, size_t D, size_t S) : RNN_ReferBase(cell.clip, D, S, 1, 1, 1) {
        // Some random values in range [0,1]
        const float H0 = 0.3;

        const float Wh = 0.5, Bh = 0.15;

        auto _f = act(cell.acts[0]);
        if (clip > 0.0f)
            _f = clip_before(_f, clip);

        Vals(X); Vals(H);

        H[0] = H0;

        for (int t = 1; t < T+1; t++) {
            X[t] = t;
            H[t] = _f(Wh*(H[t-1] +  X[t]) + Bh);
        }

        W = {Wh};
        B = {Bh};

        X.erase(X.begin());
        H.erase(H.begin());

        _d_filler.resize(2);
        _d_filler[0] = std::bind(vector_filler, _1, SizeVector {N,T,D}, X, 1);
        _d_filler[1] = std::bind(scalar_filler, _1, SizeVector {N,S}  , H0);

        _d_checker.resize(2);
        _d_checker[0] = std::bind(vector_checker, _1, SizeVector {N,T,S}, H, 1);
        _d_checker[1] = std::bind(scalar_checker, _1, SizeVector {N,S}  , H[T-1]);
    }
};

std::shared_ptr<RNN_Referee> RNN_Referee::create_referee(CellDesc cell, size_t N, size_t T, size_t D, size_t S) {
    std::shared_ptr<RNN_Referee> res;
    switch (cell.type) {
        case LSTM:
            res = std::shared_ptr<RNN_Referee>(new LSTMCell_Refer(cell, N, T, D, S));
            break;
        case GRU:
            res = std::shared_ptr<RNN_Referee>(new GRUCell_Refer(cell, N, T, D, S));
            break;
        case GRU_lbr:
            res = std::shared_ptr<RNN_Referee>(new GRUlbrCell_Refer(cell, N, T, D, S));
            break;
        case RNN:
            res = std::shared_ptr<RNN_Referee>(new RNNCell_Refer(cell, N, T, D, S));
            break;
    }
    return res;
};
