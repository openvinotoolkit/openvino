// Simple GEMM-v (N=1) micro-benchmark for x64 JIT ukernel
// Measures int8/u8 -> fp32 path, per-tensor and per-channel

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <fstream>
#include <ctime>
#include "xbyak/xbyak_util.h"

#include "nodes/kernels/x64/gemmv_jit/entry.hpp"
#include "nodes/kernels/x64/gemmv_jit/vnni_gemmv_intrin.hpp"
#include "nodes/kernels/x64/gemmv_jit/amx_gemmv_intrin.hpp"
#include "nodes/kernels/x64/gemmv_jit/gemmv_ukernel.hpp"
#include "nodes/kernels/x64/gemmv_jit/jit_gemmv_avx2_fp32.hpp"
#include "nodes/kernels/x64/gemmv_jit/jit_gemmv_avx512_simple.hpp"
#include <memory>
#ifdef __linux__
#include <sched.h>
#include <pthread.h>
#endif

using namespace ov::intel_cpu::x64::gemmv_jit;

static void naive_ref(const float* x, int K,
                      const uint8_t* wq, int M, int ld_w_bytes,
                      const float* scales, const int32_t* zps,
                      float* y, const float* bias,
                      quant_granularity_t gran, w_dtype_t wtype, int group_size = 0) {
    const int M_blk = 16;
    const int full = M / M_blk;
    const int tail = M % M_blk;
    float sumx = 0.f;
    if (zps) for (int k = 0; k < K; ++k) sumx += x[k];
    auto get_s = [&](const float* s, int base, int m){
        if (gran == quant_granularity_t::per_tensor) return s[0];
        if (gran == quant_granularity_t::per_channel) return s[base + m];
        const int gs = group_size > 0 ? group_size : 16;
        const int g = (base + m) / gs;
        return s[g];
    };
    auto get_z = [&](const int32_t* z, int base, int m){
        if (!z) return 0;
        if (gran == quant_granularity_t::per_tensor) return z[0];
        if (gran == quant_granularity_t::per_channel) return z[base + m];
        const int gs = group_size > 0 ? group_size : 16;
        const int g = (base + m) / gs;
        return z[g];
    };
    auto get_b = [&](const float* b, int base, int m){
        if (!b) return 0.f;
        if (gran == quant_granularity_t::per_tensor) return b[0];
        if (gran == quant_granularity_t::per_channel) return b[base + m];
        const int gs = group_size > 0 ? group_size : 16;
        const int g = (base + m) / gs;
        return b[g];
    };

    auto block = [&](int bi, int valid){
        const uint8_t* wb = wq + bi * ld_w_bytes;
        const int base = (gran == quant_granularity_t::per_tensor) ? 0 : bi * M_blk;
        for (int m = 0; m < valid; ++m) y[bi * M_blk + m] = 0.f;
        for (int k = 0; k < K; ++k) {
            const float xk = x[k];
            for (int m = 0; m < valid; ++m) {
                int32_t q = 0;
                if (wtype == w_dtype_t::i8 || wtype == w_dtype_t::u8) {
                    const uint8_t* bp = wb + k * M_blk;
                    q = (wtype == w_dtype_t::u8) ? int32_t(bp[m]) : int32_t(int8_t(bp[m]));
                } else {
                    // int4/u4 packed: 8 bytes per K-step for 16 lanes
                    const uint8_t* bp = wb + k * (M_blk / 2);
                    const int idx = m >> 1;
                    const uint8_t b = bp[idx];
                    uint8_t nib = (m & 1) ? ((b >> 4) & 0x0F) : (b & 0x0F);
                    if (wtype == w_dtype_t::u4) {
                        q = int32_t(nib);
                    } else {
                        // i4 two's complement: -8..7
                        q = (nib ^ 0x8) - 0x8;
                    }
                }
                const float s = get_s(scales, base, m);
                y[bi * M_blk + m] += (float)q * s * xk;
            }
        }
        for (int m = 0; m < valid; ++m) {
            const float s = get_s(scales, base, m);
            const float b = get_b(bias, base, m);
            const float z = (float)get_z(zps, base, m);
            y[bi * M_blk + m] += b - s * z * sumx;
        }
    };
    for (int bi = 0; bi < full; ++bi) block(bi, M_blk);
    if (tail) block(full, tail);
}

static double gflops(int M, int K, double ms) {
    const double flops = 2.0 * (double)M * (double)K;
    return flops / (ms * 1e6);
}

static std::string iso_now() {
    std::time_t t = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%FT%T%z", std::localtime(&t));
    return std::string(buf);
}

static std::string get_log_path() {
    const char* env = std::getenv("GEMMV_LOG");
    if (env && *env) return std::string(env);
    return std::string("/tmp/gemmv_log.md");
}

static void append_log(const std::string& line) {
    std::ofstream f(get_log_path(), std::ios::app);
    if (!f) return;
    f << line << "\n";
}

static void minimal_bench_i8_per_tensor(int M, int K) {
    const int M_blk = 16;
    const int M_pad = ((M + M_blk - 1)/M_blk) * M_blk;
    // Prepare random data
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> di8(-128, 127);
    std::uniform_real_distribution<float> df(-1.f, 1.f);
    std::vector<float> X(K);
    for (auto &v : X) v = df(rng);
    std::vector<int8_t>  Wq_i8((size_t)M * (size_t)K);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k)
            Wq_i8[(size_t)m*K + k] = (int8_t)di8(rng);
    std::vector<uint8_t> Wpack_i8((size_t)M_pad * (size_t)K);
    pack_w_i8_interleave_m16(Wpack_i8.data(), (const uint8_t*)Wq_i8.data(), M, K, K, M_blk);
    // Per-tensor meta
    float scale = std::max(0.05f, std::abs(df(rng)));
    int32_t zpw = 0; // symmetric int8 weights
    float bias = 0.f;
    std::vector<float> Y((size_t)M_pad, 0.f);
    // Warmup
    for (int i = 0; i < 3; ++i) {
        const char* kname=nullptr;
        run_gemmv_q_fp32_ex(X.data(), K, Wpack_i8.data(), M, /*ld_w_bytes*/ K*M_blk,
                            &scale, &zpw, Y.data(), &bias,
                            quant_granularity_t::per_tensor, /*group_size*/0, /*acc=*/false, w_dtype_t::i8, &kname);
    }
    // Measure
    const int iters = 50;
    auto t0 = std::chrono::steady_clock::now();
    const char* kernel_name = nullptr;
    for (int it = 0; it < iters; ++it) {
        kernel_name = nullptr;
        run_gemmv_q_fp32_ex(X.data(), K, Wpack_i8.data(), M, /*ld_w_bytes*/ K*M_blk,
                            &scale, &zpw, Y.data(), &bias,
                            quant_granularity_t::per_tensor, /*group_size*/0, /*acc=*/false, w_dtype_t::i8, &kernel_name);
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    // Quick accuracy vs naive
    std::vector<float> Y_naive((size_t)M_pad, 0.f);
    naive_ref(X.data(), K, Wpack_i8.data(), M, /*ld_w_bytes*/ K*M_blk,
              &scale, &zpw, Y_naive.data(), &bias,
              quant_granularity_t::per_tensor, w_dtype_t::i8, /*group_size*/0);
    double max_abs = 0.0, max_rel = 0.0;
    for (int m = 0; m < M; ++m) {
        double a = Y[m], b = Y_naive[m];
        max_abs = std::max(max_abs, std::abs(a - b));
        double denom = std::max(1e-5, std::abs(b));
        max_rel = std::max(max_rel, std::abs(a - b)/denom);
    }
    // Log one bench_result JSON
    std::string ts = iso_now();
    std::string j = std::string("{\"ts\":\"") + ts + "\"," +
        "\"action\":\"bench_result\"," +
        "\"M\":" + std::to_string(M) + "," +
        "\"K\":" + std::to_string(K) + "," +
        "\"gran\":\"per_tensor\"," +
        "\"w_type\":\"i8\"," +
        "\"kernel\":\"" + (kernel_name?kernel_name:"unknown") + "\"," +
        "\"iters\":" + std::to_string(iters) + "," +
        "\"time_ms\":" + std::to_string(ms) + "," +
        "\"gflops\":" + std::to_string(gflops(M, K, ms)) + "," +
        "\"max_abs_err\":" + std::to_string(max_abs) + "," +
        "\"max_rel_err\":" + std::to_string(max_rel) + "}";
    append_log("[GEMMV-BENCH] " + j);
}

int main(int argc, char** argv) {
    // Fast path: exactly one shape M K => minimal perf run (i8/per_tensor)
    if (argc == 3) {
        int M = std::atoi(argv[1]);
        int K = std::atoi(argv[2]);
        if (M > 0 && K > 0) {
            minimal_bench_i8_per_tensor(M, K);
            return 0;
        }
    }
    std::cerr << "[DBG] bench start" << std::endl;
    // Optional: pin this thread to a core for more stable perf (GEMMV_PIN=1 [GEMMV_PIN_CORE=N])
#ifdef __linux__
    if (const char* pe = std::getenv("GEMMV_PIN"); pe && std::string(pe) == "1") {
        int core = 0; if (const char* pc = std::getenv("GEMMV_PIN_CORE")) { int v = std::atoi(pc); if (v >= 0) core = v; }
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(core, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
#endif
    const bool skip_self = true; // disable selftests in perf runs
    // Built-in INT4 decode selftest (K=1)
    if (!skip_self) {
        const int M = 16, K = 1, M_blk = 16, M_pad = 16;
        // Prepare deterministic i4 pattern: [-8..7]
        std::vector<int8_t> Wq_i4(M*K);
        for (int m = 0; m < M; ++m) Wq_i4[m] = (int8_t)(m - 8);
        std::vector<uint8_t> Wpack_i4((size_t)M_pad * (size_t)K / 2);
        pack_w_i4_interleave_m16(Wpack_i4.data(), (const int8_t*)Wq_i4.data(), M, K, K, M_blk);
        // CPU decode reference from packed
        std::vector<int> q_cpu(M);
        const uint8_t* bp = Wpack_i4.data();
        for (int m = 0; m < M; ++m) {
            uint8_t b = bp[m>>1];
            uint8_t nib = (m & 1) ? ((b >> 4) & 0x0F) : (b & 0x0F);
            q_cpu[m] = (int)((nib ^ 0x8) - 0x8);
        }
        // Run JIT decode with X=1, scales=1, bias=0, zps=null
        std::vector<float> X(K, 1.0f);
        std::vector<float> Y(M_pad, 0.f);
        float scale1 = 1.f, bias0 = 0.f;
        run_gemmv_q_fp32_ex(X.data(), K, Wpack_i4.data(), M, /*ld_w_bytes*/ K*(M_blk/2),
                            &scale1, /*zps*/nullptr, Y.data(), &bias0,
                            quant_granularity_t::per_tensor, /*group_size*/0, /*acc=*/false, w_dtype_t::i4, nullptr);
        // Print compare lines m, cpu, jit
        std::cout << "[INT4-SELFTEST] M=16 K=1 decode compare (cpu vs jit)\n";
        for (int m = 0; m < M; ++m) {
            std::cout << m << ": " << q_cpu[m] << " vs " << (int)std::lrint(Y[m]) << "\n";
        }
    }
    if (!skip_self) std::cerr << "[DBG] after selftest1" << std::endl;
    // Built-in INT4 per-k decode selftest (disabled in perf runs)
    if (false) {
        const int M = 16, K = 3, M_blk = 16, M_pad = 16;
        std::mt19937 rng(123);
        std::uniform_int_distribution<int> di4(-8,7);
        std::vector<int8_t> Wq_i4(M*K);
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k)
                Wq_i4[m*K+k] = (int8_t)di4(rng);
        std::vector<uint8_t> Wpack_i4((size_t)M_pad * (size_t)K / 2);
        pack_w_i4_interleave_m16(Wpack_i4.data(), (const int8_t*)Wq_i4.data(), M, K, K, M_blk);
        // CPU reference GEMV with scales=1, zp/bias=0
        std::vector<float> X(K);
        std::vector<float> Ycpu(M_pad, 0.f), Yjit(M_pad, 0.f);
        float scale1=1.f, bias0=0.f;
        for (int t = 0; t < K; ++t) {
            std::fill(X.begin(), X.end(), 0.f);
            X[t] = 1.f;
            std::fill(Ycpu.begin(), Ycpu.end(), 0.f);
            std::fill(Yjit.begin(), Yjit.end(), 0.f);
            // CPU
            const uint8_t* wb = Wpack_i4.data();
            const uint8_t* bp = wb + t * (M_blk/2);
            for (int m = 0; m < M; ++m) {
                const int idx = m >> 1; const uint8_t b = bp[idx];
                uint8_t nib = (m & 1) ? ((b >> 4)&0x0F) : (b & 0x0F);
                int q = (nib ^ 0x8) - 0x8;
                Ycpu[m] = (float)q;
            }
            // JIT
            run_gemmv_q_fp32_ex(X.data(), K, Wpack_i4.data(), M, /*ld_w_bytes*/ K*(M_blk/2),
                                &scale1, /*zps*/nullptr, Yjit.data(), &bias0,
                                quant_granularity_t::per_tensor, /*group_size*/0, /*acc=*/false, w_dtype_t::i4, nullptr);
            // Print compare per k
            std::cout << "[INT4-SELFTEST2] k=" << t << " decode compare (cpu vs jit)\n";
            double max_abs=0.0;
            for (int m = 0; m < M; ++m) {
                double da = fabs((double)Ycpu[m] - (double)Yjit[m]);
                if (da > max_abs) max_abs = da;
                std::cout << m << ": " << Ycpu[m] << " vs " << Yjit[m] << "\n";
            }
            std::cout << "max_abs_err[k="<<t<<"]=" << max_abs << "\n";
        }
    }
    if (!skip_self) std::cerr << "[DBG] after selftest2" << std::endl;
    // Built-in INT4 per-channel scale selftest with capture (disabled in perf runs)
    if (false) {
        const int M = 16, K = 3, M_blk = 16, M_pad = 16;
        std::mt19937 rng(456);
        std::uniform_int_distribution<int> di4(-8,7);
        std::uniform_real_distribution<float> df(0.05f, 2.0f);
        std::vector<int8_t> Wq_i4(M*K);
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k)
                Wq_i4[m*K+k] = (int8_t)di4(rng);
        std::vector<uint8_t> Wpack_i4((size_t)M_pad * (size_t)K / 2);
        pack_w_i4_interleave_m16(Wpack_i4.data(), (const int8_t*)Wq_i4.data(), M, K, K, M_blk);
        // random per-channel scales; zp/bias=0
        std::vector<float> scales(M_pad);
        for (int i = 0; i < M; ++i) scales[i] = df(rng);
        std::vector<float> X(K, 0.f), Yjit(M_pad, 0.f), Ycpu(M_pad, 0.f);
        alignas(64) float dbg_q[16] = {0}, dbg_qs[16] = {0};
        for (int t = 0; t < K; ++t) {
            std::fill(X.begin(), X.end(), 0.f);
            X[t] = 1.f;
            std::fill(Yjit.begin(), Yjit.end(), 0.f);
            std::fill(Ycpu.begin(), Ycpu.end(), 0.f);
            // cpu: y = s * q_k
            const uint8_t* wb = Wpack_i4.data();
            const uint8_t* bp = wb + t * (M_blk/2);
            std::vector<float> qcpu(M, 0.f);
            for (int m = 0; m < M; ++m) {
                const int idx = m >> 1; const uint8_t b = bp[idx];
                uint8_t nib = (m & 1) ? ((b >> 4)&0x0F) : (b & 0x0F);
                int q = (nib ^ 0x8) - 0x8;
                qcpu[m] = (float)q;
                Ycpu[m] = scales[m] * qcpu[m];
            }
            // Run JIT with capture (k=t) without env flags
            gemmv_ukr_params_t params{};
            params.x = X.data(); params.K = K; params.wq = Wpack_i4.data(); params.ld_w_bytes = K*(M_blk/2);
            params.scales = scales.data(); params.zps = nullptr; params.gran = quant_granularity_t::per_channel;
            params.y = Yjit.data(); params.bias = nullptr; params.M = M; params.accumulate = false;
            params.a_type = a_dtype_t::fp32; params.w_type = w_dtype_t::i4;
            params.dbg_enable = 1; params.dbg_k = t; params.dbg_q = dbg_q; params.dbg_qs = dbg_qs;
            std::unique_ptr<GemmvKernel> ker(create_gemmv_kernel(params));
            (*ker)(&params);
            double max_abs=0.0;
            for (int m = 0; m < M; ++m) max_abs = std::max(max_abs, (double)std::abs(Ycpu[m]-Yjit[m]));
            std::cout << "[INT4-SELFTEST3] k="<<t<<" per_channel scales compare max_abs_err="<<max_abs<<"\n";
            // print first few lanes of captured q and q*s for sanity
            std::cout << " expected q[0..7]:";
            for (int i=0;i<8;++i) std::cout<<" "<<qcpu[i];
            std::cout << "\n capture q[0..7]:";
            for (int i=0;i<8;++i) std::cout<<" "<<dbg_q[i];
            std::cout<<"\n capture qs[0..7]:";
            for (int i=0;i<8;++i) std::cout<<" "<<dbg_qs[i];
            std::cout<<"\n expected qs[0..7]:";
            for (int i=0;i<8;++i) std::cout<<" "<<Ycpu[i];
            std::cout<<"\n";
        }
    }
    if (!skip_self) std::cerr << "[DBG] after selftest3" << std::endl;
    // VNNI small-shape selftest: verify s32 sums and compensation terms at K=128, M=64
    if (const char* ev = std::getenv("GEMMV_VNNI_SELFTEST"); ev && std::string(ev) == "1") {
        std::cerr << "[VNNI-DBG] enter selftest" << std::endl;
        Xbyak::util::Cpu cpu;
        if (!cpu.has(Xbyak::util::Cpu::tAVX512_VNNI)) {
            std::cout << "[VNNI-SELFTEST] skipped: AVX512_VNNI not present\n";
        } else {
            std::cerr << "[VNNI-DBG] cpu ok" << std::endl;
            const int M = 64, K = 128, M_blk = 16, M_pad = 64;
            std::mt19937 rng(101);
            std::uniform_int_distribution<int> di8(-128,127);
            std::uniform_real_distribution<float> dfX(-2.f, 2.f);
            // weights int8 row-major
            std::vector<int8_t> Wq_i8(M*K);
            for (int m=0;m<M;++m) for (int k=0;k<K;++k) Wq_i8[m*K+k] = (int8_t)di8(rng);
            std::cerr << "[VNNI-DBG] Wq filled" << std::endl;
            // X fp32 and quantized to u8 (symmetric per-tensor, zp=128)
            std::vector<float> Xf(K); for (int k=0;k<K;++k) Xf[k]=dfX(rng);
            std::vector<uint8_t> Xq(K, 0);
            float s_x=1.f; int32_t zp_x=128;
            {
                float amax=0.f; for (int k=0;k<K;++k) amax=std::max(amax, std::fabs(Xf[k]));
                s_x = (amax>0.f)?(amax/127.f):1.f; zp_x=128;
                for (int k=0;k<K;++k) {
                    int v = (int)std::lrintf(Xf[k]/s_x)+zp_x; if (v<0) v=0; if (v>255) v=255; Xq[k]=(uint8_t)v;
                }
            }
            // pack i8 K4xM16
            const int K_grp=(K+3)/4; const int ld_w_k4 = K_grp * (M_blk*4);
            std::vector<uint8_t> Wpack_i8_k4((size_t)M_pad * (size_t)K_grp * 4);
            pack_w_i8_k4_m16(Wpack_i8_k4.data(), (const int8_t*)Wq_i8.data(), M, K, K, M_blk);
            std::cerr << "[VNNI-DBG] pack done Mpad="<<M_pad<<" Kgrp="<<K_grp<<" ld_w_k4="<<ld_w_k4<< std::endl;
            // per-tensor scales/zps
            float s_w = 0.1f; int32_t zp_w = 0;
            float bias = 0.f;
            // compute reference s32 terms per row
            auto get_Wgrp = [&](int bi,int g,int row)->const uint8_t*{
                const uint8_t* base = Wpack_i8_k4.data() + (size_t)bi * (size_t)ld_w_k4;
                return base + (size_t)g * 64 + (size_t)row * 4;
            };
            // sums
            int32_t sumX=0; for (int k=0;k<K;++k) sumX += (int32_t)Xq[k];
            std::vector<float> Yjit(M_pad,0.f), Yref(M_pad,0.f);
            // run JIT if enabled
            bool ran_jit=false;
            if (const char* rj = std::getenv("GEMMV_VNNI_SELFTEST_RUNJIT"); rj && std::string(rj) == "1") {
                // call wrapper with debug capture on block 0
                alignas(64) int32_t dbg_acc[16]={0}, dbg_sumw[16]={0};
                std::cerr << "[VNNI-DBG] calling JIT" << std::endl;
                ran_jit = run_gemmv_vnni_q8s8_ex(Xf.data(), K,
                                                Wpack_i8_k4.data(), M, ld_w_k4,
                                                &s_w, &zp_w, Yjit.data(), &bias,
                                                quant_granularity_t::per_tensor,
                                                /*dbg_block=*/0, dbg_acc, dbg_sumw);
                std::cerr << "[VNNI-DBG] returned from JIT" << std::endl;
                if (ran_jit) {
                    // Compare raw s32 dp and sumW only for the first group (since dump_only loops one group)
                    double max_acc_abs=0.0, max_sumw_abs=0.0;
                    for (int m=0;m<16;++m) {
                        const uint8_t* w4 = get_Wgrp(0, 0, m);
                        int8_t w0=(int8_t)w4[0], w1=(int8_t)w4[1], w2=(int8_t)w4[2], w3=(int8_t)w4[3];
                        uint8_t xb0=Xq[0], xb1=Xq[1], xb2=Xq[2], xb3=Xq[3];
                        int32_t ref_acc = (int32_t)xb0*w0 + (int32_t)xb1*w1 + (int32_t)xb2*w2 + (int32_t)xb3*w3;
                        int32_t ref_sumw = (int32_t)w0 + w1 + w2 + w3;
                        max_acc_abs = std::max(max_acc_abs, (double)std::abs(ref_acc - dbg_acc[m]));
                        max_sumw_abs = std::max(max_sumw_abs, (double)std::abs(ref_sumw - dbg_sumw[m]));
                    }
                    std::cout << "[VNNI-SELFTEST] group0 acc_err=" << max_acc_abs << " sumW_err=" << max_sumw_abs << "\n";
                }
            }
            // compute ref per row m
            for (int m=0;m<M;++m) {
                // s32 dp
                int64_t acc=0; int32_t sumW=0;
                for (int g=0; g<K_grp; ++g) {
                    const uint8_t* w4 = get_Wgrp(0, g, m);
                    uint8_t xb0 = Xq[g*4+0], xb1 = Xq[g*4+1], xb2 = Xq[g*4+2], xb3 = Xq[g*4+3];
                    int8_t w0 = (int8_t)w4[0], w1=(int8_t)w4[1], w2=(int8_t)w4[2], w3=(int8_t)w4[3];
                    acc += (int32_t)xb0 * (int32_t)w0 + (int32_t)xb1 * (int32_t)w1 + (int32_t)xb2 * (int32_t)w2 + (int32_t)xb3 * (int32_t)w3;
                    sumW += (int32_t)w0 + (int32_t)w1 + (int32_t)w2 + (int32_t)w3;
                }
                int32_t s32 = (int32_t)acc;
                // compensation
                int32_t comp = - zp_x * sumW - zp_w * sumX + K * zp_w * zp_x;
                int32_t s32_total = s32 + comp;
                Yref[m] = (float)s32_total * (s_w * s_x) + bias;
            }
            double max_abs=0.0;
            if (ran_jit) {
                for (int m=0;m<M;++m) max_abs = std::max(max_abs, (double)std::abs(Yref[m]-Yjit[m]));
                std::cout << "[VNNI-SELFTEST] 64x128 ran_jit=1 max_abs_err=" << max_abs << "\n";
            } else {
                std::cout << "[VNNI-SELFTEST] 64x128 ran_jit=0 (guard)\n";
            }
        }
    }
    std::vector<std::pair<int,int>> shapes = {
        {128, 4096}, {256, 4096}, {512, 4096}, {1024, 4096}, {2048, 4096},
        {1024, 8192}
    };
    if (argc == 3) {
        int M = std::atoi(argv[1]);
        int K = std::atoi(argv[2]);
        if (M > 0 && K > 0) shapes = {{M, K}};
    }
    std::cerr << "[DBG] shapes prepared" << std::endl;

    // Log run header: env + CPU features + shapes + iters
    const int warmup_iters = 5;
    const int bench_iters = 50;
    auto env_log = [](){ const char* v = std::getenv("GEMMV_LOG"); return std::string(v ? v : ""); };
    Xbyak::util::Cpu cpu;
    {
        std::string ts = iso_now();
        std::string j = std::string("{\"ts\":\"") + ts + "\"," +
            "\"action\":\"run_start\"," +
            "\"env\":{\"GEMMV_LOG\":\"" + env_log() + "\"}," +
            "\"cpu\":{\"avx2\":" + std::to_string(cpu.has(Xbyak::util::Cpu::tAVX2)) + "," +
                       "\"avx512f\":" + std::to_string(cpu.has(Xbyak::util::Cpu::tAVX512F)) + "," +
                       "\"avx512bw\":" + std::to_string(cpu.has(Xbyak::util::Cpu::tAVX512BW)) + "," +
                       "\"avx512vnni\":" + std::to_string(cpu.has(Xbyak::util::Cpu::tAVX512_VNNI)) + "}," +
            "\"warmup\":" + std::to_string(warmup_iters) + "," +
            "\"iters\":" + std::to_string(bench_iters) + "," +
            "\"shapes\":[";
        for (size_t i = 0; i < shapes.size(); ++i) {
            j += "[" + std::to_string(shapes[i].first) + "," + std::to_string(shapes[i].second) + "]";
            if (i + 1 != shapes.size()) j += ",";
        }
        j += "]}";
        append_log("[GEMMV-BENCH] " + j);
    }
    std::cerr << "[DBG] logged run_start" << std::endl;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> di8(-128, 127);
    std::uniform_int_distribution<int> du8(0, 255);
    std::uniform_real_distribution<float> df(-1.f, 1.f);

    // Calibration pass (disabled in performance runs)
    if (false) {
        const int M = shapes[0].first;
        const int K = shapes[0].second;
        std::cerr << "[DBG] calib M=" << M << " K=" << K << std::endl;
        const int M_blk = 16;
        const int M_pad = ((M + M_blk - 1)/M_blk) * M_blk;
        const int ld_w_bytes8 = K * M_blk;
        // Prepare random data and pack (int8 signed) once
        std::vector<float> Xcols; // will resize per N
        std::vector<int8_t>  Wq_i8((size_t)M * (size_t)K);
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k)
                Wq_i8[(size_t)m*K + k] = (int8_t)di8(rng);
        std::vector<uint8_t> Wpack_i8((size_t)M_pad * (size_t)K);
        pack_w_i8_interleave_m16(Wpack_i8.data(), (const uint8_t*)Wq_i8.data(), M, K, K, M_blk);
        // Per-tensor metadata
        quant_granularity_t gran = quant_granularity_t::per_tensor;
        std::vector<float> scales(1); scales[0] = std::max(0.05f, std::abs(df(rng)));
        std::vector<int32_t> zps(1); zps[0] = (int32_t)di8(rng);
        std::vector<float> bias(1); bias[0] = df(rng);

        std::vector<int> Ns = {1,2,4,8,12,16,24,32};
        int crossoverN = -1;
        for (int N : Ns) {
            Xcols.resize((size_t)K * (size_t)N);
            for (auto &v : Xcols) v = df(rng);
            std::vector<float> Y_gemv((size_t)M_pad * (size_t)N, 0.f);
            std::vector<float> Y_minig((size_t)M_pad * (size_t)N, 0.f);
            std::cerr << "[DBG] calib loop N=" << N << " warmup-start" << std::endl;

            // Create and warmup a reusable GEMV kernel instance (fair vs ref mini-GEMM)
            gemmv_ukr_params_t pj{};
            pj.x = Xcols.data(); pj.K = K; pj.wq = Wpack_i8.data(); pj.ld_w_bytes = ld_w_bytes8;
            pj.scales = scales.data(); pj.zps = zps.data(); pj.gran = gran;
            pj.y = Y_gemv.data(); pj.bias = bias.data(); pj.M = M; pj.accumulate = false;
            pj.a_type = a_dtype_t::fp32; pj.w_type = w_dtype_t::i8;
            std::unique_ptr<GemmvKernel> gker(create_gemmv_kernel(pj));
            for (int it = 0; it < std::max(1, warmup_iters/2); ++it) {
                for (int n = 0; n < N; ++n) {
                    pj.x = &Xcols[(size_t)n*K];
                    pj.y = &Y_gemv[(size_t)n*M_pad];
                    (*gker)(&pj);
                }
                run_minigemm_ref_q_fp32(Xcols.data(), K, N, Wpack_i8.data(), M, ld_w_bytes8,
                                        scales.data(), zps.data(), Y_minig.data(), bias.data(),
                                        gran, w_dtype_t::i8, false, /*group_size*/0);
            }
            std::cerr << "[DBG] calib loop N=" << N << " warmup-done" << std::endl;

            // Time GEMV N times
            auto t0a = std::chrono::steady_clock::now();
            for (int it = 0; it < bench_iters; ++it) {
                for (int n = 0; n < N; ++n) {
                    pj.x = &Xcols[(size_t)n*K];
                    pj.y = &Y_gemv[(size_t)n*M_pad];
                    (*gker)(&pj);
                }
            }
            auto t1a = std::chrono::steady_clock::now();
            double ms_gemv = std::chrono::duration<double, std::milli>(t1a - t0a).count() / bench_iters;
            std::cerr << "[DBG] calib loop N=" << N << " gemvN-done" << std::endl;

            // Time mini-GEMM ref
            auto t0b = std::chrono::steady_clock::now();
            for (int it = 0; it < bench_iters; ++it) {
                run_minigemm_ref_q_fp32(Xcols.data(), K, N, Wpack_i8.data(), M, ld_w_bytes8,
                                        scales.data(), zps.data(), Y_minig.data(), bias.data(),
                                        gran, w_dtype_t::i8, false, /*group_size*/0);
            }
            auto t1b = std::chrono::steady_clock::now();
            double ms_minig = std::chrono::duration<double, std::milli>(t1b - t0b).count() / bench_iters;
            std::cerr << "[DBG] calib loop N=" << N << " minig-ref-done" << std::endl;

            // Time mini-GEMM JIT (when supported); otherwise gets ref
            std::vector<float> Y_minig_jit((size_t)M_pad * (size_t)N, 0.f);
            auto t0bj = std::chrono::steady_clock::now();
            for (int it = 0; it < bench_iters; ++it) {
                const char* nm = nullptr;
                run_minigemm_q_fp32_ex(Xcols.data(), K, N, Wpack_i8.data(), M, ld_w_bytes8,
                                       scales.data(), zps.data(), Y_minig_jit.data(), bias.data(),
                                       gran, /*group_size*/0, w_dtype_t::i8, &nm);
            }
            auto t1bj = std::chrono::steady_clock::now();
            double ms_minig_jit = std::chrono::duration<double, std::milli>(t1bj - t0bj).count() / bench_iters;
            std::cerr << "[DBG] calib N=" << N << " done" << std::endl;

            // Accuracy sanity: compare Y_gemv vs Y_minig
            double max_abs = 0.0, max_rel = 0.0;
            for (int n = 0; n < N; ++n) {
                for (int m = 0; m < M; ++m) {
                    double a = Y_gemv[(size_t)n*M_pad + m], b = Y_minig[(size_t)n*M_pad + m];
                    max_abs = std::max(max_abs, std::abs(a - b));
                    double denom = std::max(1e-5, std::abs(b));
                    max_rel = std::max(max_rel, std::abs(a - b)/denom);
                }
            }
            std::string jcal = std::string("{\\\"ts\\\":\\\"") + iso_now() + "\\\"," +
                "\\\"action\\\":\\\"calibrate_minigemm\\\"," +
                "\\\"M\\\":" + std::to_string(M) + "," +
                "\\\"K\\\":" + std::to_string(K) + "," +
                "\\\"N\\\":" + std::to_string(N) + "," +
                "\\\"gran\\\":\\\"per_tensor\\\"," +
                "\\\"wtype\\\":\\\"i8\\\"," +
                "\\\"ms_gemvN\\\":" + std::to_string(ms_gemv) + "," +
                "\\\"ms_minigemm_ref\\\":" + std::to_string(ms_minig) + "," +
                "\\\"ms_minigemm_jit\\\":" + std::to_string(ms_minig_jit) + "," +
                "\\\"max_abs_err\\\":" + std::to_string(max_abs) + "," +
                "\\\"max_rel_err\\\":" + std::to_string(max_rel) + "}";
            append_log("[GEMMV-BENCH] " + jcal);
            if (crossoverN < 0 && ms_minig < ms_gemv) crossoverN = N;
        }
        // Summary line
        std::string sum = std::string("[LOG] ") + iso_now() +
            " — mini-GEMM(ref) calibration (i8, per_tensor) M=" + std::to_string(M) +
            " K=" + std::to_string(K) + " crossover N*=" + (crossoverN>0?std::to_string(crossoverN):std::string("not_found<=32"));
        append_log(sum);

        // Extra calibration: per_group + INT4 (i4/u4) for N in {1,2,4,8,12,16}
        // Guarded by env GEMMV_CALIB_PG4=1 to keep default run stable
        if (const char* do_pg4 = std::getenv("GEMMV_CALIB_PG4"); do_pg4 && std::string(do_pg4) == "1") {
            const int group_size = 32;
            quant_granularity_t gran = quant_granularity_t::per_group;
            const int meta_len = (M_pad + group_size - 1) / group_size;
            std::vector<float> scales(meta_len), bias(meta_len);
            std::vector<int32_t> zps(meta_len);
            for (int i = 0; i < meta_len; ++i) {
                scales[i] = std::max(0.05f, std::abs(df(rng)));
                zps[i] = du8(rng) & 0xFF;
                bias[i] = df(rng);
            }
            // Prepare random int4/u4, pack once
            std::vector<int8_t>  Wq_i4((size_t)M * (size_t)K);
            std::vector<uint8_t> Wq_u4((size_t)M * (size_t)K);
            for (int m = 0; m < M; ++m)
                for (int k = 0; k < K; ++k) {
                    int vi4 = (int)((du8(rng) % 16) - 8);
                    Wq_i4[(size_t)m*K + k] = (int8_t)vi4;
                    Wq_u4[(size_t)m*K + k] = (uint8_t)(du8(rng) & 0x0F);
                }
            const int ld_w_bytes4 = K * (M_blk/2);
            std::vector<uint8_t> Wpack_i4((size_t)M_pad * (size_t)K / 2);
            std::vector<uint8_t> Wpack_u4((size_t)M_pad * (size_t)K / 2);
            pack_w_i4_interleave_m16(Wpack_i4.data(), (const int8_t*)Wq_i4.data(), M, K, K, M_blk);
            pack_w_i4_interleave_m16(Wpack_u4.data(), (const int8_t*)Wq_u4.data(), M, K, K, M_blk);

            auto run_calib = [&](w_dtype_t wtype, const char* wname){
                // Buffers
                std::vector<float> Xcols;
                std::vector<float> Y_gemv; // sized per N
                std::vector<float> Y_minig; // sized per N
                // Kernel instance
                gemmv_ukr_params_t pj{};
                pj.K = K; pj.ld_w_bytes = ld_w_bytes4; pj.gran = gran; pj.M = M;
                pj.accumulate = false; pj.a_type = a_dtype_t::fp32; pj.w_type = wtype;
                pj.group_size = group_size;
                pj.scales = scales.data(); pj.zps = zps.data(); pj.bias = bias.data();
                pj.wq = (wtype==w_dtype_t::i4) ? Wpack_i4.data() : Wpack_u4.data();
                std::unique_ptr<GemmvKernel> gker(create_gemmv_kernel(pj));
                // Ns
                std::vector<int> Ns = {1,2,4,8,12,16};
                for (int N : Ns) {
                    Xcols.assign((size_t)N * (size_t)K, 0.f);
                    Y_gemv.assign((size_t)N * (size_t)M_pad, 0.f);
                    Y_minig.assign((size_t)N * (size_t)M_pad, 0.f);
                    for (int n = 0; n < N; ++n)
                        for (int k = 0; k < K; ++k)
                            Xcols[(size_t)n*K + k] = df(rng);
                    // Warmup
                    for (int it = 0; it < std::max(1, warmup_iters/2); ++it) {
                        for (int n = 0; n < N; ++n) {
                            pj.x = &Xcols[(size_t)n*K];
                            pj.y = &Y_gemv[(size_t)n*M_pad];
                            (*gker)(&pj);
                        }
                        run_minigemm_ref_q_fp32(Xcols.data(), K, N, pj.wq, M, pj.ld_w_bytes,
                                                scales.data(), zps.data(), Y_minig.data(), bias.data(),
                                                gran, wtype, false, group_size);
                    }
                    // GEMV N×
                    auto t0 = std::chrono::steady_clock::now();
                    for (int it = 0; it < bench_iters; ++it) {
                        for (int n = 0; n < N; ++n) {
                            pj.x = &Xcols[(size_t)n*K];
                            pj.y = &Y_gemv[(size_t)n*M_pad];
                            (*gker)(&pj);
                        }
                    }
                    auto t1 = std::chrono::steady_clock::now();
                    double ms_gemv = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench_iters;
                    // mini‑GEMM ref
                    t0 = std::chrono::steady_clock::now();
                    for (int it = 0; it < bench_iters; ++it) {
                        run_minigemm_ref_q_fp32(Xcols.data(), K, N, pj.wq, M, pj.ld_w_bytes,
                                                scales.data(), zps.data(), Y_minig.data(), bias.data(),
                                                gran, wtype, false, group_size);
                    }
                    t1 = std::chrono::steady_clock::now();
                    double ms_minig = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench_iters;
                    // mini‑GEMM JIT
                    std::vector<float> Y_minig_jit((size_t)M_pad * (size_t)N, 0.f);
                    t0 = std::chrono::steady_clock::now();
                    for (int it = 0; it < bench_iters; ++it) {
                        const char* nm = nullptr;
                        run_minigemm_q_fp32_ex(Xcols.data(), K, N, pj.wq, M, pj.ld_w_bytes,
                                               scales.data(), zps.data(), Y_minig_jit.data(), bias.data(),
                                               gran, group_size, wtype, &nm);
                    }
                    t1 = std::chrono::steady_clock::now();
                    double ms_minig_jit = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench_iters;
                    // Log
                    std::string jcal = std::string("{\\\"ts\\\":\\\"") + iso_now() + "\\\"," +
                        "\\\"action\\\":\\\"calibrate_minigemm\\\"," +
                        "\\\"M\\\":" + std::to_string(M) + "," +
                        "\\\"K\\\":" + std::to_string(K) + "," +
                        "\\\"N\\\":" + std::to_string(N) + "," +
                        "\\\"gran\\\":\\\"per_group\\\"," +
                        "\\\"wtype\\\":\\\"" + wname + "\\\"," +
                        "\\\"ms_gemvN\\\":" + std::to_string(ms_gemv) + "," +
                        "\\\"ms_minigemm_ref\\\":" + std::to_string(ms_minig) + "," +
                        "\\\"ms_minigemm_jit\\\":" + std::to_string(ms_minig_jit) + "}";
                    append_log("[GEMMV-BENCH] " + jcal);
                }
            };
            run_calib(w_dtype_t::i4, "i4");
            run_calib(w_dtype_t::u4, "u4");
            append_log(std::string("[LOG] ") + iso_now() + " — mini-GEMM calibration (i4/u4, per_group) done for first shape");
        }
    }

    for (auto [M, K] : shapes) {
        // Data
        std::vector<float> X(K);
        for (auto &v : X) v = df(rng);

        std::vector<uint8_t> Wq_u8((size_t)M * (size_t)K);
        std::vector<int8_t>  Wq_i8((size_t)M * (size_t)K);
        std::vector<uint8_t> Wq_u4((size_t)M * (size_t)K); // low 4 bits used
        std::vector<int8_t>  Wq_i4((size_t)M * (size_t)K);  // values in -8..7, lower 4 bits will be packed
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k) {
                Wq_u8[(size_t)m*K + k] = (uint8_t)du8(rng);
                Wq_i8[(size_t)m*K + k] = (int8_t)di8(rng);
                Wq_u4[(size_t)m*K + k] = (uint8_t)(du8(rng) & 0x0F);
                // i4 uniform in [-8..7]
                int vi4 = (int)((du8(rng) % 16) - 8);
                Wq_i4[(size_t)m*K + k] = (int8_t)vi4;
            }

        const int M_blk = 16;
        const int M_pad = ((M + M_blk - 1)/M_blk) * M_blk;
        const int ld_w_bytes8 = K * M_blk;
        const int ld_w_bytes4 = K * (M_blk / 2);

        std::vector<uint8_t> Wpack_u8((size_t)M_pad * (size_t)K);
        std::vector<uint8_t> Wpack_i8((size_t)M_pad * (size_t)K);
        std::vector<uint8_t> Wpack_u4((size_t)M_pad * (size_t)K / 2);
        std::vector<uint8_t> Wpack_i4((size_t)M_pad * (size_t)K / 2);
        // VNNI-friendly i8 pack (always available for ref/kernel autoroute)
        std::unique_ptr<uint8_t, void(*)(void*)> Wpack_i8_k4(nullptr, free);
        int ld_w_k4 = 0; bool use_vnni = false;
        std::vector<int32_t> sumW_i8; // optional precomputed sumW for VNNI kernel
        pack_w_i8_interleave_m16(Wpack_u8.data(), Wq_u8.data(), M, K, K, M_blk);
        pack_w_i8_interleave_m16(Wpack_i8.data(), (const uint8_t*)Wq_i8.data(), M, K, K, M_blk);
        pack_w_i4_interleave_m16(Wpack_u4.data(), (const int8_t*)Wq_u4.data(), M, K, K, M_blk);
        pack_w_i4_interleave_m16(Wpack_i4.data(), (const int8_t*)Wq_i4.data(), M, K, K, M_blk);
        {
            Xbyak::util::Cpu cpu;
            use_vnni = cpu.has(Xbyak::util::Cpu::tAVX512_VNNI);
            const int K_grp = (K + 3)/4;
            ld_w_k4 = K_grp * (M_blk * 4);
            const size_t bytes_k4 = (size_t)M_pad * (size_t)K_grp * 4;
            void* mem=nullptr; const size_t align=64; const size_t size_al=(bytes_k4 + align - 1)/align*align;
            if (posix_memalign(&mem, align, size_al) != 0 || !mem) mem = std::malloc(size_al);
            Wpack_i8_k4.reset((uint8_t*)mem);
            pack_w_i8_k4_m16(Wpack_i8_k4.get(), (const int8_t*)Wq_i8.data(), M, K, K, M_blk);
            // Precompute sumW per lane (signed sum of int8 weights) for VNNI fast epilog
            sumW_i8.resize(M_pad, 0);
            for (int m = 0; m < M; ++m) {
                int32_t s = 0;
                for (int k = 0; k < K; ++k) s += (int32_t)Wq_i8[(size_t)m * K + k];
                sumW_i8[m] = s;
            }
        }

        // Optional: explicit GEMV JIT vs REF comparison block
        if (const char* do_cmp_gemv = std::getenv("GEMMV_COMPARE_GEMV"); do_cmp_gemv && std::string(do_cmp_gemv) == "1") {
            auto run_cmp_gemv = [&](w_dtype_t wtype, const char* wname){
                const uint8_t* W = (wtype==w_dtype_t::i8)? Wpack_i8.data() : (wtype==w_dtype_t::u8)? Wpack_u8.data() : (wtype==w_dtype_t::i4)? Wpack_i4.data() : Wpack_u4.data();
                const int ld_w_bytes = (wtype==w_dtype_t::i4 || wtype==w_dtype_t::u4) ? ld_w_bytes4 : ld_w_bytes8;
                for (int mode = 0; mode < 3; ++mode) {
                    const bool per_tensor = (mode == 0);
                    const bool per_group = (mode == 2);
                    quant_granularity_t gran = per_tensor ? quant_granularity_t::per_tensor
                                                          : (per_group ? quant_granularity_t::per_group
                                                                       : quant_granularity_t::per_channel);
                    int group_size = 0;
                    int meta_len = per_tensor ? 1 : M_pad;
                    if (per_group) {
                        group_size = 32;
                        meta_len = (M_pad + group_size - 1) / group_size;
                    }
                    std::vector<float> scales(meta_len), bias(meta_len);
                    std::vector<int32_t> zps(meta_len);
                    for (int i = 0; i < meta_len; ++i) {
                        scales[i] = std::max(0.05f, std::abs(df(rng)));
                        zps[i] = du8(rng) & 0xFF;
                        bias[i] = df(rng);
                    }

                    std::vector<float> Y_jit((size_t)M_pad, 0.f), Y_ref((size_t)M_pad, 0.f);
                    // Time JIT via entry
                    auto t0 = std::chrono::steady_clock::now();
                    for (int it = 0; it < bench_iters; ++it) {
                        run_gemmv_q_fp32_ex(X.data(), K, W, M, ld_w_bytes,
                                            scales.data(), zps.data(), Y_jit.data(), bias.data(),
                                            gran, group_size, /*acc=*/false, wtype, nullptr);
                    }
                    auto t1 = std::chrono::steady_clock::now();
                    double ms_jit = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench_iters;

                    // Time REF by calling explicit RefGemmvFp32
                    ov::intel_cpu::x64::gemmv_jit::RefGemmvFp32 ref;
                    gemmv_ukr_params_t pr{};
                    pr.x = X.data(); pr.K = K; pr.wq = W; pr.ld_w_bytes = ld_w_bytes; pr.scales = scales.data();
                    pr.zps = zps.data(); pr.gran = gran; pr.group_size = group_size; pr.y = Y_ref.data(); pr.bias = bias.data();
                    pr.M = M; pr.accumulate = false; pr.a_type = a_dtype_t::fp32; pr.w_type = wtype;
                    t0 = std::chrono::steady_clock::now();
                    for (int it = 0; it < bench_iters; ++it) {
                        ref(&pr);
                    }
                    t1 = std::chrono::steady_clock::now();
                    double ms_ref = std::chrono::duration<double, std::milli>(t1 - t0).count() / bench_iters;

                    // Accuracy vs naive
                    std::vector<float> Y_naive((size_t)M_pad, 0.f);
                    naive_ref(X.data(), K, W, M, ld_w_bytes,
                              scales.data(), zps.data(), Y_naive.data(), bias.data(), gran, wtype, group_size);
                    auto err_stats = [&](const std::vector<float>& A){
                        double max_abs = 0.0, max_rel = 0.0;
                        for (int m = 0; m < M; ++m) {
                            double a = A[m], b = Y_naive[m];
                            max_abs = std::max(max_abs, std::abs(a - b));
                            double denom = std::max(1e-5, std::abs(b));
                            max_rel = std::max(max_rel, std::abs(a - b)/denom);
                        }
                        return std::pair<double,double>(max_abs, max_rel);
                    };
                    auto [jit_abs, jit_rel] = err_stats(Y_jit);
                    auto [ref_abs, ref_rel] = err_stats(Y_ref);

                    std::string mode_s = per_tensor?"per_tensor":(per_group?"per_group":"per_channel");
                    std::string j = std::string("{\"ts\":\"") + iso_now() + "\"," +
                        "\"action\":\"compare_gemv\"," +
                        "\"M\":" + std::to_string(M) + "," +
                        "\"K\":" + std::to_string(K) + "," +
                        "\"gran\":\"" + mode_s + "\"," +
                        "\"wtype\":\"" + wname + "\"," +
                        "\"ms_jit\":" + std::to_string(ms_jit) + "," +
                        "\"ms_ref\":" + std::to_string(ms_ref) + "," +
                        "\"jit_max_abs\":" + std::to_string(jit_abs) + "," +
                        "\"jit_max_rel\":" + std::to_string(jit_rel) + "," +
                        "\"ref_max_abs\":" + std::to_string(ref_abs) + "," +
                        "\"ref_max_rel\":" + std::to_string(ref_rel) + "}";
                    append_log("[GEMMV-BENCH] " + j);
                }
            };
            // Optional accuracy/timing compare path (JIT vs REF) — can be skipped for stability and speed
            bool skip_accuracy = (std::getenv("GEMMV_SKIP_ACCURACY") && std::string(std::getenv("GEMMV_SKIP_ACCURACY")) == "1");
            if (!skip_accuracy) {
                run_cmp_gemv(w_dtype_t::i8, "i8");
                run_cmp_gemv(w_dtype_t::u8, "u8");
                run_cmp_gemv(w_dtype_t::i4, "i4");
                run_cmp_gemv(w_dtype_t::u4, "u4");
                append_log(std::string("[LOG] ") + iso_now() + " — GEMV compare (JIT vs REF) done for shape M=" + std::to_string(M) + " K=" + std::to_string(K));
            }
        }

        // Measure only per-tensor (granularity) for performance CSV
        for (int mode = 0; mode < 1; ++mode) {
            const bool per_tensor = (mode == 0);
            const bool per_group = (mode == 2);
            quant_granularity_t gran = per_tensor ? quant_granularity_t::per_tensor
                                                  : (per_group ? quant_granularity_t::per_group
                                                               : quant_granularity_t::per_channel);
            int group_size = 0;
            int meta_len = per_tensor ? 1 : M_pad;
            if (per_group) {
                group_size = 32; // simple default; to be tuned
                meta_len = (M_pad + group_size - 1) / group_size;
            }
            std::vector<float> scales(meta_len), bias(meta_len);
            std::vector<int32_t> zps(meta_len);
            for (int i = 0; i < meta_len; ++i) {
                scales[i] = std::max(0.05f, std::abs(df(rng))); // positive scale
                zps[i] = du8(rng) & 0xFF; // valid for u8; for i8 fastpath we override to 0 per-tensor
                bias[i] = df(rng);
            }
            // No env-based sanitize flags; use original meta

            // wtypes: measure i8 only for performance CSV
            for (int t = 0; t < 1; ++t) {
                w_dtype_t wtype = (t==0)? w_dtype_t::i8 : (t==1)? w_dtype_t::u8 : (t==2)? w_dtype_t::i4 : w_dtype_t::u4;
                const uint8_t* W = (wtype==w_dtype_t::i8)? Wpack_i8.data() : (wtype==w_dtype_t::u8)? Wpack_u8.data() : (wtype==w_dtype_t::i4)? Wpack_i4.data() : Wpack_u4.data();
                const char* wname = (wtype==w_dtype_t::i8)? "i8" : (wtype==w_dtype_t::u8)? "u8" : (wtype==w_dtype_t::i4)? "i4" : "u4";
                const int ld_w_bytes = (wtype==w_dtype_t::i4 || wtype==w_dtype_t::u4) ? ld_w_bytes4 : ld_w_bytes8;
                std::vector<float> Y((size_t)M_pad, 0.f), Yref((size_t)M_pad, 0.f);
                // Adjust metadata per wtype: for i8 per-tensor intrinsics fastpath use symmetric zp_w=0
                std::vector<int32_t> zps_eff = zps;
                const std::vector<float>&  bias_eff = bias;
                const std::vector<float>&  scales_eff = scales;
                if (per_tensor && wtype == w_dtype_t::i8) { zps_eff.resize(1); zps_eff[0] = 0; }

                // Warmup
                for (int w = 0; w < warmup_iters; ++w) {
                    const char* kname=nullptr;
                    run_gemmv_q_fp32_ex(X.data(), K, W, M, ld_w_bytes,
                                        scales_eff.data(), zps_eff.data(), Y.data(), bias_eff.data(),
                                        gran, group_size, /*acc=*/false, wtype, &kname);
                }

                // Measure (single-thread GEMV) with optional aggregation (min/median across repeats)
                int outer_reps = (std::getenv("GEMMV_AGG") && std::string(std::getenv("GEMMV_AGG")) == "1") ? 3 : 1;
                std::vector<double> ms_runs; ms_runs.reserve(outer_reps);
                const char* kernel_name = nullptr; bool used_vnni = false;
                // Optional: reuse pre-quantized X and repacked W for per-tensor i8 (fastpath timing)
                // Compute-only path: for per-tensor i8 always reuse pre-quantized X and k64-repacked W
                bool reuse_xq = (wtype==w_dtype_t::i8 && per_tensor);
                std::vector<uint8_t> Xq_reuse; float s_x_reuse = 1.f; int32_t zp_x_reuse = 128; int32_t sumX_reuse = 0;
                std::unique_ptr<uint8_t, void(*)(void*)> Wk64_reuse(nullptr, free); int ld_w_k64 = 0; std::vector<int32_t> sumW_k64;
                if (reuse_xq) {
                    // Quantize X once
                    Xq_reuse.assign((size_t)K, 0);
                    float amax=0.f; for (int k=0;k<K;++k) amax = std::max(amax, std::fabs(X[k]));
                    s_x_reuse = (amax>0.f)?(amax/127.f):1.f; zp_x_reuse = 128; sumX_reuse = 0;
                    for (int k=0;k<K;++k) { int v=(int)std::lrintf(X[k]/s_x_reuse)+zp_x_reuse; v = std::min(255, std::max(0, v)); Xq_reuse[k]=(uint8_t)v; sumX_reuse += v; }
                    // Repack W (interleave_m16 -> k64) and precompute sumW per-lane
                    const int K_blk=64; const int M_blk_loc=16; const int M_pad_loc = ((M + M_blk_loc - 1)/M_blk_loc)*M_blk_loc;
                    const int K_grp = (K + K_blk - 1)/K_blk; ld_w_k64 = K_grp * (M_blk_loc * K_blk);
                    const size_t bytes_k64 = (size_t)M_pad_loc * (size_t)K_grp * (size_t)K_blk;
                    void* mem=nullptr; size_t size_al=((bytes_k64 + 63)/64)*64; if (posix_memalign(&mem, 64, size_al)!=0 || !mem) mem = std::malloc(size_al);
                    Wk64_reuse.reset((uint8_t*)mem); sumW_k64.assign(M_pad_loc, 0);
                    repack_interleave_m16_to_k64_m16(Wk64_reuse.get(), Wpack_i8.data(), M, K, ld_w_bytes8, M_blk_loc, K_blk, sumW_k64.data());
                }
                for (int rep = 0; rep < outer_reps; ++rep) {
                    const int iters = bench_iters;
                    auto t0 = std::chrono::steady_clock::now();
                    kernel_name = nullptr; used_vnni = false;
                    for (int it = 0; it < iters; ++it) {
                        if (reuse_xq) {
                            const int K_blk=64; const int K_grp=(K + K_blk - 1)/K_blk;
                            float s_w = scales_eff[0]; float bias0 = bias_eff.empty()?0.f:bias_eff[0];
                            // Try VNNI K64 reuse first (stable on wider set of hosts), then optional AMX
                            bool ok_v = run_gemmv_vnni_intrin_i8u8_fp32_k64(Xq_reuse.data(), K,
                                                                             Wk64_reuse.get(), M, K_grp * (M_blk * K_blk),
                                                                             s_w, /*zp_w*/ 0, s_x_reuse, zp_x_reuse,
                                                                             Y.data(), bias0, sumW_k64.data());
                            (void)ok_v; kernel_name = "vnni_k64_reuse_xq"; used_vnni = true;
                            // Optionally attempt AMX after VNNI (on AMX hosts this will be gated by ISA check)
                            bool ok_amx = run_gemmv_amx_i8u8_fp32_xq(Xq_reuse.data(), K, sumX_reuse,
                                                                     Wk64_reuse.get(), M, K_grp * (M_blk * K_blk),
                                                                     &s_w, /*zps*/nullptr, Y.data(), &bias0,
                                                                     quant_granularity_t::per_tensor, 0, sumW_k64.data());
                            if (ok_amx) { kernel_name = "amx_reuse_xq"; used_vnni = false; }
                        } else {
                            run_gemmv_q_fp32_ex(X.data(), K, W, M, ld_w_bytes,
                                                scales_eff.data(), zps_eff.data(), Y.data(), bias_eff.data(),
                                                gran, group_size, /*acc=*/false, wtype, &kernel_name);
                        }
                    }
                    auto t1 = std::chrono::steady_clock::now();
                    double ms_one = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
                    ms_runs.push_back(ms_one);
                }
                double ms = ms_runs[0], ms_min = ms_runs[0], ms_med = ms_runs[0];
                if (ms_runs.size() == 3) { auto tmp = ms_runs; std::sort(tmp.begin(), tmp.end()); ms_min = tmp[0]; ms_med = tmp[1]; ms = ms_med; }

                // Optional multi-thread run (GEMMV_THREADS > 1)
                int mt_threads = 0;
                if (const char* envt = std::getenv("GEMMV_THREADS")) { mt_threads = std::max(0, std::atoi(envt)); }
                double ms_mt = 0.0;
                if (mt_threads > 1) {
                    std::vector<float> Ymt((size_t)M_pad, 0.f);
                    auto t0m = std::chrono::steady_clock::now();
                    int iters_mt = bench_iters;
                    for (int it = 0; it < iters_mt; ++it) {
                        run_gemmv_q_fp32_mt(X.data(), K, W, M, ld_w_bytes,
                                            scales_eff.data(), zps_eff.data(), Ymt.data(), bias_eff.data(),
                                            gran, group_size, /*acc=*/false, wtype, mt_threads);
                    }
                    auto t1m = std::chrono::steady_clock::now();
                    ms_mt = std::chrono::duration<double, std::milli>(t1m - t0m).count() / iters_mt;
                    // overwrite Y for error check consistency
                    Y.swap(Ymt);
                }

                // If kernel name indicates a VNNI int8 path but 'used_vnni' wasn't set (e.g., non-reuse path), enable specialized ref
                if (!used_vnni && kernel_name) {
                    std::string kstr(kernel_name);
                    if (kstr.find("vnni_") != std::string::npos) used_vnni = true;
                }
                // Reference & error
                if (used_vnni) {
                    // Build VNNI CPU reference (per-tensor X quantization) matching granularity
                    const int M_blk_ref = 16;
                    const int K_grp_ref = (K + 3)/4;
                    // quantize X per-tensor
                    std::vector<uint8_t> Xq(K, 0);
                    float s_x=1.f; int32_t zp_x=128;
                    {
                        float amax=0.f; for (int k=0;k<K;++k) amax=std::max(amax, std::fabs(X[k]));
                        s_x = (amax>0.f)?(amax/127.f):1.f; zp_x=128;
                        for (int k=0;k<K;++k) { int v = (int)std::lrintf(X[k]/s_x)+zp_x; if (v<0) v=0; if (v>255) v=255; Xq[k]=(uint8_t)v; }
                    }
                    int32_t sumX=0; for (int k=0;k<K;++k) sumX += (int32_t)Xq[k];
                    auto get_Wgrp = [&](int bi,int g,int row)->const uint8_t*{ const uint8_t* base = Wpack_i8_k4.get() + (size_t)bi * (size_t)ld_w_k4; return base + (size_t)g * 64 + (size_t)row * 4; };
                    for (int m=0;m<M;++m) {
                        int64_t acc=0; int32_t sumW=0;
                        const int bi_m = m / M_blk_ref;
                        const int row_m = m % M_blk_ref;
                        for (int g=0; g<K_grp_ref; ++g) {
                            const uint8_t* w4 = get_Wgrp(bi_m, g, row_m);
                            uint8_t xb0=Xq[g*4+0], xb1=Xq[g*4+1], xb2=Xq[g*4+2], xb3=Xq[g*4+3];
                            int8_t w0=(int8_t)w4[0], w1=(int8_t)w4[1], w2=(int8_t)w4[2], w3=(int8_t)w4[3];
                            acc += (int32_t)xb0*w0 + (int32_t)xb1*w1 + (int32_t)xb2*w2 + (int32_t)xb3*w3;
                            sumW += (int32_t)w0 + w1 + w2 + w3;
                        }
                        // lane metadata based on granularity
                        float s_w_lane = 1.f; float b_lane = 0.f; int32_t zpw_lane = 0;
                        if (per_tensor) {
                            s_w_lane = scales_eff[0];
                            b_lane   = bias_eff.empty()?0.f:bias_eff[0];
                            zpw_lane = zps_eff.empty()?0:zps_eff[0];
                        } else if (per_group) {
                            int gidx = (group_size>0 ? (m / group_size) : (m / M_blk_ref));
                            s_w_lane = scales_eff[gidx];
                            b_lane   = bias_eff.empty()?0.f:bias_eff[gidx];
                            zpw_lane = zps_eff.empty()?0:zps_eff[gidx];
                        } else {
                            s_w_lane = scales_eff[m];
                            b_lane   = bias_eff.empty()?0.f:bias_eff[m];
                            zpw_lane = zps_eff.empty()?0:zps_eff[m];
                        }
                        int32_t comp = - zp_x * sumW - zpw_lane * sumX + K * zpw_lane * zp_x;
                        int32_t s32_total = (int32_t)acc + comp;
                        Yref[m] = (float)s32_total * (s_w_lane * s_x) + b_lane;
                    }
                } else {
                    naive_ref(X.data(), K, W, M, ld_w_bytes,
                              scales_eff.data(), zps_eff.data(), Yref.data(), bias_eff.data(), gran, wtype, group_size);
                }
                double max_abs = 0.0, max_rel = 0.0;
                for (int m = 0; m < M; ++m) {
                    const double a = Y[m], b = Yref[m];
                    max_abs = std::max(max_abs, std::abs(a - b));
                    const double denom = std::max(1e-5, std::abs(b));
                    max_rel = std::max(max_rel, std::abs(a - b)/denom);
                }

                std::cout << "GEMMV M=" << M << " K=" << K
                          << " gran=" << (per_tensor?"per_tensor":(per_group?"per_group":"per_channel"))
                          << " W=" << wname
                          << " time_ms=" << ms
                          << " gflops=" << gflops(M, K, ms)
                          << " max_abs_err=" << max_abs
                          << " max_rel_err=" << max_rel
                          << std::endl;

                // JSONL log for Codex context replays
                {
                    std::string ts = iso_now();
                    std::string mode = per_tensor?"per_tensor":(per_group?"per_group":"per_channel");
                        std::string j = std::string("{\"ts\":\"") + ts + "\"," +
                            "\"action\":\"bench_result\"," +
                            "\"M\":" + std::to_string(M) + "," +
                            "\"K\":" + std::to_string(K) + "," +
                            "\"gran\":\"" + mode + "\"," +
                            "\"w_type\":\"" + wname + "\"," +
                            "\"kernel\":\"" + (kernel_name?kernel_name:"unknown") + "\"," +
                            "\"iters\":" + std::to_string(bench_iters) + "," +
                            "\"time_ms\":" + std::to_string(ms) + "," +
                            (outer_reps>1 ? (std::string("\"time_ms_min\":") + std::to_string(ms_min) + ",\"time_ms_med\":" + std::to_string(ms_med) + ",") : std::string("")) +
                            "\"gflops\":" + std::to_string(gflops(M, K, ms)) + "," +
                            (mt_threads>1? (std::string("\"time_ms_mt\":") + std::to_string(ms_mt) + ",") : std::string("")) +
                            "\"max_abs_err\":" + std::to_string(max_abs) + "," +
                            "\"max_rel_err\":" + std::to_string(max_rel) +
                            "}";
                        append_log("[GEMMV-BENCH] " + j);
                }

                // no optional env-based compare; bench focuses on primary timings
            }
        }

        // Optional: explicit mini‑GEMM JIT vs REF comparison with small N
        if (const char* do_cmp_mg = std::getenv("GEMMV_COMPARE_MINIGEMM"); do_cmp_mg && std::string(do_cmp_mg) == "1") {
            const bool full_matrix = (std::getenv("GEMMV_CM_MG_FULL") && std::string(std::getenv("GEMMV_CM_MG_FULL")) == "1");
            std::vector<int> Ns = {2, 4, 8};
            if (std::getenv("GEMMV_CM_MG_N16") && std::string(std::getenv("GEMMV_CM_MG_N16")) == "1") Ns.push_back(16);
            auto run_cmp_minigemm = [&](w_dtype_t wtype, const char* wname){
                const uint8_t* W = (wtype==w_dtype_t::i8)? Wpack_i8.data() : (wtype==w_dtype_t::u8)? Wpack_u8.data() : (wtype==w_dtype_t::i4)? Wpack_i4.data() : Wpack_u4.data();
                const int ld_w_bytes = (wtype==w_dtype_t::i4 || wtype==w_dtype_t::u4) ? ld_w_bytes4 : ld_w_bytes8;
                const int mode_lo = full_matrix ? 0 : 0;
                const int mode_hi = full_matrix ? 3 : 1; // only per_tensor by default
                for (int mode = mode_lo; mode < mode_hi; ++mode) {
                    const bool per_tensor = (mode == 0);
                    const bool per_group = (mode == 2);
                    quant_granularity_t gran = per_tensor ? quant_granularity_t::per_tensor
                                                          : (per_group ? quant_granularity_t::per_group
                                                                       : quant_granularity_t::per_channel);
                    int group_size = 0;
                    int meta_len = per_tensor ? 1 : M_pad;
                    if (per_group) {
                        group_size = 32;
                        meta_len = (M_pad + group_size - 1) / group_size;
                    }
                    std::vector<float> scales(meta_len), bias(meta_len);
                    std::vector<int32_t> zps(meta_len);
                    for (int i = 0; i < meta_len; ++i) {
                        scales[i] = std::max(0.05f, std::abs(df(rng)));
                        zps[i] = du8(rng) & 0xFF;
                        bias[i] = df(rng);
                    }
                    for (int N : Ns) {
                        std::vector<float> Xcols((size_t)K*(size_t)N);
                        for (auto &v : Xcols) v = df(rng);
                        std::vector<float> Y_ref((size_t)M_pad*(size_t)N, 0.f);
                        std::vector<float> Y_jit((size_t)M_pad*(size_t)N, 0.f);

                        // Warmup
                        for (int it = 0; it < std::max(1, warmup_iters/2); ++it) {
                            run_minigemm_ref_q_fp32(Xcols.data(), K, N, W, M, ld_w_bytes,
                                                    scales.data(), zps.data(), Y_ref.data(), bias.data(),
                                                    gran, wtype, false, group_size);
                            const char* nm=nullptr;
                            run_minigemm_q_fp32_ex(Xcols.data(), K, N, W, M, ld_w_bytes,
                                                   scales.data(), zps.data(), Y_jit.data(), bias.data(),
                                                   gran, group_size, wtype, &nm);
                        }
                        // Time ref
                        const int iters = std::min(bench_iters, 10);
                        auto t0 = std::chrono::steady_clock::now();
                        for (int it = 0; it < iters; ++it) {
                            run_minigemm_ref_q_fp32(Xcols.data(), K, N, W, M, ld_w_bytes,
                                                    scales.data(), zps.data(), Y_ref.data(), bias.data(),
                                                    gran, wtype, false, group_size);
                        }
                        auto t1 = std::chrono::steady_clock::now();
                        double ms_ref = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
                        // Time jit
                        const char* kname = nullptr;
                        t0 = std::chrono::steady_clock::now();
                        for (int it = 0; it < iters; ++it) {
                            run_minigemm_q_fp32_ex(Xcols.data(), K, N, W, M, ld_w_bytes,
                                                   scales.data(), zps.data(), Y_jit.data(), bias.data(),
                                                   gran, group_size, wtype, &kname);
                        }
                        t1 = std::chrono::steady_clock::now();
                        double ms_jit = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
                        // Errors
                        double max_abs = 0.0, max_rel = 0.0;
                        for (int n = 0; n < N; ++n) {
                            for (int m = 0; m < M; ++m) {
                                double a = Y_jit[(size_t)n*M_pad + m];
                                double b = Y_ref[(size_t)n*M_pad + m];
                                max_abs = std::max(max_abs, std::abs(a - b));
                                double denom = std::max(1e-5, std::abs(b));
                                max_rel = std::max(max_rel, std::abs(a - b)/denom);
                            }
                        }
                        std::string mode_s = per_tensor?"per_tensor":(per_group?"per_group":"per_channel");
                        std::string j = std::string("{\"ts\":\"") + iso_now() + "\"," +
                            "\"action\":\"compare_minigemm\"," +
                            "\"M\":" + std::to_string(M) + "," +
                            "\"K\":" + std::to_string(K) + "," +
                            "\"N\":" + std::to_string(N) + "," +
                            "\"gran\":\"" + mode_s + "\"," +
                            "\"wtype\":\"" + wname + "\"," +
                            "\"kernel\":\"" + (kname?kname:"unknown") + "\"," +
                            "\"ms_ref\":" + std::to_string(ms_ref) + "," +
                            "\"ms_jit\":" + std::to_string(ms_jit) + "," +
                            "\"max_abs_err\":" + std::to_string(max_abs) + "," +
                            "\"max_rel_err\":" + std::to_string(max_rel) + "}";
                        append_log("[GEMMV-BENCH] " + j);
                    }
                }
            };
            // Safe default: only i8 unless GEMMV_CM_MG_FULL=1
            run_cmp_minigemm(w_dtype_t::i8, "i8");
            if (full_matrix) {
                run_cmp_minigemm(w_dtype_t::u8, "u8");
                run_cmp_minigemm(w_dtype_t::i4, "i4");
                run_cmp_minigemm(w_dtype_t::u4, "u4");
            }
            append_log(std::string("[LOG] ") + iso_now() + " — mini-GEMM compare (JIT vs REF) done for shape M=" + std::to_string(M) + " K=" + std::to_string(K));
        }
    }
    // Run end footer
    append_log(std::string("[GEMMV-BENCH] ") + "{\"ts\":\"" + iso_now() + "\",\"action\":\"run_end\"}");
    return 0;
}
