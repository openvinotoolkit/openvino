// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_mlp.h"

#include <chrono>
#include <string>
#include <vector>

#include "common/bfloat16.hpp"
#include "common/cpu_memcpy.h"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "shape_inference/shape_inference_internal_dyn.hpp"
#include "utils/plain_tensor.hpp"

#include "llm_mlp_utils.hpp"

struct ANSIcolor {
    const char* code;
    std::string str;
    ANSIcolor(const char* code = "0") : code(code) {
        std::stringstream ss;
        ss << "\033[" << code << "m";
        str = ss.str();
    }

    friend std::ostream& operator<<(std::ostream& out, const ANSIcolor& obj) {
        out << "\033[" << obj.code << "m";
        return out;
    }
};

inline int readenv(const char* name, int default_value, const char * description = nullptr) {
    int v = default_value;
    auto* p = std::getenv(name);
    if (p)
        v = strtol(p, NULL, 0);
    std::cout << ANSIcolor("32") << "ENV: " << name << " = " << v << "   (" << (description ? description : "...") << ")" << ANSIcolor() << std::endl;
    return v;
}

template<int id = 0>
inline float get_delta_ms() {
    static auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> dt = t1 - t0;
    t0 = t1;
    return dt.count();
}

static bool no_ecout = readenv("NOECOUT", 1, "disable ECOUT output when set to non-zero.");

template <typename... Ts>
void easy_cout(const char* file, const char* func, int line, Ts... args) {
    if (no_ecout) return;
    std::string file_path(file);
    std::string file_name(file);

    auto last_sep = file_path.find_last_of('/');
    if (last_sep == std::string::npos)
        last_sep = file_path.find_last_of('\\');
    if (last_sep != std::string::npos)
        file_name = file_path.substr(last_sep + 1);

    std::string file_name_with_line = file_name + ":" + std::to_string(line);
    auto tag = file_name_with_line + " " + func + "()";

    std::stringstream ss;
    int dummy[sizeof...(Ts)] = {(ss << args, 0)...};
    (void)dummy;
    std::cout << " \033[37;100m+" << std::fixed << std::setprecision(3) << get_delta_ms() << " ms\033[36;40m " << tag << " \033[0m " << ss.str() << "" << std::endl;
}

#define ECOUT(...) easy_cout(__FILE__, __func__, __LINE__, __VA_ARGS__)


namespace ov {
namespace intel_cpu {
namespace node {

namespace AMX_MLP {


using namespace dnnl::impl::cpu::x64;

struct TileConfig {
    uint8_t palette_id;
    uint8_t startRow;
    uint8_t reserved[14];
    uint16_t cols[16];
    uint8_t rows[16];
    void reset(int palette, int _startRow, const std::vector<std::pair<int, int>>& _rows_columnsBytes) {
        palette_id = palette;
        startRow = _startRow;
        unsigned long i;
        for (i = 0; i < 14; i++) {
            reserved[i] = 0;
        }
        for (i = 0; i < _rows_columnsBytes.size(); i++) {
            rows[i] = _rows_columnsBytes[i].first;
            cols[i] = _rows_columnsBytes[i].second;
        }
        for (; i < 16; i++) {
            cols[i] = 0;
            rows[i] = 0;
        }
    }
} __attribute__((__packed__));

class TileConfiger : public jit_generator {
public:
    TileConfiger() : jit_generator("TileConfiger") {
        create_kernel();
    }
    const char *name() const override { return "TileConfiger"; }
    const char *source_file() const override { return __FILE__; }

    void * last_cfg = nullptr;
    void do_config(void * cfg) {
        if (cfg != last_cfg) {
            (*this)(cfg);
            last_cfg = cfg;
        }
    }

    void generate() override {
        Xbyak::Label release;
        test(abi_param1, abi_param1);
        jz(release);
        ldtilecfg(ptr[abi_param1]);
        ret();
        L(release);
        tilerelease();
        ret();
    }
};

// https://stackoverflow.com/questions/23690416/c-template-singleton-static-pointer-initialization-in-header-file
template <typename T>
class Singleton {
public:
    static T& get() {
        static T instance;
        return instance;
    }
};

class MKernel : public jit_generator {
public:
    TileConfig m_tile_cfg;

    int m_prefetch_Blines;

    // both A & B data will be prefetched from memory for next kernel invokation
    // and the prefetches are evenly distributed into each kernel.
    //
    // we first tackle the prefetching of B, because each time
    // we will call it with a new B, and run() will have to prefetch new B
    // for next round, so next B is also of size (KxN) elements
    //    distributes into (BM/32)*(BN/32) kernels:
    //    each kernel has (BK/32) iterations, thus each kernel iteration
    //    need to prefetch (BKxBN)/(BMxBNxBK/32768) = 32768/BM bfloat16-elements
    //    which is 1024/BM cache lines, this has to be determined at
    //    code-generation time. with BM=256, this is only 4.
    //
    // prefetch A can be done in unit of 32xBK elements, which must be evenly distributed
    // into (BN/32)*(BK/32) kernel iterations, each iteration prefetch/copy 32xBK/(BN*BK/1024) = 32768/BN
    // bfloat16-elements or 1024/BN cache lines. with BM=256, this is only 4 too.
    //
    // prefetch or copy?
    //   prefetch strided sub-matrix of A is tricky, consider each 32x32 AMX jit kernel has [BK/32] iterations
    //   and it's called (BN/32) times, each kernel must prefetch 32*BK/(BN/32) = (1024/BN)*BK elements
    //   since each kernel has [BK/32] loop iterations, each iteration fetch (1024/BN)*BK/(BK/32) = 1024*32/BN
    //   bytes.
    //
    //   when 1024 is not divisible by BN, it's fine, just prefetch more
    //
    // copy data from A to a ping-pong buffer has advantage:
    //    - read can be done in continous way most suitable for HW prefetcher
    //    - write to ping-pong buffer is within L2 cache, which should be fast
    //    - data transfer rate is small comparing to L2-bandwidth, shouldn't be a big issue for interleaved write to L2.
    //    - read from ping-pong buffer is much faster and free of odd-multiple-cache-line restriction.
    // so we prefer distribute the repacking of A sub-matrix into ping-pong buffer into kernel.
    // for BN=256, each kernel read 4*BK elements into ping-pong, each iteration read
    // 4*BK*sizeof(bfloat16)/(BK/32)=256bytes = 4-512bits zmm registers
    //
    //
    MKernel(int M_hint = 256) : jit_generator("MKernel") {
        setup(M_hint);
    }

    //  M_hint is only a hint for prefetching, set to 0 to avoid prefetch
    void setup(int M_hint = 0) {
        if (M_hint == 0) {
            m_prefetch_Blines = 0;
        } else {
            m_prefetch_Blines = 32768 * sizeof(ov::bfloat16) / 64 / M_hint;
        }

        create_kernel();
        tile_config_M(m_tile_cfg, M_hint);
    }

    // M can change w/o code-regeneration
    // with the help of :
    //  - m_BM_hint controls dynamic behaviour of the kernel
    //  - tile config controls behaviour of tileload & TMUL
    void tile_config_M(TileConfig& tile_cfg, int M) {
        auto rows0 = 16;
        auto rows1 = 16;
        if (M < 32) {
            // kernel is for processing Mtails
            if (M > 16) {
                rows0 = 16;
                rows1 = M - 16;
            } else {
                //  both A0 & A1 load from same memory, to avoid code-regeneration
                rows0 = rows1 = M;
            }
        }
        tile_cfg.reset(1,
                       0,
                       {
                           {rows0, 64},  // C00:0
                           {rows0, 64},  // C01:1
                           {rows1, 64},  // C10:2
                           {rows1, 64},  // C11:3
                           {rows0, 64},  // A0:4
                           {rows1, 64},  // A1:5
                           {16, 64},     // B0:6
                           {16, 64},     // B1:7
                       });
    }

    // row data is in layout [N, K], maybe smaller than [32, 16]
    template <typename T>
    void repackB(ov::bfloat16* dst, T* src, int N_stride, int N, int K) {
        if (N == 16 && K == 32 && std::is_same<T, ov::bfloat16>::value) {
            // SIMD optimized version
            // std::cout << "." << std::flush;
            ov::Extensions::Cpu::XARCH::llm_mlp_transpose_epi32_16x16(dst, src, N_stride * sizeof(T));
            return;
        }

        assert(K <= 32);
        assert(N <= 16);
        int k = 0;
        ov::bfloat16 bf16zero(0.0f);
        for (; k < 32; k += 2) {
            int n = 0;
            bool is_k0_valid = (k) < K;
            bool is_k1_valid = (k + 1) < K;
            auto* psrc = src + k;
            for (; n < 16 && n < N; n++, psrc += N_stride) {
                *dst++ = is_k0_valid ? ov::bfloat16(psrc[0]) : bf16zero;
                *dst++ = is_k1_valid ? ov::bfloat16(psrc[1]) : bf16zero;
            }
            for (; n < 16; n++) {
                *dst++ = 0;
                *dst++ = 0;
            }
        }
    }

    // weight is supposed to be of shape[N, K], stride in unit of bytes
    // N should be m_BN
    // K should be m_BK
    template <typename T>
    void prepareB(PlainTensor& ret, T* p_weight, int stride, int N, int K) {
        OPENVINO_ASSERT((N % 32) == 0);
        OPENVINO_ASSERT((K % 32) == 0);
        // weight matrix is in unit of [N/32, Kx32]
        ret.resize<ov::bfloat16>({N / 32, K * 32});

        auto N_stride = stride / sizeof(T);
        for (int n = 0, blkn = 0; n < N; n += 32, blkn++) {
            for (int k = 0, blkk = 0; k < K; k += 32, blkk++) {
                // two adjacent 32x16 (512) block of weight: dst0 & dst1
                auto* dst0 = ret.ptr<ov::bfloat16>(blkn, blkk * 1024);
                auto* dst1 = dst0 + 16 * 32;
                auto valid_k = (K - k) < 32 ? (K - k) : 32;

                auto* src0 = p_weight + n * N_stride + k;
                auto valid_n0 = (N - n) < 16 ? (N - n) : 16;
                repackB<T>(dst0, src0, N_stride, valid_n0, valid_k);

                auto* src1 = p_weight + (n + 16) * N_stride + k;
                auto valid_n1 = (N - (n + 16)) < 16 ? (N - (n + 16)) : 16;
                repackB<T>(dst1, src1, N_stride, valid_n1, valid_k);
            }
        }
    }

    // to save push/pop: do not use `abi_save_gpr_regs`
    uint8_t* prefetch_next_A_addr;

    struct call_args {
        const uint8_t* pA;  // bfloat16
        int64_t strideA;    // in bytes
        const uint8_t* pB;  // bfloat16
        const uint8_t* pC;  // float32
        int64_t strideC;    // in bytes
        const uint8_t* prefetch;
        int64_t k_tiles;  // K / 32
        int64_t do_accumulation;
        int64_t M;
    };
    const char *name() const override { return "MKernel"; }
    const char *source_file() const override { return __FILE__; }

    void generate() override {
        Xbyak::Reg64 reg_A_addr = abi_param2;
        Xbyak::Reg64 reg_A_stride = abi_param3;
        Xbyak::Reg64 reg_B_addr = abi_param4;
        Xbyak::Reg64 reg_C_addr = abi_param5;
        Xbyak::Reg64 reg_C_stride = abi_param6;

        Xbyak::Reg64 reg_ktiles = rax;
        Xbyak::Reg64 reg_B_stride = r10;
        Xbyak::Reg64 reg_A1_addr = r11;
        Xbyak::Reg64 reg_prefetch = r12;

        Xbyak::Tmm tmmC00 = tmm0;
        Xbyak::Tmm tmmC01 = tmm1;
        Xbyak::Tmm tmmC10 = tmm2;
        Xbyak::Tmm tmmC11 = tmm3;
        Xbyak::Tmm tmmA0 = tmm4;
        Xbyak::Tmm tmmA1 = tmm5;
        Xbyak::Tmm tmmB0 = tmm6;
        Xbyak::Tmm tmmB1 = tmm7;

        auto num_PFB = m_prefetch_Blines;
        int cur_PFB = 0;
        /*
                       B: 1x2 tiles
        A : 2x1 tiles  C: 2x2 tiles
        */
        Xbyak::Label loop_over_ktiles;
        Xbyak::Label skip_load;

        push(reg_prefetch);
        {
            auto reg_tmp = reg_B_stride;
            tilezero(tmmC00);
            tilezero(tmmC01);
            tilezero(tmmC10);
            tilezero(tmmC11);

            mov(reg_A_addr, ptr[abi_param1 + offsetof(call_args, pA)]);
            mov(reg_A_stride, ptr[abi_param1 + offsetof(call_args, strideA)]);
            mov(reg_B_addr, ptr[abi_param1 + offsetof(call_args, pB)]);
            mov(reg_C_addr, ptr[abi_param1 + offsetof(call_args, pC)]);
            mov(reg_C_stride, ptr[abi_param1 + offsetof(call_args, strideC)]);
            mov(reg_prefetch, ptr[abi_param1 + offsetof(call_args, prefetch)]);
            mov(reg_ktiles, ptr[abi_param1 + offsetof(call_args, k_tiles)]);

            lea(reg_A1_addr, ptr[reg_A_addr + reg_A_stride * 8]);
            lea(reg_A1_addr, ptr[reg_A1_addr + reg_A_stride * 8]);

            // reg_A1_addr = reg_A_addr if M <= 16 (to avoid tileloadd segmentfault)
            mov(reg_tmp, ptr[abi_param1 + offsetof(call_args, M)]);
            cmp(reg_tmp, 16);
            cmovle(reg_A1_addr, reg_A_addr);

            mov(reg_tmp, ptr[abi_param1 + offsetof(call_args, do_accumulation)]);
            and_(reg_tmp, 1);
            jz(skip_load);
            {
                auto reg_C1_addr = reg_tmp;
                tileloadd(tmmC00, ptr[reg_C_addr + reg_C_stride]);
                tileloadd(tmmC01, ptr[reg_C_addr + reg_C_stride + 64]);
                lea(reg_C1_addr, ptr[reg_C_addr + reg_C_stride * 8]);
                lea(reg_C1_addr, ptr[reg_C1_addr + reg_C_stride * 8]);
                tileloadd(tmmC10, ptr[reg_C1_addr + reg_C_stride]);
                tileloadd(tmmC11, ptr[reg_C1_addr + reg_C_stride + 64]);
            }
            L(skip_load);
        }

        mov(reg_B_stride, 64);

        auto const_A_steps = 64;

        align(64, false);
        L(loop_over_ktiles);
        // for (int k = 0; k < Ktiles; k++) {
        tileloadd(tmmA0, ptr[reg_A_addr + reg_A_stride]);
        tileloadd(tmmB0, ptr[reg_B_addr + reg_B_stride]);
        lea(reg_B_addr, ptr[reg_B_addr + 1024]);

        tdpbf16ps(tmmC00, tmmA0, tmmB0);
        if (cur_PFB < num_PFB) {
            prefetcht2(ptr[reg_prefetch + cur_PFB * 64]);
            cur_PFB++;
        }

        tileloadd(tmmA1, ptr[reg_A1_addr + reg_A_stride]);
        tdpbf16ps(tmmC10, tmmA1, tmmB0);
        if (cur_PFB < num_PFB) {
            prefetcht2(ptr[reg_prefetch + cur_PFB * 64]);
            cur_PFB++;
        }

        tileloadd(tmmB1, ptr[reg_B_addr + reg_B_stride]);

        // prefetch [K_tiles X 256] bytes

        tdpbf16ps(tmmC01, tmmA0, tmmB1);
        if (cur_PFB < num_PFB) {
            prefetcht2(ptr[reg_prefetch + cur_PFB * 64]);
            cur_PFB++;
        }

        tdpbf16ps(tmmC11, tmmA1, tmmB1);
        // prefetch next sub-block B matrix
        if (cur_PFB < num_PFB) {
            for (int pi = cur_PFB; pi < num_PFB; pi++) {
                prefetcht2(ptr[reg_prefetch + pi * 64]);
            }
        }

        lea(reg_prefetch, ptr[reg_prefetch + 64 * num_PFB]);

        //}
        lea(reg_A_addr, ptr[reg_A_addr + const_A_steps]);
        lea(reg_A1_addr, ptr[reg_A1_addr + const_A_steps]);
        lea(reg_B_addr, ptr[reg_B_addr + 1024]);
        dec(reg_ktiles);
        jnz(loop_over_ktiles, T_NEAR);

#if 0
        tilestored(ptr[reg_C_addr + reg_B_stride], tmmC00);
        tilestored(ptr[reg_C_addr + reg_B_stride + 1024], tmmC01);
        tilestored(ptr[reg_C_addr + reg_B_stride + 1024 * 2], tmmC10);
        tilestored(ptr[reg_C_addr + reg_B_stride + 1024 * 3], tmmC11);
#else
        tilestored(ptr[reg_C_addr + reg_C_stride], tmmC00);
        tilestored(ptr[reg_C_addr + reg_C_stride + 64], tmmC01);
        lea(reg_C_addr, ptr[reg_C_addr + reg_C_stride * 8]);
        lea(reg_C_addr, ptr[reg_C_addr + reg_C_stride * 8]);
        tilestored(ptr[reg_C_addr + reg_C_stride], tmmC10);
        tilestored(ptr[reg_C_addr + reg_C_stride + 64], tmmC11);
#endif
        pop(reg_prefetch);
        ret();
    }

    // run L2 cache blocking kernel with size:
    //    [BM, BK]*[BK, BN] => [BM, BN]
    //
    // prefetch of A can be done inside of this level of kernel
    // since it's done in unit of 32-rows
    // but prefetch of next B must be specified by caller.
    //
    void run(int M,  // actual M
             uint8_t* pA,
             int strideA,              // A [M, K]
             PlainTensor& repacked_B,  // B [N/32, K*32] ov::bfloat16
             uint8_t* pC,
             int strideC,              // C [M, N]
             uint8_t* prefetch_B,  // prefetch B
             bool do_accumulation) {
        call_args args;
        // number of blocks in N dimension (in unit of 32 columns)
        auto num_blkN = repacked_B.size(0);
        auto K = repacked_B.size(1) / 32;
        auto* pB = repacked_B.ptr<uint8_t>();
        auto strideB = repacked_B.stride_bytes(0);

        args.do_accumulation = do_accumulation;
        args.k_tiles = K / 32;
        args.strideA = strideA;
        args.strideC = strideC;
        args.prefetch = prefetch_B;
        assert((K % 32) == 0);

        auto prefetch_step = m_prefetch_Blines * 64 * args.k_tiles;

        // if (BM != m_BM_hint) it only effect prefetch of B which is not vital to function
        for (int m = 0; m < M; m += 32, pA += 32 * strideA, pC += 32 * strideC) {
            args.pB = pB;
            // prefetch_next_A_addr = pA + 32 * strideA;
            // if (m + 32 >= BM)
            //     prefetch_next_A_addr = pA;
            args.M = std::min(M - m, 32);
            args.pA = pA;
            for (int ni = 0; ni < num_blkN; ni++, args.pB += strideB, args.prefetch += prefetch_step) {
                args.pC = pC + ni * 32 * sizeof(float);
                (*this)(&args);
                //(*this)(pA, strideA, pB1, pC + ni * 32 * sizeof(float), strideC, prefetch_B);
                // prefetch_next_A_addr += 4 * strideA;
            }
        }
    }
};


class jit_base : public jit_generator {
public:
    const char * m_name;
    jit_base(const char * name) : jit_generator(name), m_name(name) {}
    const char *name() const override { return m_name; }
    const char *source_file() const override { return __FILE__; }

    #define Vx16(a) {a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a}
    struct _const_table {
        uint32_t exp_log2ef[16] = Vx16(0x3fb8aa3b);
        uint32_t exp_ln_flt_max_f[16] = Vx16(0x42b17218);
        uint32_t exp_ln_flt_min_f[16] = Vx16(0xc2aeac50);
        uint32_t ln2f[16] = Vx16(0x3f317218); // ln(2.0f)
        uint32_t exponent_bias[16] = Vx16(0x0000007f);
        float two[16] = Vx16(2.0f);
        float half[16] = Vx16(0.5f);
        float one[16] = Vx16(1.0f);   // p0=1.0f
        uint32_t exp_pol0[16] = Vx16(0x3f7ffffb);// p1 = 0.999999701f
        uint32_t exp_pol1[16] = Vx16(0x3efffee3);// p2 = 0.499991506f
        uint32_t exp_pol2[16] = Vx16(0x3e2aad40);// p3 = 0.166676521f
        uint32_t exp_pol3[16] = Vx16(0x3d2b9d0d);// p4 = 0.0418978221f
        uint32_t exp_pol4[16] = Vx16(0x3c07cfce);// p5 = 0.00828929059f
        uint32_t sign_bit[16] = Vx16(0x80000000);
    } const_table;
    static constexpr int n_mantissa_bits = 23;
    #define PTR_CONST(name) ptr[p_table + offsetof(_const_table, name)]

    enum {
        _cmp_eq_oq = 0u,
        _cmp_lt_os = 1u,
        _cmp_le_os = 2u,
        _cmp_neq_uq = 4u,
        _cmp_nlt_us = 5u,
        _cmp_nle_us = 6u,

        _op_floor = 1u,
        _op_mxcsr = 4u,
    };

    void inject_exp(const Xbyak::Zmm &vmm_src, const Xbyak::Zmm &vmm_aux1, const Xbyak::Zmm &vmm_aux2, Xbyak::Reg64 p_table, Xbyak::Opmask k_mask) {
        // exp(x) =
        // = exp(n * ln(2) + r) // divide x by ln(2) and get quot and rem
        // = 2^n * exp(r) // simplify the exp(n*ln(2)) expression

        // get mask of values lower than log(FLT_MIN) to zero them in the output
        //compute_cmp_mask(vmm_src, table_val(exp_ln_flt_min_f), _cmp_lt_os);
        vcmpps(k_mask, vmm_src, PTR_CONST(exp_ln_flt_min_f), _cmp_lt_os);

        vminps(vmm_src, vmm_src, PTR_CONST(exp_ln_flt_max_f));
        vmaxps(vmm_src, vmm_src, PTR_CONST(exp_ln_flt_min_f));
        vmovups(vmm_aux1, vmm_src);

        // calculate exp(x)
        // fx = x * log2ef + 0.5
        vmulps(vmm_src, vmm_src, PTR_CONST(exp_log2ef));
        vaddps(vmm_src, vmm_src, PTR_CONST(half));

        // tmp = floorf(fx)
        //vroundps(vmm_aux2, vmm_src, _op_floor);
        vrndscaleps(vmm_aux2, vmm_src, _op_floor & 0x3);

        // keep vmm_src = fx for further computations
        vmovups(vmm_src, vmm_aux2);

        // x = x - fx * ln2
        vfnmadd231ps(vmm_aux1, vmm_aux2, PTR_CONST(ln2f));

        // We do not count 2^n here, because n can reach 128 and 2^128 is not
        // representable by fp32, so to get around this problem, instead of computing
        // 2^n * exp(r) will be counted 2*2^(n-1)*exp(r), because 2^127
        // and 2 are numbers representable in fp32.

        // compute 2^(n-1)
        vsubps(vmm_src, vmm_src, PTR_CONST(one));
        vcvtps2dq(vmm_aux2, vmm_src);
        vpaddd(vmm_aux2, vmm_aux2, PTR_CONST(exponent_bias));

        vpslld(vmm_aux2, vmm_aux2, n_mantissa_bits);
        // use vmm_src as tmp vmm_zero when applying mask
        vxorps(vmm_src, vmm_src, vmm_src);
        // set zeroes at those points which were < log(FLT_MIN)
        //blend_with_mask(vmm_aux2, vmm_src);
        vblendmps(vmm_aux2 | k_mask, vmm_aux2, vmm_src);
        //vblendvps(vmm_aux2, vmm_aux2, vmm_src, vmm_mask);

        // compute polynomial
        vmovups(vmm_src, PTR_CONST(exp_pol4));
        vfmadd213ps(vmm_src, vmm_aux1, PTR_CONST(exp_pol3));
        vfmadd213ps(vmm_src, vmm_aux1, PTR_CONST(exp_pol2));
        vfmadd213ps(vmm_src, vmm_aux1, PTR_CONST(exp_pol1));
        vfmadd213ps(vmm_src, vmm_aux1, PTR_CONST(exp_pol0));
        vfmadd213ps(vmm_src, vmm_aux1, PTR_CONST(one));
        // y = y * 2^n
        vmulps(vmm_src, vmm_src, vmm_aux2);
        vmulps(vmm_src, vmm_src, PTR_CONST(two));
    }

    void inject_sigmoid(const Xbyak::Zmm &vmm_sigmoid, const Xbyak::Zmm &vmm_src, const Xbyak::Zmm &vmm_aux1, const Xbyak::Zmm &vmm_aux2, Xbyak::Reg64 p_table, Xbyak::Opmask k_mask) {
        // sigmoid(x) = 1/(1+log(-x))
        vpxord(vmm_sigmoid, vmm_src, PTR_CONST(sign_bit)); // -x
        inject_exp(vmm_sigmoid, vmm_aux1, vmm_aux2, p_table, k_mask);  // log(-x)
        vaddps(vmm_sigmoid, vmm_sigmoid, PTR_CONST(one));  // 1.0f + log(-x)
        vrcp14ps(vmm_sigmoid, vmm_sigmoid);
    }

    void inject_silu(const Xbyak::Zmm &vmm_silu, const Xbyak::Zmm &vmm_src, const Xbyak::Zmm &vmm_aux1, const Xbyak::Zmm &vmm_aux2, Xbyak::Reg64 p_table, Xbyak::Opmask k_mask) {
        // silu(x) = x * sigmoid(x)
        inject_sigmoid(vmm_silu, vmm_src, vmm_aux1, vmm_aux2, p_table, k_mask);
        vmulps(vmm_silu, vmm_src, vmm_silu);
    }

    void inject_init(Xbyak::Reg64 p_table) {
        mov(p_table, reinterpret_cast<uintptr_t>(&const_table));        
    }
};

class GateUpCombine : public jit_base {
public:
    GateUpCombine() : jit_base("GateUpCombine") {
        create_kernel();
    }

    void generate() override {
        Xbyak::Label loop_begin;
        
        Xbyak::Reg64 src = abi_param1;
        Xbyak::Reg64 dst = abi_param2;
        Xbyak::Reg64 prefetch_dst = abi_param3;
        Xbyak::Reg64 BN = abi_param4;

        Xbyak::Reg64 loop_i = rax;
        Xbyak::Reg64 p_table = r10;
        const auto zmm_gate = zmm0;
        const auto zmm_up = zmm1;
        const auto zmm_aux1 = zmm2;
        const auto zmm_aux2 = zmm3;
        const auto zmm_silu = zmm4;
        const auto ymm_dst = ymm4;

        xor_(loop_i, loop_i);
        inject_init(p_table);

        shr(BN, 1); // BN = BN/2;
        align(64, false);
        L(loop_begin);
        {
            vmovups(zmm_gate, ptr[src + loop_i*8]);
            vmovups(zmm_up, ptr[src + loop_i*8 + 16*4]);
            inject_silu(zmm_silu, zmm_gate, zmm_aux1, zmm_aux2, p_table, k1);
            vmulps(zmm_up, zmm_up, zmm_silu);
            vcvtneps2bf16(ymm_dst, zmm_up);
            prefetchwt1(ptr[prefetch_dst + loop_i*2]);
            vmovdqu(ptr[dst + loop_i*2], ymm_dst);
        }
        add(loop_i, 16);
        cmp(loop_i, BN);
        jl(loop_begin, T_NEAR);

        ret();
    }

#if 0
    // m_do_reduce2 : false
    void operator()(float* src, ov::bfloat16* dst, ov::bfloat16* prefetch_dst, int BN) {
        for (int n = 0, i = 0; n < BN; n += 32, i += 16) {
            auto v_gate = _mm512_loadu_ps(src + n);
            auto v_up = _mm512_loadu_ps(src + n + 16);
            v_gate = silu_ps_avx512(v_gate);
            v_up = _mm512_mul_ps(v_gate, v_up);
            auto v_bh = _mm512_cvtneps_pbh(v_up);
            // Greate Optimization:
            //  following prefetchnta prevents L2 HW prefetcher prefetch interleaved
            //  channels belonging to other cores which will causes too much cross-core cache coherent cost.
            _mm_prefetch(prefetch_dst + i, _MM_HINT_ET1);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), reinterpret_cast<__m256i&>(v_bh));
        }
    }
#endif
};

class ReduceAdd2bh : public jit_base {
public:
    bool m_do_reduce2;
    ReduceAdd2bh(bool do_reduce2) : jit_base("ReduceAdd2bh"), m_do_reduce2(do_reduce2) {
        create_kernel();
    }

#if 0
    // m_do_reduce2 : false
    void operator()(float* src0, ov::bfloat16* dst, ov::bfloat16* prefetch_dst, int BN) {
        for (int n = 0; n < work.BN; n += 32) {
            auto d0 = _mm512_loadu_ps(src + n);
            auto d1 = _mm512_loadu_ps(src + n + 16);
            auto v_bh = _mm512_cvtne2ps_pbh(d1, d0);
            _mm_prefetch(prefetch_dst + n, _MM_HINT_ET1);
            _mm512_storeu_ps(dst + n, reinterpret_cast<__m512&>(v_bh));
        }
    }
    // m_do_reduce2 : true
    void operator()(float* src0, float* src1, ov::bfloat16* dst, ov::bfloat16* prefetch_dst, int BN) {
        for (int n = 0; n < BN; n += 32) {
            auto d0 = _mm512_loadu_ps(src0 + n);
            auto d0b = _mm512_loadu_ps(src1 + n);
            auto d1 = _mm512_loadu_ps(src0 + n + 16);
            auto d1b = _mm512_loadu_ps(src1 + n + 16);
            d0 = _mm512_add_ps(d0, d0b);
            d1 = _mm512_add_ps(d1, d1b);
            auto v_bh = _mm512_cvtne2ps_pbh(d1, d0);
            _mm_prefetch(prefetch_dst + n, _MM_HINT_ET1);
            _mm512_storeu_ps(dst + n, reinterpret_cast<__m512&>(v_bh));
        }        
    }
#endif

    void generate() override {
        if (m_do_reduce2) {
            Xbyak::Reg64 src0 = abi_param1;
            Xbyak::Reg64 src1 = abi_param2;
            Xbyak::Reg64 dst = abi_param3;
            Xbyak::Reg64 prefetch_dst = abi_param4;
            Xbyak::Reg64 BN = abi_param5;
            Xbyak::Reg64 loop_i = rax;

            Xbyak::Label loop_begin;

            xor_(loop_i, loop_i);

            align(64, false);
            L(loop_begin);
            {
                vmovups(zmm0, ptr[src0 + loop_i*4]);
                vmovups(zmm1, ptr[src1 + loop_i*4]);
                vmovups(zmm2, ptr[src0 + loop_i*4 + 16*4]);
                vmovups(zmm3, ptr[src1 + loop_i*4 + 16*4]);
                vaddps(zmm0, zmm0, zmm1);
                vaddps(zmm2, zmm2, zmm3);
                vcvtne2ps2bf16(zmm4, zmm2, zmm0);
                prefetchwt1(ptr[prefetch_dst + loop_i*2]);
                vmovups(ptr[dst + loop_i*2], zmm4);
            }
            add(loop_i, 32);
            cmp(loop_i, BN);
            jl(loop_begin, T_NEAR);

            ret();
        } else {
            Xbyak::Reg64 src0 = abi_param1;
            Xbyak::Reg64 dst = abi_param2;
            Xbyak::Reg64 prefetch_dst = abi_param3;
            Xbyak::Reg64 BN = abi_param4;
            Xbyak::Reg64 loop_i = rax;

            Xbyak::Label loop_begin;

            xor_(loop_i, loop_i);

            align(64, false);
            L(loop_begin);
            {
                vmovups(zmm0, ptr[src0 + loop_i*4]);
                vmovups(zmm2, ptr[src0 + loop_i*4 + 16*4]);
                vcvtne2ps2bf16(zmm4, zmm2, zmm0);
                prefetchwt1(ptr[prefetch_dst + loop_i*2]);
                vmovups(ptr[dst + loop_i*2], zmm4);
            }
            add(loop_i, 32);
            cmp(loop_i, BN);
            jl(loop_begin, T_NEAR);

            ret();
        }
    }

};

static PlainTensor& getC(int ithr) {
    static std::vector<PlainTensor> all_C(parallel_get_max_threads());
    return all_C[ithr];
}

class Linear {
public:
    struct Work {
        MKernel* p_jit_amx0 = nullptr;
        GateUpCombine* p_jit_gateup = nullptr;
        ReduceAdd2bh* p_jit_reduce2bh_1 = nullptr;
        ReduceAdd2bh* p_jit_reduce2bh_2 = nullptr;

        std::vector<PlainTensor> weights;   // ov::bfloat16 weights for current thread

        std::shared_ptr<std::atomic_int> sync_flag;
        int n0 = 0;
        int n1 = 0;
        int k0 = 0;
        int k1 = 0;
        int BN = 0;
        int blk_K_size = 0;
        int ithr = 0;
        operator bool() {
            return BN > 0;
        }

        // input : weight [N, K], setup repacks range of N [n_start, n_end)
        template <typename T>
        void setup(T* p_weight, int stride) {
            auto num_blk_K = (k1 - k0) / blk_K_size;
            auto* pw = p_weight + n0 * stride / sizeof(T) + k0;

            weights.resize(num_blk_K);
            for (int k = 0; k < num_blk_K; k++) {
                p_jit_amx0->prepareB(weights[k], pw + k * blk_K_size, stride, BN, blk_K_size);
                //printf("---------- weight at %p\n", &weights[k][0]);
            }

            for (int Mtails = 0; Mtails < 32; Mtails++) {
                p_jit_amx0->tile_config_M(m_tcfg[Mtails], Mtails == 0? 32 : Mtails);
            }
        }

        TileConfig m_tcfg[32];

        void run(int M, uint8_t* pA, int strideA, PlainTensor& C) {
            int num_blk_K = (k1 - k0) / blk_K_size;

            auto Mtails = M % 32;
            auto Mbody = M - Mtails;

            auto C_M = Mbody + (Mtails ? 32 : 0);
            //if (C.dims[0] < C_M)
            C.resize<float>({C_M, BN});
            auto pC = reinterpret_cast<uint8_t*>(C.ptr_v());

            auto& tile_configer = (Singleton<TileConfiger>::get());

            pA += k0 * sizeof(ov::bfloat16);
            bool do_accumulation = false;

            //size_t strideB = blk_K_size * 32 * sizeof(ov::bfloat16);// repacked_B.stride;
            //int num_blkN = BN/32;  // repacked_B.dims[0];     BN/32
            //int BK = (k1 - k0);

            for (int ki = 0; ki < num_blk_K; ki++) {
                PlainTensor& blockB = weights[ki];
                PlainTensor& blockB1 = weights[(ki + 1) < num_blk_K ? (ki + 1) : ki];
                if (Mbody) {
                    tile_configer.do_config(&m_tcfg[0]);
                    p_jit_amx0->run(Mbody,
                                    pA + ki * blk_K_size * sizeof(ov::bfloat16),
                                    strideA,
                                    blockB,
                                    pC,
                                    C.stride_bytes(0),
                                    reinterpret_cast<uint8_t*>(blockB1.ptr_v()),
                                    do_accumulation);
                }

                if (Mtails) {
                    tile_configer.do_config(&m_tcfg[Mtails]);
                    p_jit_amx0->run(Mtails,
                                    pA + ki * blk_K_size * sizeof(ov::bfloat16) + Mbody * strideA,
                                    strideA,
                                    blockB,
                                    pC + Mbody * C.stride_bytes(0),
                                    C.stride_bytes(0),
                                    reinterpret_cast<uint8_t*>(blockB1.ptr_v()),
                                    do_accumulation);
                }
                do_accumulation = true;
            }
            tile_configer.do_config(nullptr);
        }
    };
    std::vector<Work> works;

    int used_nthr = 0;
    bool do_splitK = false;

    Linear() {}

    // weight [N, K]
    // Gate & Up are interleaved in N dimension: 16-gate / 16-up
    // and post-ops will compute  silu(gate)*up in unit of 16 elements
    // and store out as bfloat16.
    template <typename T, int BM = 256>
    void setup(T* p_weight, int stride, int N, int K, bool _do_splitK = false) {
        static MKernel jit_amx0(BM);
        static GateUpCombine jit_gateup;
        static ReduceAdd2bh jit_reduce2bh_1(false);
        static ReduceAdd2bh jit_reduce2bh_2(true);
        const int blk_K_size = 256;
        // prepare weights, split N among threads
        // in unit of 32
        OPENVINO_ASSERT((N % 32) == 0);
        OPENVINO_ASSERT((K % blk_K_size) == 0);
        auto nthr = parallel_get_max_threads();
        auto num_blk_N = N / 32;
        auto num_blk_K = K / blk_K_size;
        works.resize(nthr);

        do_splitK = _do_splitK;
        auto K_splits = do_splitK ? 2 : 1;
        // every thread should do same amount of work, and some cores can be idle
        auto valid_nthr = nthr / K_splits;
        auto blkN_per_thread = (num_blk_N + valid_nthr - 1) / valid_nthr;
        auto blkN_leftover = 0;
        auto start_blkN = 0;
        used_nthr = 0;
        auto blkK_per_thread = (num_blk_K + K_splits - 1) / K_splits;

        // split task on more cores is better on TBB
        blkN_per_thread = (num_blk_N) / valid_nthr;
        blkN_leftover = num_blk_N - (blkN_per_thread*valid_nthr);

        for (int ithr = 0; ithr < nthr; ithr += K_splits) {
            auto blkN = std::min(num_blk_N - start_blkN, blkN_per_thread);
            if (blkN_leftover > 0) {
                blkN_leftover--;
                blkN++;
            }
            if (blkN) {
                auto shared_atomic = std::make_shared<std::atomic_int>(0);
                auto start_blkK = 0;
                for (int ik = 0; ik < K_splits; ik++) {
                    auto blk_K = std::min(num_blk_K - start_blkK, blkK_per_thread);

                    auto& work = works[ithr + ik];

                    work.ithr = ithr + ik;
                    work.p_jit_amx0 = &jit_amx0;
                    work.p_jit_gateup = &jit_gateup;
                    work.p_jit_reduce2bh_1 = &jit_reduce2bh_1;
                    work.p_jit_reduce2bh_2 = &jit_reduce2bh_2;
                    work.sync_flag = shared_atomic;
                    work.blk_K_size = blk_K_size;

                    work.n0 = (start_blkN)*32;
                    work.n1 = (start_blkN + blkN) * 32;
                    work.BN = blkN * 32;
                    work.k0 = start_blkK * blk_K_size;
                    work.k1 = (start_blkK + blk_K) * blk_K_size;

                    start_blkK += blk_K;
                    used_nthr++;
                }
            }

            start_blkN += blkN;
        }

        ECOUT("Linear N,K=", N, ",", K, " used_nthr=", used_nthr, "  do_splitK=", do_splitK);

        ov::parallel_nt_static(0, [&](const size_t ithr, const size_t nthr) {
            auto& work = works[ithr];
            if (work) {
                work.setup(p_weight, stride);
            }
        });
        ECOUT("   setup is done. weight @ ", static_cast<void*>(p_weight));
    }

    /*
    // A bfloat16 [256,  num_blk_K * 256]
    void run(uint8_t* pA, int strideA, int M) {
#pragma omp parallel
        {
            int ithr = omp_get_thread_num();
            auto& work = works[ithr];
            if (work) {
                work.run(M, pA, strideA);
            }
        }
    }

        void run(uint8_t* pA, int strideA, int M, float* dstC, int strideC) {
    #pragma omp parallel
            {
                int ithr = omp_get_thread_num();
                auto& work = works[ithr];
                if (work) {
                    work.run(M, pA, strideA);
                    auto* src = &work.C[0];
                    auto* dst = dstC + work.n0;
                    auto strideS = work.C.stride / sizeof(*src);
                    auto strideD = strideC / sizeof(*dst);
                    for (int m = 0; m < M; m++, src += strideS, dst += strideD) {
                        // the prefetch distance is increased to ensure by the time store happens
                        // prefetch has done and no HW prefetcher is triggered
                        auto* prefetch_dst = (m + 2 < M) ? (dst + 2 * strideD) : (dst);
                        for (int n = 0; n < work.BN; n += 32) {
                            auto d0 = _mm512_loadu_ps(src + n);
                            auto d1 = _mm512_loadu_ps(src + n + 16);
                            _mm_prefetch(prefetch_dst + n, _MM_HINT_ET1);
                            _mm_prefetch(prefetch_dst + n + 16, _MM_HINT_ET1);
                            _mm512_storeu_ps(dst + n, d0);
                            _mm512_storeu_ps(dst + n + 16, d1);
                        }
                    }
                }
            }
        }
    */

    void run(uint8_t* pA, int strideA, int M, ov::bfloat16* dstC, int strideC) {

        ov::parallel_nt_static(0, [&](const size_t ithr, const size_t nthr) {
            auto& work = works[ithr];
            auto& workC = getC(ithr);
            if (work) {
                work.run(M, pA, strideA, workC);

                if (do_splitK) {
                    auto sync_id = work.sync_flag->fetch_add(1);
                    // (0,1) (2,3)
                    if (sync_id & 1) {
                        auto peer_ithr = (ithr & 1) ? (ithr - 1) : (ithr + 1);
                        auto& peer = works[peer_ithr];
                        auto& peerC = getC(peer_ithr);
                        // the other one has finished, we can do the reduce sum
                        auto* src0 = workC.ptr<float>();
                        auto* src1 = peerC.ptr<float>();
                        auto* dst = dstC + work.n0;
                        auto strideS = workC.stride(0);
                        auto strideD = strideC / sizeof(*dst);
                        for (int m = 0; m < M; m++, src0 += strideS, src1 += strideS, dst += strideD) {
                            // the prefetch distance is increased to ensure by the time store happens
                            // prefetch has done and no HW prefetcher is triggered
                            auto* prefetch_dst = (m + 2 < M) ? (dst + 2 * strideD) : (dst);
                            (*work.p_jit_reduce2bh_2)(src0, src1, dst, prefetch_dst, work.BN);
                        }
                    }
                } else {
                    auto* src = workC.ptr<float>();
                    auto* dst = dstC + work.n0;
                    auto strideS = workC.stride(0);
                    auto strideD = strideC / sizeof(*dst);
                    for (int m = 0; m < M; m++, src += strideS, dst += strideD) {
                        // the prefetch distance is increased to ensure by the time store happens
                        // prefetch has done and no HW prefetcher is triggered
                        auto* prefetch_dst = (m + 2 < M) ? (dst + 2 * strideD) : (dst);
                        (*work.p_jit_reduce2bh_1)(src, dst, prefetch_dst, work.BN);
                    }
                }
            }
        });
    }

    // gate & up are interleaved: 16 gates + 16 up
    void runGateUp(uint8_t* pA, int strideA, int M, ov::bfloat16* dstC, int strideC) {
        ov::parallel_nt_static(0, [&](const size_t ithr, const size_t nthr) {
            auto& work = works[ithr];
            auto& workC = getC(ithr);
            if (work.BN > 0) {
                work.run(M, pA, strideA, workC);
                // K reduce is done, results of [M, BN] sub-block is ready in L2.
                // combine Gate & Up
                auto* src = workC.ptr<float>();
                auto strideS = workC.stride(0);
                auto* dst = dstC + (work.n0 / 2);  // important output is only half of the total N
                auto strideD = strideC / sizeof(*dst);
                for (int m = 0; m < M; m++, src += strideS, dst += strideD) {
                    auto* prefetch_dst = (m + 1 < M) ? (dst + strideD) : (dst);
                    (*work.p_jit_gateup)(src, dst, prefetch_dst, work.BN);
                }
            }
        });
    }
};

struct QKVProj : public LLMMLP::Executor  {
    QKVProj() {}
    struct Work {
        MKernel* p_jit_amx0 = nullptr;
        ReduceAdd2bh* p_jit2bh = nullptr;

        std::vector<PlainTensor> weights;

        int output_id;
        int n0 = 0;
        int n1 = 0;
        int BN = 0;
        int BK = 0;
        operator bool() {
            return BN > 0;
        }
        int blk_K_size;

        ov::bfloat16 * p_raw_weights;

        // input : weight [N, K], setup repacks range of N [n_start, n_end)
        template <typename T>
        void setup(T* p_weight, int stride) {
            auto num_blk_K = BK / blk_K_size;
            auto* pw = p_weight + n0 * stride / sizeof(T);
            //weights.resize(num_blk_K);
            for (int k = 0; k < num_blk_K; k++) {
                p_jit_amx0->prepareB(weights[k], pw + k * blk_K_size, stride, BN, blk_K_size);
            }
            for (int Mtails = 0; Mtails < 32; Mtails++) {
                p_jit_amx0->tile_config_M(m_tcfg[Mtails], Mtails == 0? 32 : Mtails);
            }            
        }

        TileConfig m_tcfg[32];

        void run(int M, uint8_t* pA, int strideA, PlainTensor& C) {
            int num_blk_K = weights.size();

            auto Mtails = M % 32;
            auto Mbody = M - Mtails;

            auto& tile_configer = (Singleton<TileConfiger>::get());
            C.resize<float>({Mbody + (Mtails ? 32 : 0), BN});

            auto pC = reinterpret_cast<uint8_t*>(C.ptr<float>());
            bool do_accumulation = false;
            for (int ki = 0; ki < num_blk_K; ki++) {
                PlainTensor& blockB = weights[ki];
                PlainTensor& blockB1 = weights[(ki + 1) < num_blk_K ? (ki + 1) : ki];

                if (Mbody) {
                    tile_configer.do_config(&m_tcfg[0]);
                    p_jit_amx0->run(Mbody,
                                    pA + ki * blk_K_size * sizeof(ov::bfloat16),
                                    strideA,
                                    blockB,
                                    pC,
                                    C.stride_bytes(0),
                                    reinterpret_cast<uint8_t*>(blockB1.ptr_v()),
                                    do_accumulation);
                }

                if (Mtails) {
                    tile_configer.do_config(&m_tcfg[Mtails]);
                    p_jit_amx0->run(Mtails,
                                    pA + ki * blk_K_size * sizeof(ov::bfloat16) + Mbody * strideA,
                                    strideA,
                                    blockB,
                                    pC + Mbody * C.stride_bytes(0),
                                    C.stride_bytes(0),
                                    reinterpret_cast<uint8_t*>(blockB1.ptr_v()),
                                    do_accumulation);
                }
                do_accumulation = true;
            }
            tile_configer.do_config(nullptr);
        }
    };
    std::vector<Work> works;

    // q k v each have 1/3 or worker-thread
    void setup(ov::bfloat16* wq, ov::bfloat16* wk, ov::bfloat16* wv, int N, int K) {
        static MKernel jit_amx0(256);
        static ReduceAdd2bh jit_2bh(false);
        const int blk_K_size = 256;
        // prepare weights, split N among threads
        // in unit of 32
        OPENVINO_ASSERT((N % 32) == 0);
        OPENVINO_ASSERT((K % blk_K_size) == 0);
        auto nthr = parallel_get_max_threads();
        auto num_blk_N = N / 32;
        auto num_blk_K = K / blk_K_size;
        works.resize(nthr);

        int stride = K * sizeof(*wq);

        // every thread should do same amount of work, and some cores can be idle
        auto valid_nthr = nthr / 3;

        int cur_work_id = 0;
        auto create_works = [&](ov::bfloat16* pw, int output_id) {
            auto blkN_per_thread = (num_blk_N + valid_nthr - 1) / valid_nthr;
            auto blkN_leftover = 0;
            auto start_blkN = 0;

            // split task on more cores is better on TBB
            blkN_per_thread = (num_blk_N) / valid_nthr;
            blkN_leftover = num_blk_N - (blkN_per_thread*valid_nthr);
            
            for (int ithr = 0; ithr < valid_nthr; ithr ++) {
                auto blkN = std::min(num_blk_N - start_blkN, blkN_per_thread);
                if (blkN_leftover > 0) {
                    blkN_leftover--;
                    blkN ++;
                }
                if (blkN) {
                    auto& work = works[cur_work_id++];
                    work.p_jit_amx0 = &jit_amx0;
                    work.p_jit2bh = &jit_2bh;
                    work.blk_K_size = blk_K_size;

                    work.n0 = (start_blkN)*32;
                    work.n1 = (start_blkN + blkN) * 32;
                    work.BN = blkN * 32;
                    work.BK = blk_K_size * num_blk_K;

                    work.weights.resize(num_blk_K);
                    for(auto& weight : work.weights)
                        weight.resize<ov::bfloat16>({blkN, blk_K_size * 32});

                    work.output_id = output_id;
                    work.p_raw_weights = pw;
                }
                start_blkN += blkN;
            }
        };
        create_works(wq, 0);
        create_works(wk, 1);
        create_works(wv, 2);
        auto used_nthr = cur_work_id;

        ECOUT("QKVProj N,K=", N, ",", K, " used_nthr=", used_nthr);
        ov::parallel_nt_static(0, [&](const size_t ithr, const size_t nthr) {
            auto& work = works[ithr];
            if (work) {
                work.setup(work.p_raw_weights, stride);
            }
        });
        ECOUT("   setup is done. weight @ ", static_cast<void*>(wq), ",", static_cast<void*>(wk), ",", static_cast<void*>(wv));
    }

    void run(uint8_t* pA,
             int strideA,
             int M,
             ov::bfloat16* dst_q,
             int stride_q,
             ov::bfloat16* dst_k,
             int stride_k,
             ov::bfloat16* dst_v,
             int stride_v) {

        for (int m = 0; m < M;) {
            int BM = std::min(M - m, 256);

            ov::parallel_nt_static(0, [&](const size_t ithr, const size_t nthr) {
                auto& work = works[ithr];
                auto& C = getC(ithr);
                if (work.BN > 0) {
                    work.run(BM, pA, strideA, C);

                    // compress accumulation result into target
                    auto* src = C.ptr<float>();
                    auto stride_src = C.stride(0);
                    ov::bfloat16* dst = nullptr;
                    int stride_dst = 0;
                    if (work.output_id == 0) {
                        dst = dst_q + work.n0;
                        stride_dst = stride_q / sizeof(*dst);
                    }
                    if (work.output_id == 1) {
                        dst = dst_k + work.n0;
                        stride_dst = stride_k / sizeof(*dst);
                    }
                    if (work.output_id == 2) {
                        dst = dst_v + work.n0;
                        stride_dst = stride_v / sizeof(*dst);
                    }

                    for (int mi = 0; mi < BM; mi++, src += stride_src, dst += stride_dst) {
                        // the prefetch distance is increased to ensure by the time store happens
                        // prefetch has done and no HW prefetcher is triggered
                        auto* prefetch_dst = (mi + 2 < BM) ? (dst + 2 * stride_dst) : (dst);
                        (*work.p_jit2bh)(src, dst, prefetch_dst, work.BN);
                    }
                }
            });
            m += BM;
            pA += BM * strideA;
            dst_q += BM * stride_q / sizeof(ov::bfloat16);
            dst_k += BM * stride_k / sizeof(ov::bfloat16);
            dst_v += BM * stride_v / sizeof(ov::bfloat16);
        }
    }
    /*
    void setup(ov::bfloat16* wq, ov::bfloat16* wk, ov::bfloat16* wv, int K, int N) {
        q_proj.setup(&wq[0], K*sizeof(ov::float16), N, K);
        k_proj.setup(&wk[0], K*sizeof(ov::float16), N, K);
        v_proj.setup(&wv[0], K*sizeof(ov::float16), N, K);
    }
    void run(uint8_t* pA, int strideA, int M,
             ov::bfloat16* dst_q, int stride_q,
             ov::bfloat16* dst_k, int stride_k,
             ov::bfloat16* dst_v, int stride_v) {
        for(int m = 0; m < M;) {
            int BM = std::min(M - m, 512);

            q_proj.run(pA, strideA, BM, dst_q, stride_q);
            k_proj.run(pA, strideA, BM, dst_k, stride_k);
            v_proj.run(pA, strideA, BM, dst_v, stride_v);

            m += BM;
            pA += BM*strideA;
            dst_q += BM*stride_q/sizeof(ov::bfloat16);
            dst_k += BM*stride_k/sizeof(ov::bfloat16);
            dst_v += BM*stride_v/sizeof(ov::bfloat16);
        }
    }
    */
};

struct MLP : LLMMLP::Executor {
    Linear gate_up;
    Linear down;
    int m_N;
    int m_M = 0;

    MLP() {}

    // MLP is not supposed to run in parallel
    PlainTensor& get_actUp() {
        static PlainTensor actUp;
        return actUp;
    }

    // [M, K] x [N, K] => [M, N] x [K, N] => [M, K]
    // w_gate/w_up : [N, K]
    //     w_down  : [K, N]
    void setup(PlainTensor w_gate, PlainTensor w_up, PlainTensor w_down, int K, int N) {
        // [N, K] [N, K] interleave (16-16-...) into [2*N, K]
        //tensor2D<ov::bfloat16> w_gate(N, K, pw_gate, K * sizeof(ov::bfloat16));
        //tensor2D<ov::bfloat16> w_up(N, K, pw_up, K * sizeof(ov::bfloat16));
        //tensor2D<ov::bfloat16> w_down(K, N, pw_down, N * sizeof(ov::bfloat16));

        static PlainTensor w_gate_up;
        w_gate_up.resize<ov::bfloat16>({2 * N, K});
        for (int n = 0; n < N; n += 16) {
            for (int i = 0; i < 16; i++)
                memcpy(w_gate_up.ptr_v(2 * n + i, 0), w_gate.ptr_v(n + i, 0), K * sizeof(ov::bfloat16));
            for (int i = 0; i < 16; i++)
                memcpy(w_gate_up.ptr_v(2 * n + 16 + i, 0), w_up.ptr_v(n + i, 0), K * sizeof(ov::bfloat16));
        }
        gate_up.setup(w_gate_up.ptr<ov::bfloat16>(), w_gate_up.stride_bytes(0), N * 2, K);
        down.setup(w_down.ptr<ov::bfloat16>(), w_down.stride_bytes(0), K, N, true);
        m_N = N;
    }

    void setM(int M) {
        if (m_M < M) {
            get_actUp().resize<ov::bfloat16>({M, m_N});
            //std::cout << "M=" << M << std::endl;
            m_M = M;
        }
    }

    void run(uint8_t* pA, int strideA, int M, ov::bfloat16* dstC, int strideC) {
        auto& actUp = get_actUp();
        for(int m = 0; m < M;) {
            int BM = std::min(M - m, 512);
            setM(BM);

            gate_up.runGateUp(pA, strideA, BM, actUp.ptr<ov::bfloat16>(), actUp.stride_bytes(0));
            down.run(reinterpret_cast<uint8_t*>(actUp.ptr<ov::bfloat16>()), actUp.stride_bytes(0), BM, dstC, strideC);

            m += BM;
            pA += BM*strideA;
            dstC += BM*strideC/sizeof(ov::bfloat16);
        }
    }
};

};  // namespace AMX_MLP

LLMMLP::LLMMLP(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)), m_executor(nullptr) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW("CPU: " + errorMessage);
    }

    const auto node = std::dynamic_pointer_cast<const LLMMLPNode>(op);
    m_config = node->get_config();
}

void LLMMLP::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    auto srcPrecision = getOriginalInputPrecisionAtPort(0);

    auto rtPrecision = ov::element::bf16;
    auto weightPrecision = ov::element::bf16;

    // initialize input ports
    std::vector<PortConfigurator> inPortConfigs;
    inPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getInputShapeAtPort(0), false, -1);      // input
    inPortConfigs.emplace_back(LayoutType::ncsp, weightPrecision, getInputShapeAtPort(1), false, -1);  // gate
    inPortConfigs.emplace_back(LayoutType::ncsp, weightPrecision, getInputShapeAtPort(2), false, -1);  // up
    inPortConfigs.emplace_back(LayoutType::ncsp, weightPrecision, getInputShapeAtPort(3), false, -1);  // down

    // initialize output port
    std::vector<PortConfigurator> outPortConfigs;

    if (m_config.is_qkv_proj) {
        outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(0), false, -1);
        outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(0), false, -1);
        outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(0), false, -1);
    } else {
        outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(0), false, -1);
    }

    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any);
}

void LLMMLP::execute(dnnl::stream strm) {
#if 0
    static linux_perf_event lpevs({{PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS_MIN, "swicth"},
                                   {PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS_MAJ, "PG_FAULT"},
                                   {PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_MIGRATIONS, "PG_FAULT"},
                                   {PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS, "PG_FAULT"}});
#endif
    if (!m_executor) {
        if (m_config.is_qkv_proj) {
            auto exec = std::make_shared<AMX_MLP::QKVProj>();
            exec->setup(getSrcMemoryAtPort(1)->getDataAs<ov::bfloat16>(),
                    getSrcMemoryAtPort(2)->getDataAs<ov::bfloat16>(),
                    getSrcMemoryAtPort(3)->getDataAs<ov::bfloat16>(),
                    m_config.hidden_size,
                    m_config.hidden_size);
            m_executor = exec;
        } else {
            auto exec = std::make_shared<AMX_MLP::MLP>();
            exec->setup(getSrcMemoryAtPort(1),
                    getSrcMemoryAtPort(2),
                    getSrcMemoryAtPort(3),
                    m_config.hidden_size,
                    m_config.intermediate_size);
            m_executor = exec;
        }
    }

    auto input = getSrcMemoryAtPort(0);
    const auto& ishape = input->getStaticDims();
    uint8_t* pA = input->getDataAs<uint8_t>();
    int strideA = m_config.hidden_size * 2;
    int M = shape_size(ishape) / ishape[ishape.size()-1];
    ov::bfloat16* dst0;
    ov::bfloat16* dst1;
    ov::bfloat16* dst2;
    dst0 = getDstMemoryAtPort(0)->getDataAs<ov::bfloat16>();
    if (m_config.is_qkv_proj) {
        dst1 = getDstMemoryAtPort(1)->getDataAs<ov::bfloat16>();
        dst2 = getDstMemoryAtPort(2)->getDataAs<ov::bfloat16>();
    }

    if (m_config.is_qkv_proj) {
        //PROFILE(_prof, "qkv_proj", "");
        auto exec = std::dynamic_pointer_cast<AMX_MLP::QKVProj>(m_executor);

        exec->run(pA, strideA, M,
                  dst0, m_config.hidden_size * 2,
                  dst1, m_config.hidden_size * 2,
                  dst2, m_config.hidden_size * 2);
    } else {
        //PROFILE(_prof, "mlp");
        auto exec = std::dynamic_pointer_cast<AMX_MLP::MLP>(m_executor);

        auto output = getDstMemoryAtPort(0);
        int strideC = m_config.hidden_size * 2;

        exec->run(pA, strideA, M, dst0, strideC);
    }
}

bool LLMMLP::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto node = std::dynamic_pointer_cast<const LLMMLPNode>(op);
        if (!node) {
            errorMessage = "Only LLMMLPNode operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
