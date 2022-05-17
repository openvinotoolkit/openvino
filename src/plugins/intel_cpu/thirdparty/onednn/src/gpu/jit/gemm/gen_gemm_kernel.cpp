/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "gpu/jit/gemm/gen_gemm_kernel.hpp"
#include "gemm_recipes.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

namespace {

char layout_char(MatrixLayout layout) {
    switch (layout) {
        default: assert(!"Unknown layout.");
        case MatrixLayout::PackedColumns: return 'A';
        case MatrixLayout::PackedRows: return 'B';
        case MatrixLayout::Nontranspose: return 'N';
        case MatrixLayout::Transpose: return 'T';
    }
}

char precision_char(Type T) {
    switch (T) {
        default: assert(!"Unknown type.");
        case Type::f16: return 'H';
        case Type::bf16: return 'B';
        case Type::f32: return 'S';
        case Type::u8:
        case Type::s8: return 'O';
        case Type::u16:
        case Type::s16: return 'W';
        case Type::u32:
        case Type::s32: return 'I';
    }
}

AccessType get_access_type(char c) {
    switch (c) {
        default: assert(!"Unknown access type.");
        case 'b': return AccessType::Block;
        case 's': return AccessType::Scattered;
        case 'u': return AccessType::ChannelScattered;
    }
}

ngen::AddressBase get_address_base(char c) {
    switch (c) {
        default: assert(!"Unknown address space.");
        case 'a': return ngen::AddressBase::createA64(true);
        case 's': return ngen::AddressBase::createBTS(0);
    }
}

} // anonymous namespace

bool gen_gemm_kernel_t::matching_hw(ngen::HW hw, ngen::HW hw_ref) {
    using ngen::HW;
    if (hw == hw_ref) return true;
    if (hw == HW::XeHPG && hw_ref == HW::XeHP) return true;
    return false;
}

status_t gen_gemm_kernel_t::complete_strategy() {
    using ngen::HW;

    problem_.nonuniformWGs = false;
    problem_.fused = (hw_ >= HW::XeLP);
    strategy_.emulate = EmulationStrategy(hw_);
    strategy_.checkAdd32 = strategy_.emulate.emulate64;
    strategy_.spf = !problem_.fused;

    for (int r = 0; r < gemm_recipe_count; r++) {
        auto &recipe = gemm_recipes[r];
        if (matching_hw(hw_, recipe.hw)
                && recipe.precisions[0] == precision_char(problem_.Ta)
                && recipe.precisions[1] == precision_char(problem_.Tb)
                && recipe.precisions[2] == precision_char(problem_.Tc)
                && recipe.layouts[0] == layout_char(problem_.A.layout)
                && recipe.layouts[1] == layout_char(problem_.B.layout)
                && recipe.layouts[2] == layout_char(problem_.C.layout)
                && recipe.extra.aCP == problem_.A.crosspack
                && recipe.extra.bCP == problem_.B.crosspack
                && (problem_.A.alignment % recipe.extra.aAlign == 0)
                && (problem_.B.alignment % recipe.extra.bAlign == 0)
                && recipe.unrollM == strategy_.unroll[LoopM]
                && recipe.unrollN == strategy_.unroll[LoopN]
                && recipe.tag == strategy_tag_) {

            // Align alignments to recipe alignments.
            if (utils::one_of(
                        problem_.A.layout, MatrixLayout::N, MatrixLayout::T))
                problem_.A.setAlignment(
                        std::max(problem_.Ta.size(), recipe.extra.aAlign));
            if (utils::one_of(
                        problem_.B.layout, MatrixLayout::N, MatrixLayout::T))
                problem_.B.setAlignment(
                        std::max(problem_.Tb.size(), recipe.extra.bAlign));
            problem_.C.setAlignment(problem_.Tc_ext.size());
            problem_.CO.setAlignment(problem_.Tco.size());

            return read_strategy(recipe.strategyString);
        }
    }

    return status::unimplemented;
}

status_t gen_gemm_kernel_t::read_strategy(const char *str) {
    using ngen::HW;
    std::stringstream s(str);

    bool override_fused_loop = false;
    bool override_register_scheme = false;
    bool override_c_remainder = false;

    strategy_.ka_load_masked = strategy_.kb_load_masked = 0;
    strategy_.unroll[LoopK] = 1;
    strategy_.fmaSIMD = 64
            / std::max<int>({problem_.Ta.size(), problem_.Tb.size(),
                    problem_.Tc.size()});

    strategy_.remHandling[LoopM] = problem_.A.padded
            ? RemainderHandling::General
            : RemainderHandling::Split;
    strategy_.remHandling[LoopN] = problem_.B.padded
            ? RemainderHandling::General
            : RemainderHandling::Split;
    strategy_.remHandling[LoopK] = RemainderHandling::General;

    char asA, asB, asC, accessA, accessB, accessC, eat;
    char accessAPrefetch = 's', accessBPrefetch = 's', accessCPrefetch = 's';

    s >> std::ws >> asA >> accessA >> strategy_.ka_load;
    if (s.peek() == '/') s >> eat >> strategy_.ka_load_masked;
    if (s.peek() == 'x') s >> eat >> strategy_.A_copies;
    if (s.peek() == '+') {
        strategy_.prefetchA = 1;
        s >> eat >> accessAPrefetch >> strategy_.ka_prefetch;
        if (s.peek() == ',') s >> eat >> strategy_.ka_pfStride;
        if (s.peek() == '@') s >> eat >> strategy_.prefetchA;
    }
    s >> std::ws >> asB >> accessB >> strategy_.kb_load;
    if (s.peek() == '/') s >> eat >> strategy_.kb_load_masked;
    if (s.peek() == 'x') s >> eat >> strategy_.B_copies;
    if (s.peek() == '+') {
        strategy_.prefetchB = 1;
        s >> eat >> accessBPrefetch >> strategy_.kb_prefetch;
        if (s.peek() == ',') s >> eat >> strategy_.kb_pfStride;
        if (s.peek() == '@') s >> eat >> strategy_.prefetchB;
    }
    s >> std::ws >> asC >> accessC;
    if (s.peek() == '+') {
        strategy_.prefetchC = 1;
        s >> eat >> accessCPrefetch;
        if (s.peek() == '@') s >> eat >> strategy_.prefetchC;
    }

    problem_.A.base = get_address_base(asA);
    problem_.B.base = get_address_base(asB);
    problem_.C.base = get_address_base(asC);
    strategy_.A.accessType = get_access_type(accessA);
    strategy_.B.accessType = get_access_type(accessB);
    strategy_.C.accessType = get_access_type(accessC);

    strategy_.A_prefetch.atomic = false;
    strategy_.B_prefetch.atomic = false;
    strategy_.C_prefetch.atomic = false;
    strategy_.A_prefetch.prefetch = true;
    strategy_.B_prefetch.prefetch = true;
    strategy_.C_prefetch.prefetch = true;
    strategy_.A_prefetch.accessType = get_access_type(accessAPrefetch);
    strategy_.B_prefetch.accessType = get_access_type(accessBPrefetch);
    strategy_.C_prefetch.accessType = get_access_type(accessCPrefetch);

    while (!s.eof()) {
        std::string mod;
        s >> mod;
        if (mod == "cs") {
            override_register_scheme = true;
            strategy_.registerScheme = GEMMStrategy::CSeparate;
        } else if (mod == "acb") {
            override_register_scheme = true;
            strategy_.registerScheme = GEMMStrategy::ACB;
        } else if (mod == "bca") {
            override_register_scheme = true;
            strategy_.registerScheme = GEMMStrategy::BCA;
        } else if (mod == "vnc") {
            override_register_scheme = true;
            strategy_.registerScheme = GEMMStrategy::VNC;
        } else if (mod == "int") {
            override_register_scheme = true;
            strategy_.registerScheme = GEMMStrategy::ABInterleave;
        } else if (mod == "ar") {
            override_c_remainder = true;
            strategy_.altCRemainder = true;
        } else if (mod == "sr") {
            override_c_remainder = true;
            strategy_.altCRemainder = false;
        } else if (mod == "ac")
            strategy_.cAccumulators = true;
        else if (mod == "fs")
            strategy_.fixedSystolic = strategy_.systolic = true;
        else if (mod == "da")
            strategy_.duplicateA = true;
        else if (mod == "db")
            strategy_.duplicateB = true;
        else if (mod == "el")
            strategy_.cLoadAhead = true;
        else if (mod == "di")
            strategy_.delayABInc = true;
        else if (mod == "sc")
            strategy_.splitCopy = true;
        else if (mod == "sm")
            strategy_.slmMBlockSplit = true;
        else if (mod == "sn")
            strategy_.slmNBlockSplit = true;
        else if (mod == "ek")
            strategy_.slmEarlyKMask = true;
        else if (mod == "pab")
            problem_.A.padded = problem_.B.padded = true;
        else if (mod == "nmk") {
            strategy_.loopOrder[0] = LoopN;
            strategy_.loopOrder[1] = LoopM;
            strategy_.loopOrder[2] = LoopK;
        } else if (mod == "fm") {
            problem_.fusedLoop = LoopM;
            override_fused_loop = true;
        } else if (mod == "fn") {
            problem_.fusedLoop = LoopN;
            override_fused_loop = true;
        } else if (mod == "kb") {
            strategy_.kParallel = true;
            strategy_.C.atomic = true;
        } else if (mod == "wg") {
            char x;
            s >> strategy_.wg[LoopM];
            s >> std::ws >> x;
            s >> strategy_.wg[LoopN];
        } else if (mod == "bo")
            strategy_.boustrophedon = true;
        else if (mod == "hi")
            strategy_.hilbertOrder = true;
        else if (mod == "sys")
            strategy_.systolic = true;
        else if (mod == "grf256")
            strategy_.GRFs = 256;
        else if (mod.substr(0, 2) == "ms")
            strategy_.mSplitThresh = stoi(mod.substr(2));
        else if (mod.substr(0, 2) == "ns")
            strategy_.nSplitThresh = stoi(mod.substr(2));
        else if (mod.substr(0, 2) == "kr")
            strategy_.kParallelLocal = stoi(mod.substr(2));
        else if (mod.substr(0, 2) == "bm")
            strategy_.blocking[LoopM] = stoi(mod.substr(2));
        else if (mod.substr(0, 2) == "bn")
            strategy_.blocking[LoopN] = stoi(mod.substr(2));
        else if (mod.substr(0, 2) == "bk")
            strategy_.blocking[LoopK] = stoi(mod.substr(2));
        else {
            switch (mod[0]) {
                case 'c': {
                    mod.erase(0, 1);
                    if (mod[0] == 'a') {
                        mod.erase(0, 1);
                        strategy_.slmA = true;
                    }
                    if (mod[0] == 'b') {
                        mod.erase(0, 1);
                        strategy_.slmB = true;
                    }
                    std::stringstream ms(mod);
                    ms >> strategy_.slmBuffers;
                    ms >> eat;
                    if (!ms.eof()) ms >> strategy_.slmCopies;
                    break;
                }
                case 'k': {
                    std::stringstream ms(mod);
                    ms >> eat >> strategy_.unroll[LoopK];
                    if (!ms.eof() && (ms.peek() == '/'))
                        ms >> eat >> strategy_.unrollK_masked;
                    break;
                }
                case 'l': strategy_.optAlignAB = stoi(mod.substr(1)); break;
                case 'r': {
                    bool is_a = (mod[1] == 'a');
                    (is_a ? strategy_.ka_repack : strategy_.kb_repack)
                            = stoi(mod.substr(2));
                    break;
                }
                default: return status::runtime_error;
            }
        }
    }

    if (!override_fused_loop) {
        problem_.fusedLoop = strategy_.loopOrder[0];
        if (problem_.fused) {
            if (strategy_.wg[LoopM] == 1)
                problem_.fusedLoop = LoopN;
            else if (strategy_.wg[LoopN] == 1)
                problem_.fusedLoop = LoopM;
        }
    }

    if (!override_c_remainder) {
        strategy_.altCRemainder = (strategy_.C.accessType == AccessType::Block)
                || strategy_.kParallel;
    }

    if (!override_register_scheme && (hw_ == HW::XeLP)) {
        strategy_.registerScheme
                = (strategy_.unroll[LoopM] * problem_.Ta.size()
                          == strategy_.unroll[LoopN] * problem_.Tb.size())
                ? GEMMStrategy::VNC
                : GEMMStrategy::ABInterleave;
    }

    if (strategy_.ka_load_masked == 0)
        strategy_.ka_load_masked = strategy_.ka_load;
    if (strategy_.kb_load_masked == 0)
        strategy_.kb_load_masked = strategy_.kb_load;

    strategy_.preflight(hw_, problem_);

    return status::success;
}

status_t gen_gemm_kernel_t::init_interface() {
    using namespace ngen;

    interface_ = NEOInterfaceHandler {hw_};
    auto s_type_ngen = problem_.Ts.ngen();

    interface_.newArgument("A", ExternalArgumentType::GlobalPtr);
    interface_.newArgument("B", ExternalArgumentType::GlobalPtr);
    interface_.newArgument("C", ExternalArgumentType::GlobalPtr);
    interface_.newArgument("offset_A", DataType::q);
    interface_.newArgument("offset_B", DataType::q);
    interface_.newArgument("offset_C", DataType::q);
    interface_.newArgument("lda", DataType::d);
    interface_.newArgument("ldb", DataType::d);
    interface_.newArgument("ldc", DataType::d);
    interface_.newArgument("m", DataType::d);
    interface_.newArgument("n", DataType::d);
    interface_.newArgument("k", DataType::d);
    interface_.newArgument("alpha_real", s_type_ngen);
    interface_.newArgument("beta_real", s_type_ngen);
    if (problem_.abOffset != ABOffset::None)
        interface_.newArgument("abo", DataType::ud);
    if (problem_.cOffset != COffset::None) {
        interface_.newArgument("CO", ExternalArgumentType::GlobalPtr);
        interface_.newArgument("offset_CO", DataType::d);
    }
    interface_.newArgument("flags", DataType::ud);
    if (strategy_.kParallel || strategy_.kParallelLocal)
        interface_.newArgument("k0", DataType::d);
    if (problem_.batch == BatchMode::Strided) {
        if (problem_.batchDims > 1) {
            interface_.newArgument("stride_A1", DataType::d);
            interface_.newArgument("stride_B1", DataType::d);
            interface_.newArgument("stride_C1", DataType::d);
        }
        interface_.newArgument("stride_A", DataType::d);
        interface_.newArgument("stride_B", DataType::d);
        interface_.newArgument("stride_C", DataType::d);
        if (problem_.batchDims > 1) {
            interface_.newArgument("batch_size1", DataType::ud);
            interface_.newArgument("recip_batch_size1", DataType::ud);
        }
    }
    if (strategy_.linearOrder()) {
        interface_.newArgument("group_count_m", DataType::ud);
        interface_.newArgument("group_count_n", DataType::ud);
    }
    if (strategy_.hilbertOrder) {
        interface_.newArgument("hilbert_vd", DataType::ud);
        interface_.newArgument("hilbert_uvd_recip", DataType::ud);
        interface_.newArgument("hilbert_bail", DataType::ud);
    } else if (strategy_.boustrophedon) {
        interface_.newArgument("bslice", DataType::d);
        interface_.newArgument("bthresh", DataType::d);
    }

    interface_.externalName(kernel_name());

    return status::success;
}

cl_kernel gen_gemm_kernel_t::get_kernel(
        cl_context context, cl_device_id device) {
    using ngen::HW;

    cl_kernel ocl_kernel = nullptr;

    switch (hw_) {
        case HW::Gen9: {
            gemm_kernel_generator_t<HW::Gen9> generator;
            generator.gemm(problem_, strategy_, interface_);
            ocl_kernel = generator.getKernel(context, device);
            break;
        }
        case HW::XeLP: {
            gemm_kernel_generator_t<HW::XeLP> generator;
            generator.gemm(problem_, strategy_, interface_);
            ocl_kernel = generator.getKernel(context, device);
            break;
        }
        case HW::XeHP: {
            gemm_kernel_generator_t<HW::XeHP> generator;
            generator.gemm(problem_, strategy_, interface_);
            ocl_kernel = generator.getKernel(context, device);
            break;
        }
        case HW::XeHPG: {
            gemm_kernel_generator_t<HW::XeHPG> generator;
            generator.gemm(problem_, strategy_, interface_);
            ocl_kernel = generator.getKernel(context, device);
            break;
        }
        default: assert(!"Unsupported architecture"); break;
    }

    return ocl_kernel;
}

CommonDriverInfo gen_gemm_kernel_t::driver_info() const {
    return strategy_.driverInfo(problem_);
}

namespace {

// clang-format off
struct align_req_t {
    int a, b, c;

    constexpr align_req_t() : a(1), b(1), c(1) {}
};

struct kernel_table_t {
    int unrolls[2];
    int max_accept[2];  // Maximum values for m/n for which this kernel will
                        //   always be chosen. (-1 = last in list).
    int min_reject[2];  // Minimum values for m/n beyond which this kernel will
                        //   always be rejected (0 if none).
    align_req_t aligns; // Alignment requirements for A/B/C, if any.
    char tag;           // Optional tag character, to select between strategies
                        //   with identical unrolls.
};

const kernel_table_t gen9_f32_nocopy_nn_table[] = {
    {{8,  4 }, { 0,  0}, {256, 0}, {}, {}},
    {{16, 8 }, { 0,  0}, {0,   0}, {}, {}},
    {{16, 16}, { 0,  0}, {0,   0}, {}, {}},
    {{32, 16}, {-1, -1}, {0,   0}, {}, {}},
};

const kernel_table_t gen9_f32_nocopy_nt_table[] = {
    {{8,  8 }, { 0,  0}, {512, 0}, {}, {}},
    {{16, 16}, { 0,  0}, {0,   0}, {}, {}},
    {{32, 16}, {-1, -1}, {0,   0}, {}, {}}
};

const kernel_table_t gen9_f32_nocopy_tn_table[] = {
    {{8,  4 }, {16, 32}, {0, 0}, {}, {}},
    {{8,  8 }, { 0,  0}, {0, 0}, {}, {}},
    {{16, 8 }, { 0,  0}, {0, 0}, {}, {}},
    {{16, 16}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t gen9_f32_nocopy_tt_table[] = {
    {{16, 32}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t *gen9_f32_nocopy_tables[2][2] = {
    {gen9_f32_nocopy_nn_table, gen9_f32_nocopy_nt_table},
    {gen9_f32_nocopy_tn_table, gen9_f32_nocopy_tt_table}
};

const kernel_table_t gen9_f16_nocopy_nn_table[] = {
    {{32, 32}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t gen9_f16_nocopy_nt_table[] = {
    {{32, 32}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t gen9_f16_nocopy_tn_table[] = {
    {{16, 16}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t gen9_f16_nocopy_tt_table[] = {
    {{32, 32}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t *gen9_f16_nocopy_tables[2][2] = {
    {gen9_f16_nocopy_nn_table, gen9_f16_nocopy_nt_table},
    {gen9_f16_nocopy_tn_table, gen9_f16_nocopy_tt_table}
};

const kernel_table_t gen9_x8_nocopy_nn_table[] = {
    {{32, 16}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t gen9_x8_nocopy_nt_table[] = {
    {{32, 16}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t gen9_x8_nocopy_tn_table[] = {
    {{16, 16}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t gen9_x8_nocopy_tt_table[] = {
    {{16, 32}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t *gen9_x8_nocopy_tables[2][2] = {
    {gen9_x8_nocopy_nn_table, gen9_x8_nocopy_nt_table},
    {gen9_x8_nocopy_tn_table, gen9_x8_nocopy_tt_table}
};

const kernel_table_t *gen9_bf16_nocopy_tables[2][2] = {
    {nullptr, nullptr},
    {nullptr, nullptr}
};

const kernel_table_t xe_lp_f32_nocopy_nn_table[] = {
    {{8,  4 }, { 0,  0}, {0, 0}, {}, {}},
    {{8,  8 }, { 0,  0}, {0, 0}, {}, {}},
    {{16, 8 }, { 0,  0}, {0, 0}, {}, {}},
    {{32, 8 }, { 0,  0}, {0, 0}, {}, {}},
    {{32, 12}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t xe_lp_f32_nocopy_nt_table[] = {
    {{8,  4 }, { 0,  0}, {0, 0}, {}, {}},
    {{8,  8 }, { 0,  0}, {0, 0}, {}, {}},
    {{16, 16}, { 0,  0}, {0, 0}, {}, {}},
    {{32, 16}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t xe_lp_f32_nocopy_tn_table[] = {
    {{8,  4 }, { 0,  0}, {0, 0}, {}, {}},
    {{16, 8 }, { 0,  0}, {0, 0}, {}, {}},
    {{16, 16}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t xe_lp_f32_nocopy_tt_table[] = {
    {{12, 32}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t *xe_lp_f32_nocopy_tables[2][2] = {
    {xe_lp_f32_nocopy_nn_table, xe_lp_f32_nocopy_nt_table},
    {xe_lp_f32_nocopy_tn_table, xe_lp_f32_nocopy_tt_table}
};

const kernel_table_t xe_lp_f16_nocopy_nn_table[] = {
    {{32, 32}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t xe_lp_f16_nocopy_nt_table[] = {
    {{32, 32}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t xe_lp_f16_nocopy_tn_table[] = {
    {{32, 16}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t xe_lp_f16_nocopy_tt_table[] = {
    {{32, 32}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t *xe_lp_f16_nocopy_tables[2][2] = {
    {xe_lp_f16_nocopy_nn_table, xe_lp_f16_nocopy_nt_table},
    {xe_lp_f16_nocopy_tn_table, xe_lp_f16_nocopy_tt_table}
};

const kernel_table_t xe_lp_x8_nocopy_nn_table[] = {
    {{32, 16}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t xe_lp_x8_nocopy_nt_table[] = {
    {{16, 32}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t xe_lp_x8_nocopy_tn_table[] = {
    {{16, 16}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t xe_lp_x8_nocopy_tt_table[] = {
    {{16, 32}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t *xe_lp_x8_nocopy_tables[2][2] = {
    {xe_lp_x8_nocopy_nn_table, xe_lp_x8_nocopy_nt_table},
    {xe_lp_x8_nocopy_tn_table, xe_lp_x8_nocopy_tt_table}
};

const kernel_table_t *xe_lp_bf16_nocopy_tables[2][2] = {
    {nullptr, nullptr},
    {nullptr, nullptr}
};

const kernel_table_t xe_hp_f16_nocopy_nn_table[] = {
    {{16,  4}, {0,   0}, {1,  0}, {}, {}},
    {{32, 16}, {64,  0}, {64, 0}, {}, 'B'},
    {{32, 16}, {-1, -1}, {0,  0}, {}, 'A'}
};

const kernel_table_t xe_hp_f16_nocopy_nt_table[] = {
    {{32, 32}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t xe_hp_f16_nocopy_tn_table[] = {
    {{16,  8}, {32, 32}, {32, 32}, {}, {}},
    {{16, 16}, {-1, -1}, {0,   0}, {}, {}}
};

const kernel_table_t xe_hp_f16_nocopy_tt_table[] = {
    {{32, 32}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t *xe_hp_f16_nocopy_tables[2][2] = {
    {xe_hp_f16_nocopy_nn_table, xe_hp_f16_nocopy_nt_table},
    {xe_hp_f16_nocopy_tn_table, xe_hp_f16_nocopy_tt_table}
};

const kernel_table_t xe_hp_f32_nocopy_nn_table[] = {
    {{8,  4}, { 0,  0}, {1024,  0}, {}, {}},
    {{16, 4}, { 0,  0}, {   0, 31}, {}, {}},
    {{16, 8}, { 0,  0}, {   0,  0}, {}, {}},
    {{32, 8}, { 0,  0}, {   0,  0}, {}, {}},
    {{64, 8}, {-1, -1}, {   0,  0}, {}, {}}
};

const kernel_table_t xe_hp_f32_nocopy_nt_table[] = {
    {{8,   8}, { 0,  0}, {128, 128}, {}, 'K'},
    {{8,   8}, { 0,  0}, {  0,   0}, {}, {}},
    {{16,  8}, { 0,  0}, {  0,   0}, {}, {}},
    {{16, 16}, { 0,  0}, {  0,   0}, {}, {}},
    {{32, 16}, {-1, -1}, {  0,   0}, {}, {}}
};

const kernel_table_t xe_hp_f32_nocopy_tn_table[] = {
    {{8,   4}, { 0,  0}, {0, 0}, {}, {}},
    {{8,   8}, { 0,  0}, {0, 0}, {}, {}},
    {{8,  16}, { 0,  0}, {0, 0}, {}, {}},
    {{16, 16}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t xe_hp_f32_nocopy_tt_table[] = {
    {{8, 64}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t *xe_hp_f32_nocopy_tables[2][2] = {
    {xe_hp_f32_nocopy_nn_table, xe_hp_f32_nocopy_nt_table},
    {xe_hp_f32_nocopy_tn_table, xe_hp_f32_nocopy_tt_table}
};

const kernel_table_t xe_hp_x8_nocopy_nn_table[] = {
    {{32,  4}, { 0,  0}, {0, 0}, {}, {}},
    {{16, 16}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t xe_hp_x8_nocopy_nt_table[] = {
    {{32, 16}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t xe_hp_x8_nocopy_tn_table[] = {
    {{16, 16}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t xe_hp_x8_nocopy_tt_table[] = {
    {{16, 16}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t *xe_hp_x8_nocopy_tables[2][2] = {
    {xe_hp_x8_nocopy_nn_table, xe_hp_x8_nocopy_nt_table},
    {xe_hp_x8_nocopy_tn_table, xe_hp_x8_nocopy_tt_table}
};

const kernel_table_t xe_hp_bf16_nocopy_nn_table[] = {
    {{32, 8}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t xe_hp_bf16_nocopy_nt_table[] = {
    {{16, 16}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t xe_hp_bf16_nocopy_tn_table[] = {
    {{32, 8}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t xe_hp_bf16_nocopy_tt_table[] = {
    {{8, 32}, {-1, -1}, {0, 0}, {}, {}}
};

const kernel_table_t *xe_hp_bf16_nocopy_tables[2][2] = {
    {xe_hp_bf16_nocopy_nn_table, xe_hp_bf16_nocopy_nt_table},
    {xe_hp_bf16_nocopy_tn_table, xe_hp_bf16_nocopy_tt_table}
};

// clang-format on

} // anonymous namespace

void gen_gemm_nocopy_kernel_t::choose_unrolls(compute::gpu_arch_t arch,
        int hw_threads, bool trans_a, bool trans_b, data_type_t a_type,
        data_type_t b_type, data_type_t c_type, int align_a, int align_b,
        int align_c, dim_t m, dim_t n, dim_t k, dim_t batch, int batch_dims,
        int &unroll_m, int &unroll_n, char &tag) {

    unroll_m = unroll_n = 1;

    using tables_t = decltype(gen9_f32_nocopy_tables);
    const tables_t *all_tables[4][3]
            = {{&gen9_f32_nocopy_tables, &xe_lp_f32_nocopy_tables,
                       &xe_hp_f32_nocopy_tables},
                    {&gen9_f16_nocopy_tables, &xe_lp_f16_nocopy_tables,
                            &xe_hp_f16_nocopy_tables},
                    {&gen9_bf16_nocopy_tables, &xe_lp_bf16_nocopy_tables,
                            &xe_hp_bf16_nocopy_tables},
                    {&gen9_x8_nocopy_tables, &xe_lp_x8_nocopy_tables,
                            &xe_hp_x8_nocopy_tables}};
    // clang-format off
    int arch_idx = (arch == compute::gpu_arch_t::xe_lp) ? 1
                 : (arch >= compute::gpu_arch_t::xe_hp) ? 2
                 : 0;
    int type_idx = (c_type == data_type::f16) ? 1
                : (c_type == data_type::bf16) ? 2
                :  (c_type == data_type::s32) ? 3 : 0;
    // clang-format on

    const kernel_table_t *table
            = (*all_tables[type_idx][arch_idx])[trans_a][trans_b];
    if (!table) {
        assert(!"Unsupported type for hardware.");
        return;
    }

    // Loop through kernel set, from smallest to largest unrolls.
    for (; table->max_accept[0] != -1; table++) {
        // Check if kernel alignment requirements are met.
        if (align_a % table->aligns.a || align_b % table->aligns.b
                || align_c % table->aligns.c)
            continue;

        // 'K' tag kernels require k parallelization, which can't be used for batch gemm.
        if (table->tag == 'K' && (batch > 1)) continue;

        // If m/n under "always use" threshold, use this kernel.
        // If m/n over "reject" threshold, don't use this kernel.
        if (m <= table->max_accept[0] || n <= table->max_accept[1]) break;
        if (table->min_reject[0] > 0 && m > table->min_reject[0]) continue;
        if (table->min_reject[1] > 0 && n > table->min_reject[1]) continue;

        // Otherwise, check if more HW threads would be spawned than are
        // available on the GPU. If so, enlarge unrolls.
        auto trial_unroll_m = table->unrolls[0];
        auto trial_unroll_n = table->unrolls[1];
        auto mnb_threads = utils::div_up(m, trial_unroll_m)
                * utils::div_up(n, trial_unroll_n) * batch;
        if (mnb_threads <= hw_threads) break;
        // To reduce register pressure for specific cases, use smaller unrolls if batch_dims > 1
        if (batch_dims > 1) {
            if ((arch_idx == 0) && (c_type == data_type::f32)
                    && (a_type == data_type::f32) && !(trans_a) && !(trans_b)
                    && (trial_unroll_n == 16)) {
                break;
            }
        }
    }
    unroll_m = table->unrolls[0];
    unroll_n = table->unrolls[1];
    tag = table->tag;
}

void gen_gemm_xehp_systolic_kernel_t::choose_unrolls(compute::gpu_arch_t arch,
        int eu_count, data_type_t a_type, data_type_t b_type,
        data_type_t c_type, dim_t m, dim_t n, dim_t k, dim_t batch,
        int &unroll_m, int &unroll_n, char &tag) {
    if (unroll_m == 0) unroll_m = 32;
    if (unroll_n == 0) unroll_n = (m * n >= 6144 * eu_count) ? 48 : 32;

    if (unroll_n == 32)
        tag = '\0';
    else
        tag = (m * n >= 13824 * eu_count) ? 'B' : 'A';
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
