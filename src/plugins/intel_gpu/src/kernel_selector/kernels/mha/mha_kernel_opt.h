// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "mha_kernel_ref.h"

namespace kernel_selector {

class MHAKernelOpt : public KernelBaseOpenCL {
public:
    struct TuningData {
        size_t num_words;      // N: NUM_WORDS
        size_t depth_size;     // d: DEPTH_SIZE
        size_t blk_row_size;   // Br: BLK_ROW_SIZE
        size_t blk_col_size;   // Bc: BLK_COL_SIZE
        size_t num_blk_row;    // Tr: NUM_BLK_ROW
        size_t num_blk_col;    // Tc: NUM_BLK_COL
        size_t q_blk_size;     // Br * d: Q_BLK_SIZE
        size_t k_blk_size;     // Bc * d: K_BLK_SIZE
        size_t v_blk_size;     // Bc * d: V_BLK_SIZE
        size_t score_mat_size; // Br * Bc: SCORE_MAT_SIZE
        size_t out_blk_size;   // Br * d: OUT_BLK_SIZE

        std::string to_string() {
            std::stringstream ss;
            ss << "num_words: " << num_words << std::endl;
            ss << "depth_size: " << depth_size << std::endl;
            ss << "blk_row_size: " << blk_row_size << std::endl;
            ss << "blk_col_size: " << blk_col_size << std::endl;
            ss << "num_blk_row: " << num_blk_row << std::endl;
            ss << "num_blk_col: " << num_blk_col << std::endl;
            ss << "q_blk_size: " << q_blk_size << std::endl;
            ss << "k_blk_size: " << k_blk_size << std::endl;
            ss << "v_blk_size: " << v_blk_size << std::endl;
            ss << "score_mat_size: " << score_mat_size << std::endl;
            ss << "out_blk_size: " << out_blk_size << std::endl;
            return ss.str();
        }
    };

    MHAKernelOpt() : KernelBaseOpenCL("mha_opt") {}
    virtual ~MHAKernelOpt() {}
    virtual JitConstants GetJitConstants(const mha_params& params) const;
    virtual CommonDispatchData SetDefault(const mha_params& params) const;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { };
    }
    KernelsPriority GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const override {
        return FORCE_PRIORITY_1;
    }

protected:
    bool Validate(const Params& p, const optional_params& o) const override;

private:
    TuningData GetTuningData(const mha_params& params) const;
};
}  // namespace kernel_selector
