// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gru_seq_inst.h"
#include "intel_gpu/runtime/utils.hpp"
#include  "registry/implementation_manager.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>


namespace cldnn {
namespace onednn {

struct GRUSeqImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("onednn::gru_seq")
    GRUSeqImplementationManager(shape_types shape_type) : ImplementationManager(impl_types::onednn, shape_type) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        assert(node.is_type<gru_seq>());
        return true;
    }

    in_out_fmts_t query_formats(const program_node& node) const override {
        assert(node.is_type<gru_seq>());
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);

        size_t out_rank = node.get_output_layout().get_rank();
        for (size_t idx = 0 ; idx < node.get_dependencies().size(); idx++) {
            in_fmts[idx] = format::get_default_format(out_rank);
        }
        out_fmts[0] = format::ybfx;
        out_fmts[1] = format::bfyx;

        return {in_fmts, out_fmts};
    }
};

}  // namespace onednn
}  // namespace cldnn
