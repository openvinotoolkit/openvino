// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/rms.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/rms.hpp"

namespace ov {
namespace op {
namespace internal {
using RMS = ov::intel_gpu::op::RMS;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov {
namespace intel_gpu {

static void CreateRMSOp(ProgramBuilder& p, const std::shared_ptr<op::RMS>& op) {
    validate_inputs_count(op, {2});
    auto inputs = p.GetInputInfo(op);
    std::string primitive_name = layer_type_name_ID(op);

    auto get_output_data_types = [&]() {
        std::vector<cldnn::optional_data_type> output_data_types;
        auto type = op->get_output_element_type(0);
        output_data_types.push_back(cldnn::element_type_to_data_type(type));
        return output_data_types;
    };
    auto rms = cldnn::rms(primitive_name,
                          inputs[0],
                          inputs[1],
                          op->get_epsilon());
    rms.output_data_types = get_output_data_types();
    p.add_primitive(*op, rms);
}

REGISTER_FACTORY_IMPL(internal, RMS);

}  // namespace intel_gpu
}  // namespace ov
