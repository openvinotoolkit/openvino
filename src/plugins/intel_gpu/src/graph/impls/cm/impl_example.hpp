// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "fully_connected_inst.h"
#include "registry/implementation_manager.hpp"

namespace cldnn {
namespace cm {

struct ExampleImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("cm::example")
    ExampleImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr)
        : ImplementationManager(impl_types::cm, shape_type, vf) {}

    std::unique_ptr<primitive_impl> create_impl(const program_node& node,
                                                const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        assert(node.is_type<fully_connected>());

        auto &engine = node.get_program().get_engine();
        auto &config = node.get_program().get_config();
        if (!check_cm_jit_support(engine, config)) {
            return false;
        }

        // Example impl should not be chosen unless forced
        return false;
    }
};

}  // namespace cm
}  // namespace cldnn
