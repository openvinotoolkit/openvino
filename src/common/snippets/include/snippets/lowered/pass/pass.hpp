// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/lowered/linear_ir.hpp"

#include "openvino/core/rtti.hpp"
#include "openvino/core/type.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface Pass
 * @brief Base class for transformations on linear IR
 * @ingroup snippets
 */
class Pass {
public:
    Pass() = default;
    virtual ~Pass() = default;
    // Note that get_type_info_static and get_type_info are needed to mimic OPENVINO_RTTI interface,
    // so the standard OPENVINO_RTTI(...) macros could be used in derived classes.
    _OPENVINO_HIDDEN_METHOD static const ::ov::DiscreteTypeInfo& get_type_info_static() {
        static ::ov::DiscreteTypeInfo type_info_static {"Pass"};
        type_info_static.hash();
        return type_info_static;
    }

    virtual const DiscreteTypeInfo& get_type_info() const {
        return get_type_info_static();
    }

    const char* get_type_name() const {
        return get_type_info().name;
    }

    virtual bool run(lowered::LinearIR& linear_ir) = 0;
};

class PassPipeline {
public:
    PassPipeline() = default;

    void register_pass(const std::shared_ptr<Pass>& pass);

    template<typename T, class... Args>
    void register_pass(Args&&... args) {
        static_assert(std::is_base_of<Pass, T>::value, "Pass not derived from lowered::Pass");
        auto pass = std::make_shared<T>(std::forward<Args>(args)...);
        register_pass(pass);
    }

    void run(lowered::LinearIR& linear_ir) const;

private:
    std::vector<std::shared_ptr<Pass>> m_passes;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
