// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <intel_gpu/primitives/slice.hpp>
#include "primitive_inst.h"

namespace cldnn {

using slice_node = typed_program_node<slice>;

// This class is needed to have one place where decision
// is made which Slice inputs are used by the kernel on GPU.
// Unfortnately, the same decison needs to be made
// in multiple places, including:
// - slice_inst::update_shape_info_tensor
// - slice_impl::get_arguments
// - slice_impl::create
// This class was created to encapsulate that logic in single place.
// NOTE: the placement of this class is the 'lesser evil'. Normally such logic
// should be a part of codegen/jitter, which should output some struct with information
// about which data is needed by the kernel, how it should be provided, bindings, etc.
// Currently it is scattered in mutiple places, where basically similar logic has to be applied.
// NOTE: This class implicietly depends on logic inside SliceKernelRef and the kernel
// itself. If you make any changes of how params are provided to kernel,
// likely you will needed to update this one too.
class SliceKernelRefNeededInputs {
public:
    enum InputIndices {
        kData,
        kStart,
        kEnd,
        kStep,
        kAxes,
        kInputsNum
    };

    // Creates instance of SliceKernelRefNeededInputs.
    static SliceKernelRefNeededInputs Create(const slice_node& node);

    // Retruns needed indexes in runtime.
    const std::vector<size_t>& GetNeededInputIndexes() const;

    // Returns true if given input is needed in runtime.
    bool IsInputNeededInRuntime(InputIndices type) const;

private:
    std::vector<size_t> neededIndexes;
};

template <>
class typed_primitive_inst<slice> : public typed_primitive_inst_base<slice> {
    using parent = typed_primitive_inst_base<slice>;
    using parent::parent;

public:
    template<typename ShapeType>
        static std::vector<layout> calc_output_layouts(const slice_node& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(slice_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(slice_node const& node);

    typed_primitive_inst(network& network, slice_node const& desc);
    void update_shape_info_tensor(const kernel_impl_params& params) override;
};

using slice_inst = typed_primitive_inst<slice>;

///////////////////////////////////////////////////////////////////
//
// INLINES:
//
///////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////
inline const std::vector<size_t>& SliceKernelRefNeededInputs::GetNeededInputIndexes() const {
    return neededIndexes;
}

///////////////////////////////////////////////////////////////////
inline bool SliceKernelRefNeededInputs::IsInputNeededInRuntime(InputIndices type) const {
    for (auto idx : neededIndexes) {
        if (idx == type)
            return true;
    }
    return false;
}

} // namespace cldnn
