// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <intel_gpu/primitives/slice_scatter.hpp>
#include "primitive_inst.h"

namespace cldnn {

using slice_scatter_node = typed_program_node<slice_scatter>;

// Helper class for determining which SliceScatter inputs are needed by the kernel at runtime.
// Similar to SliceKernelRefNeededInputs for Slice.
class SliceScatterKernelRefNeededInputs {
public:
    enum InputIndices {
        kData,
        kUpdates,
        kStart,
        kEnd,
        kStep,
        kAxes,
        kInputsNum
    };

    static SliceScatterKernelRefNeededInputs Create(const slice_scatter_node& node);

    const std::vector<size_t>& GetNeededInputIndexes() const;
    bool IsInputNeededInRuntime(InputIndices type) const;

private:
    std::vector<size_t> neededIndexes;
};

template <>
struct typed_program_node<slice_scatter> : public typed_program_node_base<slice_scatter> {
    using parent = typed_program_node_base<slice_scatter>;

public:
    using parent::parent;
    typed_program_node(const std::shared_ptr<slice_scatter> prim, program& prog) : parent(prim, prog) {}

    program_node& input(std::size_t i = 0) const { return get_dependency(i); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

template <>
class typed_primitive_inst<slice_scatter> : public typed_primitive_inst_base<slice_scatter> {
    using parent = typed_primitive_inst_base<slice_scatter>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(slice_scatter_node const& node, kernel_impl_params const& impl_param);
    static layout calc_output_layout(slice_scatter_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(slice_scatter_node const& node);

    typed_primitive_inst(network& network, slice_scatter_node const& desc);
    void update_output_memory() override;
    void update_shape_info_tensor(const kernel_impl_params& params) override;

private:
    void on_execute() override;
};

using slice_scatter_inst = typed_primitive_inst<slice_scatter>;

inline const std::vector<size_t>& SliceScatterKernelRefNeededInputs::GetNeededInputIndexes() const {
    return neededIndexes;
}

}  // namespace cldnn
