// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/non_max_suppression.hpp"
#include "primitive_inst.h"

#include <memory>
#include <string>

namespace cldnn {

template <>
struct typed_program_node<non_max_suppression> : public typed_program_node_base<non_max_suppression> {
    using parent = typed_program_node_base<non_max_suppression>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program& prog)
        : parent(prim, prog)
    {}

    program_node& input() const { return get_dependency(0); }

    program_node& input_boxes() const {
        return get_dependency(0);
    }

    program_node& input_scores() const {
        return get_dependency(1);
    }

    bool has_num_select_per_class() const { return !get_primitive()->num_select_per_class.empty(); }
    program_node& num_select_per_class_node() const {
        return get_dependency(2);
    }

    bool has_iou_threshold() const { return !get_primitive()->iou_threshold.empty(); }
    program_node& iou_threshold_node() const {
        size_t offset = 2;
        offset += has_num_select_per_class();
        return get_dependency(offset);
    }

    bool has_score_threshold() const { return !get_primitive()->score_threshold.empty(); }
    program_node& score_threshold_node() const {
        size_t offset = 2;
        offset += has_num_select_per_class();
        offset += has_iou_threshold();
        return get_dependency(offset);
    }

    bool has_soft_nms_sigma() const { return !get_primitive()->soft_nms_sigma.empty(); }
    program_node& soft_nms_sigma_node() const {
        size_t offset = 2;
        offset += has_num_select_per_class();
        offset += has_iou_threshold();
        offset += has_score_threshold();
        return get_dependency(offset);
    }

    bool has_second_output() const { return !get_primitive()->second_output.empty(); }
    program_node& second_output_node() const {
        size_t offset = 2;
        offset += has_num_select_per_class();
        offset += has_iou_threshold();
        offset += has_score_threshold();
        offset += has_soft_nms_sigma();
        return get_dependency(offset);
    }

    bool has_third_output() const { return !get_primitive()->third_output.empty(); }
    program_node& third_output_node() const {
        size_t offset = 2;
        offset += has_num_select_per_class();
        offset += has_iou_threshold();
        offset += has_score_threshold();
        offset += has_soft_nms_sigma();
        offset += has_second_output();
        return get_dependency(offset);
    }
    bool use_multiple_outputs() const { return get_primitive()->output_size() == 3; }

    std::vector<size_t> get_shape_infer_dependencies() const override { return {2}; }
};

using non_max_suppression_node = typed_program_node<non_max_suppression>;

template <>
class typed_primitive_inst<non_max_suppression> : public typed_primitive_inst_base<non_max_suppression> {
    using parent = typed_primitive_inst_base<non_max_suppression>;
    using parent::parent;

    size_t get_iou_threshold_offset() const {
        size_t offset = 2;
        offset += has_num_select_per_class();
        return offset;
    }

    size_t get_score_threshold_offset() const {
        size_t offset = 2;
        offset += has_num_select_per_class();
        offset += has_iou_threshold();
        return offset;
    }

    size_t get_soft_nms_sigma_offset() const {
        size_t offset = 2;
        offset += has_num_select_per_class();
        offset += has_iou_threshold();
        offset += has_score_threshold();
        return offset;
    }

public:
    typed_primitive_inst(network& network, non_max_suppression_node const& node)
        : parent(network, node)
    {}

    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(non_max_suppression_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(non_max_suppression_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(non_max_suppression_node const& node);

    memory::ptr input_boxes_mem() const {
        return dep_memory_ptr(0);
    }

    memory::ptr input_scores_mem() const {
        return dep_memory_ptr(1);
    }

    bool has_num_select_per_class() const { return !get_typed_desc<non_max_suppression>()->num_select_per_class.empty(); }
    memory::ptr num_select_per_class_mem() const {
        return dep_memory_ptr(2);
    }
    const primitive_inst* num_select_per_class_inst() const {
        return dependencies().at(2).first;
    }

    bool has_iou_threshold() const { return !get_typed_desc<non_max_suppression>()->iou_threshold.empty(); }
    memory::ptr iou_threshold_mem() const {
        return dep_memory_ptr(get_iou_threshold_offset());
    }
    const primitive_inst* iou_threshold_inst() const {
        return dependencies().at(get_iou_threshold_offset()).first;
    }

    bool has_score_threshold() const { return !get_typed_desc<non_max_suppression>()->score_threshold.empty(); }
    memory::ptr score_threshold_mem() const {
        return dep_memory_ptr(get_score_threshold_offset());
    }
    const primitive_inst* score_threshold_inst() const {
        return dependencies().at(get_score_threshold_offset()).first;
    }

    bool has_soft_nms_sigma() const { return !get_typed_desc<non_max_suppression>()->soft_nms_sigma.empty(); }
    memory::ptr soft_nms_sigma_mem() const {
        return dep_memory_ptr(get_soft_nms_sigma_offset());
    }
    const primitive_inst* soft_nms_sigma_inst() const {
        return dependencies().at(get_soft_nms_sigma_offset()).first;
    }

    bool has_second_output() const { return !get_typed_desc<non_max_suppression>()->second_output.empty(); }
    memory::ptr second_output_mem() const {
        size_t offset = 2;
        offset += has_num_select_per_class();
        offset += has_iou_threshold();
        offset += has_score_threshold();
        offset += has_soft_nms_sigma();
        return dep_memory_ptr(offset);
    }

    bool has_third_output() const { return !get_typed_desc<non_max_suppression>()->third_output.empty(); }
    memory::ptr third_output_mem() const {
        size_t offset = 2;
        offset += has_num_select_per_class();
        offset += has_iou_threshold();
        offset += has_score_threshold();
        offset += has_soft_nms_sigma();
        offset += has_second_output();
        return dep_memory_ptr(offset);
    }
};

using non_max_suppression_inst = typed_primitive_inst<non_max_suppression>;

template <>
struct typed_program_node<non_max_suppression_gather> : typed_program_node_base<non_max_suppression_gather> {
    using parent = typed_program_node_base<non_max_suppression_gather>;
    using parent::parent;

public:
    typed_program_node(const std::shared_ptr<non_max_suppression_gather> prim, program& prog) : parent(prim, prog) {
        can_be_optimized(true);
        set_runtime_skippable(true);
    }

    std::vector<size_t> get_shape_infer_dependencies() const override { return {0, 1, 2}; }
};

using non_max_suppression_gather_node = typed_program_node<non_max_suppression_gather>;

template <>
class typed_primitive_inst<non_max_suppression_gather> : public typed_primitive_inst_base<non_max_suppression_gather> {
public:
    using parent = typed_primitive_inst_base<non_max_suppression_gather>;
    using parent::parent;

    static layout calc_output_layout(const non_max_suppression_gather_node& node, const kernel_impl_params& impl_param);
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(const non_max_suppression_gather_node& node, const kernel_impl_params& impl_param);
    static std::string to_string(const non_max_suppression_gather_node& node);

    typed_primitive_inst(network& network, non_max_suppression_gather_node const& node);
    void update_output_memory() override;

private:
    void on_execute() override;
};

using non_max_suppression_gather_inst = typed_primitive_inst<non_max_suppression_gather>;

}  // namespace cldnn
