// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/grid_sample.hpp"

#include "element_visitor.hpp"
#include "grid_sample_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/reference/grid_sample.hpp"

namespace ov {
namespace op {
namespace v9 {

struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(Tensor& output,
                             const Tensor& data,
                             const Tensor& grid,
                             const Shape& data_shape,
                             const Shape& grid_shape,
                             const GridSample::Attributes& attributes) {
        using namespace ov::element;
        return IF_TYPE_OF(eval_by_grid_type,
                          OV_PP_ET_LIST(f32),
                          EvalByGridType,
                          grid.get_element_type(),
                          output.data<T>(),
                          data.data<const T>(),
                          grid,
                          data_shape,
                          grid_shape,
                          attributes);
    }

private:
    struct EvalByGridType : public element::NoAction<bool> {
        using element::NoAction<bool>::visit;

        template <element::Type_t ET, class T, class G = fundamental_type_for<ET>>
        static result_type visit(T* output,
                                 const T* data,
                                 const Tensor& grid,
                                 const Shape& data_shape,
                                 const Shape& grid_shape,
                                 const GridSample::Attributes& attributes) {
            reference::grid_sample(output,
                                   data,
                                   grid.data<const G>(),
                                   data_shape,
                                   grid_shape,
                                   attributes.align_corners,
                                   attributes.mode,
                                   attributes.padding_mode);
            return true;
        }
    };
};

GridSample::GridSample(const Output<Node>& data, const Output<Node>& grid, const Attributes& attributes)
    : op::Op{{data, grid}},
      m_attributes{attributes} {
    constructor_validate_and_infer_types();
}

bool GridSample::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v9_GridSample_visit_attributes);
    visitor.on_attribute("align_corners", m_attributes.align_corners);
    visitor.on_attribute("mode", m_attributes.mode);
    visitor.on_attribute("padding_mode", m_attributes.padding_mode);
    return true;
}

void GridSample::validate_and_infer_types() {
    OV_OP_SCOPE(v9_GridSample_validate_and_infer_types);
    if (!get_input_element_type(1).is_dynamic()) {
        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(1).is_real(),
                              "The element type of the grid input tensor must be a floating point type.");
    }

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto out_shapes = shape_infer(this, input_shapes);
    set_output_type(0, get_input_element_type(0), out_shapes[0]);
}

std::shared_ptr<Node> GridSample::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v9_GridSample_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<GridSample>(new_args.at(0), new_args.at(1), get_attributes());
}

bool GridSample::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v9_GridSample_evaluate);

    OPENVINO_ASSERT(outputs.size() == 1);

    const auto& out_shape = shape_infer(this, ov::util::get_tensors_partial_shapes(inputs)).front().to_shape();
    outputs[0].set_shape(out_shape);

    using namespace ov::element;
    return IF_TYPE_OF(v9_GridSample_evaluate,
                      OV_PP_ET_LIST(f32),
                      Evaluate,
                      inputs[0].get_element_type(),
                      outputs[0],
                      inputs[0],
                      inputs[1],
                      inputs[0].get_shape(),
                      inputs[1].get_shape(),
                      m_attributes);
}

bool GridSample::has_evaluate() const {
    return get_input_element_type(0) == element::f32 && get_input_element_type(1) == element::f32;
}
}  // namespace v9
}  // namespace op

std::ostream& operator<<(std::ostream& s, const op::v9::GridSample::InterpolationMode& mode) {
    return s << as_string(mode);
}

std::ostream& operator<<(std::ostream& s, const op::v9::GridSample::PaddingMode& padding_mode) {
    return s << as_string(padding_mode);
}

template <>
OPENVINO_API EnumNames<op::v9::GridSample::InterpolationMode>& EnumNames<op::v9::GridSample::InterpolationMode>::get() {
    static auto enum_names =
        EnumNames<op::v9::GridSample::InterpolationMode>("op::v9::GridSample::InterpolationMode",
                                                         {{"bilinear", op::v9::GridSample::InterpolationMode::BILINEAR},
                                                          {"bicubic", op::v9::GridSample::InterpolationMode::BICUBIC},
                                                          {"nearest", op::v9::GridSample::InterpolationMode::NEAREST}});
    return enum_names;
}

template <>
OPENVINO_API EnumNames<op::v9::GridSample::PaddingMode>& EnumNames<op::v9::GridSample::PaddingMode>::get() {
    static auto enum_names =
        EnumNames<op::v9::GridSample::PaddingMode>("op::v9::GridSample::PaddingMode",
                                                   {{"zeros", op::v9::GridSample::PaddingMode::ZEROS},
                                                    {"border", op::v9::GridSample::PaddingMode::BORDER},
                                                    {"reflection", op::v9::GridSample::PaddingMode::REFLECTION}});
    return enum_names;
}

AttributeAdapter<op::v9::GridSample::InterpolationMode>::~AttributeAdapter() = default;
AttributeAdapter<op::v9::GridSample::PaddingMode>::~AttributeAdapter() = default;
}  // namespace ov
