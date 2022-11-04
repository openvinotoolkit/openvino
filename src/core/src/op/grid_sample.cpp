// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/grid_sample.hpp"

#include "grid_sample_shape_inference.hpp"
#include "itt.hpp"
#include "ngraph/runtime/reference/grid_sample.hpp"
#include "ngraph/validation_util.hpp"

namespace ov {
op::v9::GridSample::GridSample(const Output<Node>& data, const Output<Node>& grid, const Attributes& attributes)
    : op::Op{{data, grid}},
      m_attributes{attributes} {
    constructor_validate_and_infer_types();
}

bool op::v9::GridSample::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v9_GridSample_visit_attributes);
    visitor.on_attribute("align_corners", m_attributes.align_corners);
    visitor.on_attribute("mode", m_attributes.mode);
    visitor.on_attribute("padding_mode", m_attributes.padding_mode);
    return true;
}

void op::v9::GridSample::validate_and_infer_types() {
    OV_OP_SCOPE(v9_GridSample_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(1).is_real(),
                          "The element type of the grid input tensor must be a floating point type.");

    std::vector<PartialShape> out_shapes(1);
    shape_infer(this, {get_input_partial_shape(0), get_input_partial_shape(1)}, out_shapes);
    set_output_type(0, get_input_element_type(0), out_shapes[0]);
}

std::shared_ptr<Node> op::v9::GridSample::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v9_GridSample_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<op::v9::GridSample>(new_args.at(0), new_args.at(1), this->get_attributes());
}

std::ostream& operator<<(std::ostream& s, const op::v9::GridSample::InterpolationMode& mode) {
    return s << as_string(mode);
}

std::ostream& operator<<(std::ostream& s, const op::v9::GridSample::PaddingMode& padding_mode) {
    return s << as_string(padding_mode);
}

template <>
NGRAPH_API EnumNames<op::v9::GridSample::InterpolationMode>& EnumNames<op::v9::GridSample::InterpolationMode>::get() {
    static auto enum_names =
        EnumNames<op::v9::GridSample::InterpolationMode>("op::v9::GridSample::InterpolationMode",
                                                         {{"bilinear", op::v9::GridSample::InterpolationMode::BILINEAR},
                                                          {"bicubic", op::v9::GridSample::InterpolationMode::BICUBIC},
                                                          {"nearest", op::v9::GridSample::InterpolationMode::NEAREST}});
    return enum_names;
}

template <>
NGRAPH_API EnumNames<op::v9::GridSample::PaddingMode>& EnumNames<op::v9::GridSample::PaddingMode>::get() {
    static auto enum_names =
        EnumNames<op::v9::GridSample::PaddingMode>("op::v9::GridSample::PaddingMode",
                                                   {{"zeros", op::v9::GridSample::PaddingMode::ZEROS},
                                                    {"border", op::v9::GridSample::PaddingMode::BORDER},
                                                    {"reflection", op::v9::GridSample::PaddingMode::REFLECTION}});
    return enum_names;
}

namespace {

template <element::Type_t DATA_ET, element::Type_t GRID_ET>
bool evaluate_exec(const HostTensorPtr& output,
                   const HostTensorPtr& data,
                   const HostTensorPtr& grid,
                   const op::v9::GridSample::Attributes& attributes) {
    ngraph::runtime::reference::grid_sample(output->get_data_ptr<DATA_ET>(),
                                            data->get_data_ptr<DATA_ET>(),
                                            grid->get_data_ptr<GRID_ET>(),
                                            data->get_shape(),
                                            grid->get_shape(),
                                            attributes.align_corners,
                                            attributes.mode,
                                            attributes.padding_mode);
    return true;
}

#define GRID_SAMPLE_TYPE_CASE(a, ...)                                 \
    case element::Type_t::a: {                                        \
        OV_OP_SCOPE(OV_PP_CAT3(evaluate_exec_grid_sample, _, a));     \
        rc = evaluate_exec<DATA_ET, element::Type_t::a>(__VA_ARGS__); \
    } break

template <element::Type_t DATA_ET>
bool evaluate(const HostTensorPtr& output,
              const HostTensorPtr& data,
              const HostTensorPtr& grid,
              const op::v9::GridSample::Attributes& attributes) {
    auto rc = true;
    switch (grid->get_element_type()) {
        GRID_SAMPLE_TYPE_CASE(f32, output, data, grid, attributes);
    default:
        rc = false;
        break;
    }
    return rc;
}

bool evaluate_grid_sample(const HostTensorPtr& output,
                          const HostTensorPtr& data,
                          const HostTensorPtr& grid,
                          const op::v9::GridSample::Attributes& attributes) {
    auto rc = true;
    switch (output->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_grid_sample, f32, output, data, grid, attributes);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace

bool op::v9::GridSample::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v9_GridSample_evaluate);
    OPENVINO_ASSERT(ngraph::validate_host_tensor_vector(inputs, 2), "Invalid GridSample input TensorVector.");
    OPENVINO_ASSERT(ngraph::validate_host_tensor_vector(outputs, 1), "Invalid GridSample output TensorVector.");

    return evaluate_grid_sample(outputs[0], inputs[0], inputs[1], m_attributes);
}

bool op::v9::GridSample::has_evaluate() const {
    return get_input_element_type(0) == element::f32 && get_input_element_type(1) == element::f32;
}
}  // namespace ov
