// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/deformable_convolution_base.hpp"

#include "itt.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;

BWDCMP_RTTI_DEFINITION(ov::op::util::DeformableConvolutionBase);

ov::op::util::DeformableConvolutionBase::DeformableConvolutionBase(const OutputVector& arguments,
                                                                   const Strides& strides,
                                                                   const CoordinateDiff& pads_begin,
                                                                   const CoordinateDiff& pads_end,
                                                                   const Strides& dilations,
                                                                   const PadType& auto_pad,
                                                                   const int64_t group,
                                                                   const int64_t deformable_group)
    : Op(arguments),
      m_strides(strides),
      m_dilations(dilations),
      m_pads_begin(pads_begin),
      m_pads_end(pads_end),
      m_auto_pad(auto_pad),
      m_group(group),
      m_deformable_group(deformable_group) {}

bool ov::op::util::DeformableConvolutionBase::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(util_DeformableConvolutionBase_visit_attributes);
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("dilations", m_dilations);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("auto_pad", m_auto_pad);
    visitor.on_attribute("group", m_group);
    visitor.on_attribute("deformable_group", m_deformable_group);
    return true;
}

void ov::op::util::DeformableConvolutionBase::validate_and_infer_types() {
    OV_OP_SCOPE(util_DeformableConvolutionBase_validate_and_infer_types);
    const PartialShape& data_batch_pshape = get_input_partial_shape(0);
    const PartialShape& offsets_pshape = get_input_partial_shape(1);
    const PartialShape& filters_pshape = get_input_partial_shape(2);

    element::Type data_batch_et = get_input_element_type(0);
    element::Type offsets_et = get_input_element_type(1);
    element::Type filters_et = get_input_element_type(2);

    element::Type result_et;
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, data_batch_et, offsets_et) &&
                              element::Type::merge(result_et, result_et, filters_et),
                          "Element types of inputs do not match. Got: data batch (",
                          data_batch_et,
                          "), offsets (",
                          offsets_et,
                          ") and filters (",
                          filters_et,
                          ")");

    NODE_VALIDATION_CHECK(this,
                          result_et.is_real() || result_et.is_integral_number(),
                          "Element type of inputs must be numeric. Got: ",
                          result_et);

    Rank result_ps_rank{};
    NODE_VALIDATION_CHECK(this,
                          Rank::merge(result_ps_rank, data_batch_pshape.rank(), offsets_pshape.rank()) &&
                              Rank::merge(result_ps_rank, result_ps_rank, filters_pshape.rank()),
                          "Ranks of inputs do not match. Got: data batch shape ",
                          data_batch_pshape,
                          ", offsets shape ",
                          offsets_pshape,
                          ", filters shape ",
                          filters_pshape);

    NODE_VALIDATION_CHECK(this, result_ps_rank.compatible(4), "Inputs must be of rank 4. Got: ", result_ps_rank);

    NODE_VALIDATION_CHECK(this, m_group > 0, "Attribute 'group' must be any value starting from 1. Got: ", m_group);

    NODE_VALIDATION_CHECK(this,
                          m_deformable_group > 0,
                          "Attribute 'deformable group' must be any value starting from 1. Got: ",
                          m_deformable_group);

    if (offsets_pshape.rank().is_static()) {
        if (offsets_pshape[1].is_static()) {
            if (filters_pshape.rank().is_static() && filters_pshape[2].is_static() && filters_pshape[3].is_static()) {
                auto offsets_channels =
                    m_deformable_group * filters_pshape[2].get_length() * filters_pshape[3].get_length() * 2;
                NODE_VALIDATION_CHECK(this,
                                      offsets_pshape[1].get_length() == offsets_channels,
                                      "The channels dimension of offsets input is not "
                                      "compatible with filters and 'deformable group' attribute. "
                                      "Offsets input shape: ",
                                      offsets_pshape,
                                      ", deformable 'group' attribute value: ",
                                      m_deformable_group,
                                      ", filters shape: ",
                                      filters_pshape);
            } else {
                // At least we can check if offsets channels is evenly divisible by deformable
                // group attribute
                NODE_VALIDATION_CHECK(this,
                                      offsets_pshape[1].get_length() % m_deformable_group == 0,
                                      "The channels dimension of offsets input must be "
                                      "evenly divisible by the 'deformable group' value along the "
                                      "channels axis. Offsets input shape: ",
                                      offsets_pshape,
                                      ", 'deformable group' attribute value: ",
                                      m_deformable_group);
            }
        }

        if (data_batch_pshape.rank().is_static()) {
            NODE_VALIDATION_CHECK(this,
                                  offsets_pshape[0].compatible(data_batch_pshape[0]),
                                  "Data batch and offsets batch dimension must be same value. Got: ",
                                  offsets_pshape[0],
                                  " and ",
                                  data_batch_pshape[0]);
        }
    }

    if (data_batch_pshape.rank().is_static() && data_batch_pshape[1].is_static()) {
        NODE_VALIDATION_CHECK(this,
                              data_batch_pshape[1].get_length() % m_group == 0,
                              "The input data shape must be evenly divisible by the 'group' value "
                              "along the channels axis. Current input shape: ",
                              data_batch_pshape,
                              ", 'group' attribute value: ",
                              m_group);
    }

    if (filters_pshape.rank().is_static() && filters_pshape[0].is_static()) {
        NODE_VALIDATION_CHECK(this,
                              filters_pshape[0].get_length() % m_group == 0,
                              "The filters shape must be evenly divisible by the 'group' value along "
                              "the channels axis. Current filters shape: ",
                              filters_pshape,
                              ", 'group' attribute value: ",
                              m_group);
    }

    // adjust filter shape to reuse regular infer_convolution_forward()
    const auto new_filters_pshape = [&](int64_t groups) {
        auto new_shape(filters_pshape);
        if (new_shape.rank().is_static()) {
            new_shape[1] *= groups;
        }
        return new_shape;
    }(m_group);
    PartialShape result_shape = ngraph::validate_and_infer_convolution_forward_output_shape(this,
                                                                                            result_ps_rank,
                                                                                            data_batch_pshape,
                                                                                            new_filters_pshape,
                                                                                            m_auto_pad,
                                                                                            m_strides,
                                                                                            m_dilations,
                                                                                            m_pads_begin,
                                                                                            m_pads_end);

    if (result_shape.rank().is_static() && offsets_pshape.rank().is_static()) {
        PartialShape result_spatial_shape = [&result_shape]() {
            vector<Dimension> result_spatial_dims{result_shape};
            result_spatial_dims.erase(result_spatial_dims.begin(), result_spatial_dims.begin() + 2);
            return PartialShape{result_spatial_dims};
        }();

        PartialShape offsets_spatial_shape = [&offsets_pshape]() {
            vector<Dimension> offsets_spatial_dims{offsets_pshape};
            offsets_spatial_dims.erase(offsets_spatial_dims.begin(), offsets_spatial_dims.begin() + 2);
            return PartialShape{offsets_spatial_dims};
        }();

        NODE_VALIDATION_CHECK(this,
                              offsets_spatial_shape.compatible(result_spatial_shape),
                              "Spatial dimensions of offsets and output must be equal. Got: ",
                              offsets_spatial_shape,
                              " and ",
                              result_spatial_shape);

        if (result_shape[0].is_dynamic()) {
            result_shape[0] = offsets_pshape[0];  // batch size
        }
    }
    set_output_type(0, result_et, result_shape);
}
