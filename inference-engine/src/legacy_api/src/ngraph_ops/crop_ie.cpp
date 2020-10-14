// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/ngraph_ops/crop_ie.hpp"

#include <algorithm>
#include <memory>
#include <vector>

#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::CropIE::type_info;

op::CropIE::CropIE(const Output<Node>& data, std::vector<int64_t> axes, std::vector<int64_t> dim,
                   std::vector<int64_t> offset)
    : Op({data}), axes(axes), dim(dim), offset(offset) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::CropIE::clone_with_new_inputs(const OutputVector& new_args) const {
    if (new_args.size() != 1) {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<CropIE>(new_args.at(0), axes, dim, offset);
}

void op::CropIE::validate_and_infer_types() {
    auto input_shape = get_input_partial_shape(0).to_shape();
    NODE_VALIDATION_CHECK(this, axes.size() == dim.size(), "axes and dim needs to have same number of values");

    NODE_VALIDATION_CHECK(this, axes.size() == offset.size(), "axes and offset needs to have same number of values");

    ngraph::Shape output_shape(input_shape);
    for (int i = 0; i < axes.size(); ++i) {
        NODE_VALIDATION_CHECK(this, axes[i] >= 0 && axes[i] < static_cast<int64_t>(output_shape.size()),
                              "axes should be positive and less than number of input dims");
        output_shape[axes[i]] = dim[i];
    }

    set_output_type(0, get_input_element_type(0), PartialShape(output_shape));
}

bool op::CropIE::evaluate(const HostTensorVector &outputs, const HostTensorVector &inputs) const {
    if (inputs.front()->get_element_type() != outputs.front()->get_element_type()) {
        throw ngraph_error("Input and output data types must be the same!");
    }

    auto *dst_ptr = outputs.front()->get_data_ptr<uint8_t>();

    const int ndims = dim.size();

    const size_t OFFSET_N = (ndims > 0) ? offset.at(0) : 0;
    const size_t OFFSET_C = (ndims > 1) ? offset.at(1) : 0;
    const size_t OFFSET_D = (ndims > 4) ? offset.at(ndims - 3) : 0;
    const size_t OFFSET_H = (ndims > 2) ? offset.at(ndims - 2) : 0;
    const size_t OFFSET_W = (ndims > 3) ? offset.at(ndims - 1) : 0;

    auto outputShape = get_output_partial_shape(0).get_shape();

    const size_t ON = (ndims > 0) ? outputShape.at(0) : 1;
    const size_t OC = (ndims > 1) ? outputShape.at(1) : 1;
    const size_t OD = (ndims > 4) ? outputShape.at(ndims - 3) : 1;
    const size_t OH = (ndims > 2) ? outputShape.at(ndims - 2) : 1;
    const size_t OW = (ndims > 3) ? outputShape.at(ndims - 1) : 1;

    auto inputShape = get_input_partial_shape(0).get_shape();

    const size_t IN = (ndims > 0) ? inputShape.at(0) : 1;
    const size_t IC = (ndims > 1) ? inputShape.at(1) : 1;
    const size_t ID = (ndims > 4) ? inputShape.at(ndims - 3) : 1;
    const size_t IH = (ndims > 2) ? inputShape.at(ndims - 2) : 1;
    const size_t IW = (ndims > 3) ? inputShape.at(ndims - 1) : 1;

    auto dst_off = [=](size_t n, size_t c, size_t d, size_t h, size_t w) -> size_t {
        return (n * OC * OD * OH * OW + c * OD * OH * OW + d * OH * OW + h * OW + w);
    };
    auto src_off = [=](size_t n, size_t c, size_t d, size_t h, size_t w) -> size_t {
        return (n * IC * ID * IH * IW + c * ID * IH * IW + d * IH * IW + h * IW + w);
    };

    if (IN - OFFSET_N < ON) {
        throw ngraph_error("Wrong offset!");
    }
    if (IC - OFFSET_C < OC) {
        throw ngraph_error("Wrong offset!");
    }
    if (IC - OFFSET_C < OC) {
        throw ngraph_error("Wrong offset!");
    }
    if (ID - OFFSET_D < OD) {
        throw ngraph_error("Wrong offset!");
    }
    if (IH - OFFSET_H < OH) {
        throw ngraph_error("Wrong offset!");
    }
    if (IW - OFFSET_W < OW) {
        throw ngraph_error("Wrong offset!");
    }

    size_t dataSize = inputs.front()->get_element_type().size();

    auto src_ptr = inputs.front()->get_data_ptr<const uint8_t>();
    for (size_t n = 0; n < ON; ++n) {
        for (size_t c = 0; c < OC; ++c) {
            for (size_t d = 0; d < OD; ++d) {
                for (size_t h = 0; h < OH; ++h) {
                    for (size_t w = 0; w < OW; ++w) {
                        memcpy(dst_ptr + dataSize * dst_off(n, c, d, h, w),
                               src_ptr + dataSize * src_off(n + OFFSET_N, c + OFFSET_C, d + OFFSET_D, h + OFFSET_H, w + OFFSET_W),
                               dataSize);
                    }
                }
            }
        }
    }

    return true;
}
