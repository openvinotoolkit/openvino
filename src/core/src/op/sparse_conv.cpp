// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/sparse_conv.hpp"

#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/runtime/host_tensor.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(ov::op::v1::SparseConv);

ov::op::v1::SparseConv::SparseConv(const Output<Node>& features,
                                   const Output<Node>& inp_pos,
                                   const Output<Node>& out_pos,
                                   const Output<Node>& kernel,
                                   const Output<Node>& offset,
                                   const Output<Node>& voxel_size)
    : Op({features, inp_pos, out_pos, kernel, offset, voxel_size}) {
    constructor_validate_and_infer_types();
}

bool op::v1::SparseConv::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v1_SparseConv_visit_attributes);
    return true;
}

shared_ptr<Node> op::v1::SparseConv::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v1_SparseConv_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v1::SparseConv>(new_args.at(0),
                                           new_args.at(1),
                                           new_args.at(2),
                                           new_args.at(3),
                                           new_args.at(4),
                                           new_args.at(5));
}

void op::v1::SparseConv::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(v1_SparseConv_validate_and_infer_types);
    auto outShape = get_input_partial_shape(2);
    auto kernelShape = get_input_partial_shape(3);
    outShape[1] = kernelShape[0];
    set_output_type(0, get_input_element_type(0), outShape);
}

bool op::v1::SparseConv::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    NGRAPH_OP_SCOPE(v1_SparseConv_evaluate);

    const float* features = inputs[0]->get_data_ptr<float>();
    const float* inpPos = inputs[1]->get_data_ptr<float>();
    const float* outPos = inputs[2]->get_data_ptr<float>();
    const float* kernel = inputs[3]->get_data_ptr<float>();
    const float* offset = inputs[4]->get_data_ptr<float>();
    const float* voxel_size = inputs[5]->get_data_ptr<float>();
    float* out = outputs[0]->get_data_ptr<float>();

    OPENVINO_ASSERT(voxel_size[0] == 1.0f, "Only voxel_size=1 is supported");

    const Shape outDims = get_default_output().get_shape();
    memset(out, 0, sizeof(float) * outDims[0] * outDims[1]);

    size_t numInpPoints = get_input_partial_shape(1).to_shape()[0];
    const size_t numOutPoints = get_input_partial_shape(2).to_shape()[0];
    Shape kernelDims = get_input_partial_shape(3).to_shape();

    // Kernel layout is OCxICxDxHxW
    const int OC = kernelDims[0];
    const int IC = kernelDims[1];
    const int kd = kernelDims[2];
    const int kh = kernelDims[3];
    const int kw = kernelDims[4];
    const int kernelPlane = kd * kh * kw;

    // See https://github.com/isl-org/Open3D/blob/master/python/open3d/ml/torch/python/layers/convolutions.py
    float rw = kw * 0.51f;
    float rh = kh * 0.51f;
    float rd = kd * 0.51f;

    for (size_t i = 0; i < numInpPoints; ++i) {
        if (inpPos[i * 3] < 0) {
            numInpPoints = i;
            break;
        }
    }

    for (size_t i = 0; i < numOutPoints; ++i) {
        const float xi = outPos[i * 3] - offset[0];
        const float yi = outPos[i * 3 + 1] - offset[1];
        const float zi = outPos[i * 3 + 2] - offset[2];

        // Accumulate features which inside the kernel
        for (size_t j = 0; j < numInpPoints; ++j) {
            const float xj = inpPos[j * 3];
            const float yj = inpPos[j * 3 + 1];
            const float zj = inpPos[j * 3 + 2];

            if (xi - rw <= xj && xj <= xi + rw && yi - rh <= yj && yj <= yi + rh && zi - rd <= zj && zj <= zi + rd) {
                const int w = std::min(static_cast<int>(xj - xi + kw * 0.5f), kw - 1);
                const int h = std::min(static_cast<int>(yj - yi + kh * 0.5f), kh - 1);
                const int d = std::min(static_cast<int>(zj - zi + kd * 0.5f), kd - 1);

                const float* featuresOffset = features + j * IC;
                const float* kernelOffset = kernel + (d * kh + h) * kw + w;
                for (size_t oc = 0; oc < OC; ++oc) {
                    for (size_t ic = 0; ic < IC; ++ic) {
                        out[i * OC + oc] += kernelOffset[0] * featuresOffset[ic];
                        kernelOffset += kernelPlane;
                    }
                }
            }
        }
    }
    return true;
}

bool op::v1::SparseConv::has_evaluate() const {
    NGRAPH_OP_SCOPE(v1_SparseConv_has_evaluate);
    return true;
}
