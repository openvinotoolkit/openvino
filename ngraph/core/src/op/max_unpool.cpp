#include "ngraph/op/max_unpool.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v8::MaxPoolGrad, "MaxPoolGrad", 1);

op::v8::MaxPoolGrad::MaxPoolGrad(const ngraph::Output<ngraph::Node>& poolInp,
                   const ngraph::Output<ngraph::Node>& poolOut,
                   const ngraph::Output<ngraph::Node>& inp,
                   const ngraph::Output<ngraph::Node>& shape): Op({poolInp, poolOut, inp, shape}) 
{
    constructor_validate_and_infer_types();
}

bool op::v8::MaxPoolGrad::visit_attributes(ngraph::AttributeVisitor &visitor) {
    return true;
}

std::shared_ptr<ngraph::Node> op::v8::MaxPoolGrad::clone_with_new_inputs(const ngraph::OutputVector &new_args) const {
    if (new_args.size() != 4) {
        throw ngraph::ngraph_error("Incorrect number of new arguments");
    }
    return std::make_shared<MaxPoolGrad>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}

void op::v8::MaxPoolGrad::validate_and_infer_types() {
    auto outShape = get_input_partial_shape(3);
    auto poolInpShape = get_input_partial_shape(0);
    if (poolInpShape.is_static()) {
        auto inpShape = poolInpShape.to_shape();
        outShape[0] = inpShape[0];  // Use only spatial dimensions from shape
        outShape[1] = inpShape[1];  // and restore batch and channels
        set_output_type(0, get_input_element_type(0), outShape);
    }
}

bool op::v8::MaxPoolGrad::evaluate(const HostTensorVector& outputs,
                               const HostTensorVector& inputs) const
{
    std::cout << "Unpool EVALUATE" << std::endl;
    
    const float* poolInp = inputs[0]->get_data_ptr<float>();
    const float* poolOut = inputs[1]->get_data_ptr<float>();
    const float* inp     = inputs[2]->get_data_ptr<float>();
    float* out = outputs[0]->get_data_ptr<float>();

    Shape outDims = get_default_output().get_shape();
    Shape poolInpDims = get_input_partial_shape(0).to_shape();
    Shape poolOutDims = get_input_partial_shape(1).to_shape();

    const size_t batch    = poolInpDims[0];
    const size_t channels = poolInpDims[1];
    const size_t height   = poolInpDims[2];
    const size_t width    = poolInpDims[3];
    const size_t outHeight = outDims[2];
    const size_t outWidth  = outDims[3];
    const size_t poolOutHeight = poolOutDims[2];
    const size_t poolOutWidth  = poolOutDims[3];

    // size_t out_size = poolOutDims[0] * poolOutDims[1] * poolOutDims[2] * poolOutDims[3];
    // std::vector<bool> mask(out_size, false);
    // memset(out, 0, out_size * sizeof(bool));
    // for (size_t d = 0; d < batch * channels; ++d) {
    //     for (int y = 0; y < height; ++y) {
    //         for (int x = 0; x < width; ++x) {
    //             int poolOutIdx = (d * poolOutHeight + y / 2) * poolOutWidth + x / 2;
    //             int poolInpIdx = (d * height + y) * width + x;
    //             int dstIdx = d * outHeight * outWidth + (y * width + x);
    //             if (fabs(poolInp[poolInpIdx] - poolOut[poolOutIdx]) < 1e-5f && !mask[poolOutIdx]) {
    //                 out[dstIdx] = inp[poolOutIdx];
    //                 mask[poolOutIdx] = true;
    //             }
    //         }
    //     }
    // }
    
    for (size_t d = 0; d < batch*channels; ++d) {
        for (int y = height - 1; y >= 0; --y) {
            for (int x = width - 1; x >= 0; --x) {
                int poolOutIdx = (d * poolOutHeight + y / 2) * poolOutWidth + x / 2;
                int poolInpIdx = (d * height + y) * width + x;
                int dstIdx = d * outHeight * outWidth + (y * width + x);
                if (fabs(poolInp[poolInpIdx] - poolOut[poolOutIdx]) < 1e-6f) {
                    out[dstIdx] = inp[poolOutIdx];
                } else {
                    out[dstIdx] = 0;
                }
            }
        }
    }
    return true;
}

bool op::v8::MaxPoolGrad::has_evaluate() const {
    return true;
}
