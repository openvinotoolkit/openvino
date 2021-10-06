#include "ngraph/op/max_unpool.hpp"

#include "ngraph/attribute_visitor.hpp"

#include "itt.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v8::MaxUnpool);

op::v8::MaxUnpool::MaxUnpool(const ngraph::Output<ngraph::Node>& poolInp,
                             const ngraph::Output<ngraph::Node>& poolOut,
                             const ngraph::Output<ngraph::Node>& inp,
                             const ngraph::Output<ngraph::Node>& shape,
                             const Strides& strides,
                             const Shape& pads_begin,
                             const Shape& pads_end,
                             const Shape& kernel)
    : Op({poolInp, poolOut, inp, shape}), m_kernel(kernel), m_pads_begin(pads_begin), m_pads_end(pads_end), m_strides(strides) {
    constructor_validate_and_infer_types();
}

bool op::v8::MaxUnpool::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("kernel_shape", m_kernel);
    visitor.on_attribute("pad_begin", m_pads_begin);
    visitor.on_attribute("pad_end", m_pads_end);
    visitor.on_attribute("strides", m_strides);
    return true;
}

std::shared_ptr<ngraph::Node> op::v8::MaxUnpool::clone_with_new_inputs(const OutputVector& new_args) const {
    if (new_args.size() != 4) {
        throw ngraph::ngraph_error("Incorrect number of new arguments");
    }
    return std::make_shared<MaxUnpool>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3),
                                       m_strides, m_pads_begin, m_pads_end, m_strides);
}

void op::v8::MaxUnpool::validate_and_infer_types() {
    // std::cout << "validate_and_infer_types" << std::endl;
    // std::cout << m_kernel << std::endl;
    // std::cout << m_pads_begin << std::endl;
    // std::cout << m_pads_end << std::endl;
    // std::cout << m_strides << std::endl;

    // auto kenel_size = m_kernel.size();
    // if(m_kernel[kenel_size-2] != 2 || m_kernel[kenel_size-1] != 2){
    //     throw std::invalid_argument( "Unsupported kernel shape! Only [2, 2] kernel is supported." );
    // }

    auto outShape = get_input_partial_shape(3);
    auto poolInpShape = get_input_partial_shape(0);
    if (poolInpShape.is_static()) {  // need to fix it
        auto inpShape = poolInpShape.to_shape();
        outShape[0] = inpShape[0];  // Use only spatial dimensions from shape
        outShape[1] = inpShape[1];  // and restore batch and channels
        set_output_type(0, get_input_element_type(0), outShape);
    }
}

namespace maxunpool {
template <typename T>
void maxunpool(const T* poolInp, const T* poolOut, const T* inp, T* out,
               const Shape& outDims, const Shape& poolInpDims, const Shape& poolOutDims) {
    const size_t batch = poolInpDims[0];
    const size_t channels = poolInpDims[1];
    const size_t height = poolInpDims[2];
    const size_t width = poolInpDims[3];
    const size_t outHeight = outDims[2];
    const size_t outWidth = outDims[3];
    const size_t poolOutHeight = poolOutDims[2];
    const size_t poolOutWidth = poolOutDims[3];

    size_t mask_size = 1, out_size = 1;
    for (int i = 0; i < 4; ++i) {
        mask_size *= poolOutDims[i];
        out_size *= outDims[i];
    }
    std::vector<bool> mask(mask_size, false);
    memset(out, 0, out_size * sizeof(float));

    for (size_t d = 0; d < batch * channels; ++d) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int poolOutIdx = (d * poolOutHeight + y / 2) * poolOutWidth + x / 2;
                int poolInpIdx = (d * height + y) * width + x;
                int dstIdx = d * outHeight * outWidth + (y * width + x);
                if (fabs(poolInp[poolInpIdx] - poolOut[poolOutIdx]) < 1e-5f && !mask[poolOutIdx]) {
                    out[dstIdx] = inp[poolOutIdx];
                    mask[poolOutIdx] = true;
                }
            }
        }
    }
}

template <element::Type_t ET>
inline bool evaluate(const HostTensorPtr& arg0, const HostTensorPtr& arg1, const HostTensorPtr& arg2, const HostTensorPtr& out, 
                     const Shape& outDims, const Shape& poolInpDims, const Shape& poolOutDims) {
    using T = typename element_type_traits<ET>::value_type;
    maxunpool<T>(arg0->get_data_ptr<ET>(), arg1->get_data_ptr<ET>(), 
                 arg2->get_data_ptr<ET>(), out->get_data_ptr<ET>(), 
                 outDims, poolInpDims, poolOutDims);
    return true;
}

bool evaluate_maxunpool(const HostTensorPtr& arg0, const HostTensorPtr& arg1, const HostTensorPtr& arg2, const HostTensorPtr& out, 
                        const Shape& outDims, const Shape& poolInpDims, const Shape& poolOutDims) {
    bool rc = true;
    switch (arg0->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_maxunpool, f16, arg0, arg1, arg2, out, outDims, poolInpDims, poolOutDims);
        NGRAPH_TYPE_CASE(evaluate_maxunpool, f32, arg0, arg1, arg2, out, outDims, poolInpDims, poolOutDims);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace maxunpool

bool op::v8::MaxUnpool::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    const Shape outDims = get_default_output().get_shape();
    const Shape poolInpDims = get_input_partial_shape(0).to_shape();
    const Shape poolOutDims = get_input_partial_shape(1).to_shape();
    return maxunpool::evaluate_maxunpool(inputs[0], inputs[1], inputs[2], outputs[0], 
                                         outDims, poolInpDims, poolOutDims);
}

bool op::v8::MaxUnpool::has_evaluate() const {
    switch (get_input_element_type(0)) {
    case ngraph::element::f16:
    case ngraph::element::f32:
        return true;
    default:
        break;
    }
    return false;
}
