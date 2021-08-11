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
    std::cout<< "MAX_UNPOOL" << std::endl;
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

bool op::v8::MaxPoolGrad::evaluate(const HostTensorVector& outputs,
                               const HostTensorVector& inputs) const
{
    return true;
}