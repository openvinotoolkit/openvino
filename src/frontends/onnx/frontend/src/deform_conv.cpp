#include "deform_conv.hpp"

using namespace std;
using namespace ov;
using namespace ov::op::v1;

DeformConv::DeformConv(const Output<Node>& data, const Output<Node>& offsets, const Output<Node>& weights)
    : Op({data, offsets, weights}) {
    constructor_validate_and_infer_types();
}

void DeformConv::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

shared_ptr<Node> DeformConv::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<DeformConv>(new_args.at(0), new_args.at(1), new_args.at(2));
}
