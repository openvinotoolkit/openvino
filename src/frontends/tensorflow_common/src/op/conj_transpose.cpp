#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/concat.hpp"
#include "common_op_table.hpp"
#include "openvino/op/matmul.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_conj_transpose_op(const NodeContext& node){
	default_op_checks(node, 2, {"conj_transpose"});

	auto op_type = node.get_op_type();
	auto x = node.get_input(0);
	auto perm = node.get_input(1);
	
	
	auto complex_type_mark = as_type_ptr<ComplexTypeMark>(x.get_node_shared_ptr());
    TENSORFLOW_OP_VALIDATION(
            node,
            complex_type_mark,
            "[TensorFlow Frontend] internal error: ComplexTypeMark is not set at the input of " + op_type);
    auto data = complex_type_mark->input_value(0);


    auto real_index = make_shared<v0::Constant>(element::i32, Shape{}, 0);
	auto imag_index = make_shared<v0::Constant>(element::i32, Shape{}, 1);

    auto gather_axis = make_shared<v0::Constant>(element::i32, Shape{1}, -1);
	
	auto real = make_shared<v8::Gather>(data, real_index, gather_axis)->output(0);
    auto imag = make_shared<v8::Gather>(data, imag_index, gather_axis)->output(0);

	
	auto const_minus_one = make_shared<v0::Constant>(element::i32, Shape{}, -1);
	imag = make_shared<v0::MatMul>(imag, const_minus_one, false, false);
	

	auto conj_tensor = make_shared<v0::Concat>(OutputVector{real, imag}, -1)->output(0);

	auto conj_transpose = make_shared<v1::Transpose>(conj_tensor, perm);

	set_node_name(node.get_name(), conj_transpose);
	return {conj_transpose};
    }
}
}
}
}
	