#include "transformations/aszp_decomposition.hpp"
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/cc/ngraph/itt.hpp>
#include "backend/gna_limitations.hpp"
#include "aszp_decomposition.hpp"
#include <memory>

using namespace ngraph;
using namespace op;

namespace ov {
namespace intel_gna {
namespace pass {

static std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> ExtractPadding(ov::CoordinateDiff pads_begin, ov::CoordinateDiff pads_end) {
	auto Hbegin = pads_begin[0];
	auto Hend   = pads_end[0];
	auto Wbegin = pads_begin[1];
	auto Wend   = pads_end[1];

	return std::make_tuple(
		Hbegin, Hend, Wbegin, Wend, 
		Hbegin > Hend ? Hbegin - Hend : Hend - Hbegin, 
		Wbegin > Wend ? Wbegin - Wend : Wend - Wbegin);
}



std::shared_ptr<ngraph::opset8::Transpose> CreateTranspose(const Output<Node>& input) {
    return std::make_shared<ngraph::opset8::Transpose>(input,
		ngraph::op::Constant::create(element::Type_t::i64, Shape{2}, {1, 0}));
}

std::shared_ptr<ngraph::opset8::Reshape> CreateReshape(const Output<Node>& input, uint64_t ndims, ov::Shape shape) {
	return std::make_shared<ngraph::opset8::Reshape>(input, 
		ngraph::op::Constant::create(ngraph::element::i64, Shape{ndims}, shape)->output(0),false);
}

std::shared_ptr<ngraph::op::Constant> CreateZeroConst(ov::Shape shape) {
	return ngraph::op::Constant::create(ngraph::element::f32, shape, std::vector<float>(shape[0]*shape[1], 0.0f));
}



//returns nullptr if detects that convolution isn't surrounded by transpositions
std::shared_ptr<ngraph::opset8::Transpose> GetTransposeBefore(std::shared_ptr<ov::Node> conv) {
	const Output<Node>& parent = conv->input_value(0);
	
	auto transpose_before = std::dynamic_pointer_cast<ngraph::opset8::Transpose>(parent.get_node()->shared_from_this());
	if (nullptr == transpose_before) return nullptr;

	auto convolution_children = conv->output(0).get_target_inputs();
	auto convolution_bias = std::dynamic_pointer_cast<ngraph::opset8::Add>(convolution_children.begin()->get_node()->shared_from_this());
	
	std::shared_ptr<ngraph::opset8::Transpose> transpose_after;
	
	if (nullptr != convolution_bias) {
		auto add_children = convolution_bias->output(0).get_target_inputs();
        if (add_children.size() != 1) return nullptr;
        
		transpose_after = std::dynamic_pointer_cast<ngraph::opset8::Transpose>(add_children.begin()->get_node()->shared_from_this());
	} else transpose_after = std::dynamic_pointer_cast<ngraph::opset8::Transpose>(convolution_children.begin()->get_node()->shared_from_this());
	
	if (transpose_after == nullptr) return nullptr;

	return transpose_before;
}



static std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> GetInputDimensions(ov::Shape input_shape) {   
	uint64_t N = input_shape[0]; // batchsize
	uint64_t H = input_shape[1]; // no input channels (input depth)
	uint64_t W = input_shape[2]; // input height
	uint64_t C = input_shape[3]; // input widith

	return std::make_tuple(N,H,W,C);
}



std::shared_ptr<ov::op::v0::Concat> ConcatZeros(
	uint32_t pad_begin,
	uint32_t pad_end,
	std::shared_ptr<ov::Node> padding_const,
	std::shared_ptr<ov::Node> input_node) {

	OutputVector concat_vector;
	if (pad_begin > pad_end) {
		concat_vector.push_back(padding_const->output(0));
		concat_vector.push_back(input_node->output(0));
	}else {
		concat_vector.push_back(input_node->output(0));
		concat_vector.push_back(padding_const->output(0));	
	}
	return std::make_shared<ngraph::op::Concat>(concat_vector,1);

}




Output<Node> DecomposeHeight(Output<Node> input, ov::CoordinateDiff pads_begin, ov::CoordinateDiff pads_end, ov::Shape conv_input_shape) {
	uint32_t Hbegin, Hend, Wbegin, Wend, Hpad, Wpad;
	uint64_t N, C, H, W;
    std::tie(Hbegin, Hend, Wbegin, Wend, Hpad, Wpad) = ExtractPadding(pads_begin, pads_end);
    std::tie(N,H,W,C) = GetInputDimensions(conv_input_shape);

	if (Hpad != 0) {
		auto new_reshape   = CreateReshape(input, 2, Shape{H,W*C});
		auto new_transpose = CreateTranspose(new_reshape->output(0));

		auto padding_const = CreateZeroConst(Shape{W*C,Hpad});
		auto new_concat    = ConcatZeros(Hbegin,Hend,padding_const,new_transpose);
		
		auto new_untranspose = CreateTranspose(new_concat->output(0));	// Shape {W*C,H+Hpad}

		if (0 == Wpad) return CreateReshape(new_untranspose->output(0), 4, Shape{N, H+Hpad, W ,C})->output(0);
		else return (new_untranspose->output(0));
	}else return input;


}



Output<Node> DecomposeWidth(Output<Node> input, ov::CoordinateDiff pads_begin, ov::CoordinateDiff pads_end, ov::Shape conv_input_shape) {
	uint32_t Hbegin, Hend, Wbegin, Wend, Hpad, Wpad;
	uint64_t N, H, W, C;
    std::tie(Hbegin, Hend, Wbegin, Wend, Hpad, Wpad) = ExtractPadding(pads_begin, pads_end); //todo cleanup. maybe get only what we need?
    std::tie(N,H,W,C) = GetInputDimensions(conv_input_shape);

	if (Wpad != 0) {
		auto new_reshape   = CreateReshape(input, 2, Shape{(H+Hpad)*W, C});
		auto new_transpose = CreateTranspose(new_reshape->output(0));
		auto new_reshape2  = CreateReshape(new_transpose->output(0),2, Shape{C*(H+Hpad), W});

		auto padding_const = CreateZeroConst( Shape{C*(H+Hpad),Wpad} );
		auto new_concat    = ConcatZeros(Wbegin,Wend,padding_const,new_reshape2);

		auto new_unshape2    = CreateReshape(new_concat->output(0), 2, Shape{C, (H + Hpad) * (W + Wpad)});
		auto new_untranspose = CreateTranspose(new_unshape2->output(0));
		auto new_unshape     = CreateReshape(new_untranspose->output(0), 4, {N, H+Hpad, W+Wpad ,C});

		return new_unshape->output(0);
	}else return input;
}



void TrimmPadding(ov::CoordinateDiff& pads_begin, ov::CoordinateDiff& pads_end) {
	if ( pads_begin[0] > pads_end[0]) 
		 pads_begin[0] = pads_end[0];
	else pads_end[0] = pads_begin[0];
	if ( pads_begin[1] > pads_end[1]) 
		 pads_begin[1] = pads_end[1];
	else pads_end[1] = pads_begin[1];
}


std::shared_ptr<ov::Node> CreateConvolution (
	std::shared_ptr<ov::Node> conv, 
	const Output<Node>& input, 
	ov::CoordinateDiff pads_begin, 
	ov::CoordinateDiff pads_end){

	TrimmPadding(pads_begin, pads_end);

	auto cnn_node = std::dynamic_pointer_cast<ngraph::opset8::Convolution>(conv);
	if (nullptr != cnn_node){
		return std::make_shared<opset8::Convolution>(
			input, 
			cnn_node->input_value(1), 
			cnn_node->get_strides(), 
			pads_begin, pads_end, 
			cnn_node->get_dilations(), 
			cnn_node->get_auto_pad());
	}

	return nullptr;
}


// TODO ask about the input type. Convolution or Node. How to generalize for groupconv, dilated convolution
static bool Decompose(std::shared_ptr<ngraph::opset8::Convolution> conv) {
	auto pads_begin = conv->get_pads_begin();
    auto pads_end   = conv->get_pads_end();
	if (pads_begin[0] == pads_end[0] && pads_begin[1] == pads_end[1]) return false;

	auto transpose_before = GetTransposeBefore(conv);
    if (nullptr == transpose_before) return false;
	
	Output<Node> input = transpose_before->input_value(0);
	auto input_shape = input.get_shape();
	if (input_shape[0] != 1 || input_shape.size() != 4) return false;

	Output<Node> skip_input_H_const = DecomposeHeight(input, pads_begin, pads_end, input_shape);
	Output<Node> skip_input_W_const = DecomposeWidth(skip_input_H_const, pads_begin, pads_end, input_shape);

	auto final_transpose = std::make_shared<ngraph::opset8::Transpose>(skip_input_W_const, ngraph::op::Constant::create(element::Type_t::i64, Shape{4}, {0, 3, 1, 2}));

	auto new_conv = CreateConvolution(conv, final_transpose->output(0), pads_begin, pads_end);
	
	ngraph::replace_node(conv, new_conv);
	return true;
}



AszpDecomposition::AszpDecomposition() {
	MATCHER_SCOPE(AszpDecomposition);
	auto conv = ngraph::pattern::wrap_type<ngraph::opset8::Convolution>();

	ov::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto conv = std::dynamic_pointer_cast<ngraph::opset8::Convolution>(m.get_match_root());
		return Decompose(conv);
    };

	auto m = std::make_shared<ngraph::pattern::Matcher>(conv, matcher_name);
    this->register_matcher(m, callback);
}



}  // namespace pass
}  // namespace intel_gna
}  // namespace ov