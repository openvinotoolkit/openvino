#include <inference_engine.hpp>
#include <opencv2/core/core.hpp>
#include "ie_plugin_config.hpp"
#include "hetero/hetero_plugin_config.hpp"


int main() {
using namespace InferenceEngine;
//! [part5]
cv::Mat frame = ...;  // regular CV_8UC3 image, interleaved
// creating blob that wraps the OpenCV’s Mat
// (the data it points should persists until the blob is released):
InferenceEngine::SizeVector dims_src = {
	1 	    /* batch, N*/,
	(size_t)frame.rows  /* Height */,
	(size_t)frame.cols    /* Width */,
	(size_t)frame.channels() /*Channels,*/,
	};
TensorDesc desc(InferenceEngine::Precision::U8, dims_src, InferenceEngine::NHWC);
InferenceEngine::TBlob<uint8_t>::Ptr p = InferenceEngine::make_shared_blob<uint8_t>( desc, (uint8_t*)frame.data, frame.step[0] * frame.rows);
inferRequest.SetBlob("input", p);
inferRequest.Infer();
// …
// similarly, you can wrap the output tensor (let’s assume it is FP32)
// notice that the output should be also explicitly stated as NHWC with setLayout
const float* output_data = output_blob->buffer().
		as<PrecisionTrait<Precision::FP32>::value_type*>();
cv::Mat res (rows, cols, CV_32FC3, output_data, CV_AUTOSTEP);
//! [part5]

return 0;
}
