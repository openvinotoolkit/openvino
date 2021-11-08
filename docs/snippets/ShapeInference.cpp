#include <ie_core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>


int main() {
int batch_size = 1;
//! [part0]
    InferenceEngine::Core core;
    // ------------- 0. Read IR and image ----------------------------------------------
    InferenceEngine::CNNNetwork network = core.ReadNetwork("path/to/IR/xml");
    cv::Mat image = cv::imread("path/to/image");
    // ---------------------------------------------------------------------------------

    // ------------- 1. Collect the map of input names and shapes from IR---------------
    auto input_shapes = network.getInputShapes();
    // ---------------------------------------------------------------------------------

    // ------------- 2. Set new input shapes -------------------------------------------
    std::string input_name;
    InferenceEngine::SizeVector input_shape;
    std::tie(input_name, input_shape) = *input_shapes.begin(); // let's consider first input only
    input_shape[0] = batch_size; // set batch size to the first input dimension
    input_shape[2] = image.rows; // changes input height to the image one
    input_shape[3] = image.cols; // changes input width to the image one
    input_shapes[input_name] = input_shape;
    // ---------------------------------------------------------------------------------

    // ------------- 3. Call reshape ---------------------------------------------------
    network.reshape(input_shapes);
    // ---------------------------------------------------------------------------------

    //...

    // ------------- 4. Loading model to the device ------------------------------------
    std::string device = "CPU";
    InferenceEngine::ExecutableNetwork executable_network = core.LoadNetwork(network, device);
    // ---------------------------------------------------------------------------------

//! [part0]

return 0;
}
