#include <ie_core.hpp>
#include <ngraph/ngraph.hpp>
#include "onnx_import/onnx.hpp"
#include <iostream>
#include <fstream>

int main() {
//! [part2]
 const char * resnet50_path = "resnet50/model.onnx";
 std::ifstream resnet50_stream(resnet50_path);
 if (resnet50_stream.is_open())
 {
     try
     {
         const std::shared_ptr<ngraph::Function> ng_function = ngraph::onnx_import::import_onnx_model(resnet50_stream);

         // Check shape of the first output, for example
         std::cout << ng_function->get_output_shape(0) << std::endl;
         // The output is Shape{1, 1000}
     }
     catch (const ngraph::ngraph_error& error)
     {
         std::cout << "Error when importing ONNX model: " << error.what() << std::endl;
     }
 }
 resnet50_stream.close();
//! [part2]
return 0;
}
