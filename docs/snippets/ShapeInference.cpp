#include <openvino/runtime/core.hpp>
#include <openvino/core/layout.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

int main() {
ov::Core core;
auto model = core.read_model("path/to/model");

//! [picture_snippet]
    model->reshape({8, 3, 448, 448});
//! [picture_snippet]

size_t new_batch = 8;
//! [set_batch]
    // Mark up batch in the layout of the input(s) and reset batch to the new value
    model->get_parameters()[0]->set_layout("N...");
    ov::set_batch(model, new_batch);
//! [set_batch]

//! [spatial_reshape]
    // Read an image and adjust models single input for image to fit
    cv::Mat image = cv::imread("path/to/image");
    model->reshape({1, 3, image.rows, image.cols});
//! [spatial_reshape]

//! [obj_to_shape]
    std::map<ov::Output<ov::Node>, ov::PartialShape> port_to_shape;
    for (const ov::Output<ov::Node>& input : model->inputs()) {
        ov::PartialShape shape = input.get_partial_shape();
        // Modify shape to fit your needs
        // ...
        port_to_shape[input] = shape;
    }
    model->reshape(port_to_shape);
//! [obj_to_shape]

//! [idx_to_shape]
    size_t i = 0;
    std::map<size_t, ov::PartialShape> idx_to_shape;
    for (const ov::Output<ov::Node>& input : model->inputs()) {
        ov::PartialShape shape = input.get_partial_shape();
        // Modify shape to fit your needs
        // ...
        idx_to_shape[i++] = shape;
    }
    model->reshape(idx_to_shape);
//! [idx_to_shape]

//! [name_to_shape]
    std::map<std::string, ov::PartialShape> name_to_shape;
    for (const ov::Output<ov::Node>& input : model->inputs()) {
        ov::PartialShape shape = input.get_partial_shape();
        // input may have no name, in such case use map based on input index or port instead
        if (!input.get_names().empty()) {
        // Modify shape to fit your needs
        // ...
            name_to_shape[input.get_any_name()] = shape;
        }
    }
    model->reshape(name_to_shape);
//! [name_to_shape]

return 0;
}
