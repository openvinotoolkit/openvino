#include <openvino/runtime/core.hpp>


int main() {
    {
        //! [static_shape]
        ov::Core core;
        auto model = core.read_model("model.xml");
        ov::Shape static_shape = {10, 20, 30, 40};

        model->reshape(static_shape);
        //! [static_shape]
    }

    return 0;
}
