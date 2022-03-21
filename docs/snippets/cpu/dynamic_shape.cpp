#include <openvino/runtime/core.hpp>


int main() {
    {
        //! [defined_upper_bound]
        ov::Core core;
        auto model = core.read_model("model.xml");

        model->reshape({{ov::Dimension(1, 10), ov::Dimension(1, 20), ov::Dimension(1, 30), ov::Dimension(1, 40)}});
        //! [defined_upper_bound]
    }

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
