
#include "common.hpp"


ov::PartialShape get_default_shape(ov::PartialShape given_shape) {
    if (given_shape.is_static()) {
        return given_shape.get_shape();
    }
    if (given_shape.size() != 4) {
        // unlikely an image
        throw std::runtime_error("For dynamic shapes, only number of dimentions equal to 4 is supported.");
    }
    const ov::Shape default_shape = {1, 3, 480, 480};
    ov::Shape new_shape;
    for (int index = 0; index < given_shape.size(); index++) {
        auto dimention = given_shape[index];
        auto default_value = default_shape[index];
        if (dimention.is_static()) {
            new_shape.push_back(dimention.get_length());
        } else {
            auto interval = dimention.get_interval();
            if (interval.contains(default_value)) {
                new_shape.push_back(default_value);
            } else if (default_value > interval.get_max_val()) {
                new_shape.push_back(interval.get_max_val());
            } else {
                new_shape.push_back(interval.get_min_val());
            }
        }
    }
    return new_shape;
}
