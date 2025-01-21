#pragma once
#include <napi.h>
#include <ov/node.hpp>

#include <openvino/openvino.hpp>

class NodeWrap : public Napi::ObjectWrap<NodeWrap> {
public:
    /**
     * constructor
     * @brief Constructs NodeWrap from the Napi::CallbackInfo.
     * @param info contains passed arguments. Can be empty.
     * @param node pointer to ov::Node object.
     */
    NodeWrap(const Napi::CallbackInfo& info, ov::Node* node);

    // member functions
    Napi::Value get_name(const Napi::CallbackInfo& info);

private:
    ov::Node* node_;
};

#endif //NODE_WRAP_H