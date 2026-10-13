---
sidebar_position: 6
sidebar_label: "Remote Context"
description: "Use the ov::RemoteContext class as the base class for a plugin-specific remote context."
---

# Remote Context

ov::RemoteContext class functionality:

- Represents device-specific inference context.
- Allows to create remote device specific tensor.

:::note
If plugin provides a public API for own Remote Context, the API should be header only and does not depend on the plugin library.
:::

## RemoteContext Class

OpenVINO Plugin API provides the interface ov::IRemoteContext which should be used as a base class for a plugin specific remote context. Based on that, a declaration of an compiled model class can look as follows:

```cpp
class RemoteContext : public ov::IRemoteContext {
public:
    RemoteContext();
    const std::string& get_device_name() const override;
    const ov::AnyMap& get_property() const override;
    ov::SoPtr<IRemoteTensor> create_tensor(const ov::element::Type& type,
                                           const ov::Shape& shape,
                                           const ov::AnyMap& params = {}) override;

private:
    std::string m_name;
    ov::AnyMap m_property;
};
```

### Class Fields

The example class has several fields:

- `m_name` - Device name.
- `m_property` - Device-specific context properties. It can be used to cast RemoteContext to device specific type.

### RemoteContext Constructor

This constructor should initialize the remote context device name and properties.

```cpp
RemoteContext::RemoteContext() : m_name("TEMPLATE") {}
```

### get\_device\_name()

The function returns the device name from the remote context.

```cpp
const std::string& RemoteContext::get_device_name() const {
    return m_name;
}
```

### get\_property()

The implementation returns the remote context properties.

```cpp
const ov::AnyMap& RemoteContext::get_property() const {
    return m_property;
}
```

### create\_tensor()

The method creates device specific remote tensor.

```cpp
ov::SoPtr<ov::IRemoteTensor> RemoteContext::create_tensor(const ov::element::Type& type,
                                                          const ov::Shape& shape,
                                                          const ov::AnyMap& params) {
    std::shared_ptr<ov::IRemoteTensor> tensor;

    switch (type) {
    case ov::element::boolean:
        tensor =
            std::make_shared<VectorTensorImpl<ov::element_type_traits<ov::element::boolean>::value_type>>(type, shape);
        break;
    case ov::element::bf16:
        tensor =
            std::make_shared<VectorTensorImpl<ov::element_type_traits<ov::element::bf16>::value_type>>(type, shape);
        break;
    case ov::element::f16:
        tensor = std::make_shared<VectorTensorImpl<ov::element_type_traits<ov::element::f16>::value_type>>(type, shape);
        break;
    case ov::element::f32:
        tensor = std::make_shared<VectorTensorImpl<ov::element_type_traits<ov::element::f32>::value_type>>(type, shape);
        break;
    case ov::element::f64:
        tensor = std::make_shared<VectorTensorImpl<ov::element_type_traits<ov::element::f64>::value_type>>(type, shape);
        break;
    case ov::element::i8:
        tensor = std::make_shared<VectorTensorImpl<ov::element_type_traits<ov::element::i8>::value_type>>(type, shape);
        break;
    case ov::element::i16:
        tensor = std::make_shared<VectorTensorImpl<ov::element_type_traits<ov::element::i16>::value_type>>(type, shape);
        break;
    case ov::element::i32:
        tensor = std::make_shared<VectorTensorImpl<ov::element_type_traits<ov::element::i32>::value_type>>(type, shape);
        break;
    case ov::element::i64:
        tensor = std::make_shared<VectorTensorImpl<ov::element_type_traits<ov::element::i64>::value_type>>(type, shape);
        break;
    case ov::element::u8:
        tensor = std::make_shared<VectorTensorImpl<ov::element_type_traits<ov::element::u8>::value_type>>(type, shape);
        break;
    case ov::element::u16:
        tensor = std::make_shared<VectorTensorImpl<ov::element_type_traits<ov::element::u16>::value_type>>(type, shape);
        break;
    case ov::element::u32:
        tensor = std::make_shared<VectorTensorImpl<ov::element_type_traits<ov::element::u32>::value_type>>(type, shape);
        break;
    case ov::element::u64:
        tensor = std::make_shared<VectorTensorImpl<ov::element_type_traits<ov::element::u64>::value_type>>(type, shape);
        break;
    default:
        OPENVINO_THROW("Cannot create remote tensor for unsupported type: ", type);
    }
    return std::make_shared<VectorImpl>(tensor);
}
```

The next step to support device specific tensors is a creation of device specific [Remote Tensor](remote-tensor.md) class.
