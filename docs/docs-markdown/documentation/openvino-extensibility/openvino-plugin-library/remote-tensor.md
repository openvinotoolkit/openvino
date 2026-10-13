---
sidebar_position: 7
sidebar_label: "Remote Tensor"
description: "Use the ov::IRemoteTensor interface as a base class for device-specific remote tensors."
---

# Remote Tensor

ov::RemoteTensor class functionality:

- Provides an interface to work with device-specific memory.

:::note
If plugin provides a public API for own Remote Tensor, the API should be header only and does not depend on the plugin library.
:::

## Device Specific Remote Tensor Public API

The public interface to work with device specific remote tensors should have header only implementation and doesn't depend on the plugin library.

```cpp
class VectorTensor : public ov::RemoteTensor {
public:
    /**
     * @brief Checks that type defined runtime parameters are presented in remote object
     * @param tensor a tensor to check
     */
    static void type_check(const Tensor& tensor) {
        RemoteTensor::type_check(
            tensor,
            {{ov::device::full_name.name(), {"TEMPLATE"}}, {"vector_data_ptr", {}}, {"vector_data", {}}});
    }

    /**
     * @brief Returns the underlying vector
     * @return const reference to vector if T is compatible with element type
     */
    template <class T>
    const std::vector<T>& get_data() const {
        auto params = get_params();
        OPENVINO_ASSERT(params.count("vector_data"), "Cannot get data. Tensor is incorrect!");
        try {
            auto& vec = params.at("vector_data").as<const std::vector<T>>();
            return vec;
        } catch (const std::bad_cast&) {
            OPENVINO_THROW("Cannot get data. Vector type is incorrect!");
        }
    }

    /**
     * @brief Returns the underlying vector
     * @return reference to vector if T is compatible with element type
     */
    template <class T>
    std::vector<T>& get_data() {
        auto params = get_params();
        OPENVINO_ASSERT(params.count("vector_data"), "Cannot get data. Tensor is incorrect!");
        try {
            auto& vec = params.at("vector_data").as<std::vector<T>>();
            return vec;
        } catch (const std::bad_cast&) {
            OPENVINO_THROW("Cannot get data. Vector type is incorrect!");
        }
    }

    /**
     * @brief Returns the const pointer to the data
     *
     * @return const pointer to the tensor data
     */
    const void* get_data() const {
        auto params = get_params();
        OPENVINO_ASSERT(params.count("vector_data"), "Cannot get data. Tensor is incorrect!");
        try {
            auto* data = params.at("vector_data_ptr").as<const void*>();
            return data;
        } catch (const std::bad_cast&) {
            OPENVINO_THROW("Cannot get data. Tensor is incorrect!");
        }
    }

    /**
     * @brief Returns the pointer to the data
     *
     * @return pointer to the tensor data
     */
    void* get_data() {
        auto params = get_params();
        OPENVINO_ASSERT(params.count("vector_data"), "Cannot get data. Tensor is incorrect!");
        try {
            auto* data = params.at("vector_data_ptr").as<void*>();
            return data;
        } catch (const std::bad_cast&) {
            OPENVINO_THROW("Cannot get data. Tensor is incorrect!");
        }
    }
};
```

The implementation below has several methods:

### type\_check()

Static method is used to understand that some abstract remote tensor can be casted to this particular remote tensor type.

### get\_data()

The set of methods (specific for the example, other implementation can have another API) which are helpers to get an access to remote data.

## Device-Specific Internal tensor implementation

The plugin should have the internal implementation of remote tensor which can communicate with public API.
The example contains the implementation of remote tensor which wraps memory from stl vector.

OpenVINO Plugin API provides the interface ov::IRemoteTensor which should be used as a base class for remote tensors.

The example implementation have two remote tensor classes:

- Internal type dependent implementation which has as an template argument the vector type and create the type specific tensor.
- The type independent implementation which works with type dependent tensor inside.

Based on that, an implementation of a type independent remote tensor class can look as follows:

```cpp
class VectorImpl : public ov::IRemoteTensor {
private:
    std::shared_ptr<ov::IRemoteTensor> m_tensor;

public:
    VectorImpl(const std::shared_ptr<ov::IRemoteTensor>& tensor) : m_tensor(tensor) {}

    template <class T>
    operator std::vector<T>&() const {
        auto impl = std::dynamic_pointer_cast<VectorTensorImpl<T>>(m_tensor);
        OPENVINO_ASSERT(impl, "Cannot get vector. Type is incorrect!");
        return impl->get();
    }

    void* get_data() {
        auto params = get_properties();
        OPENVINO_ASSERT(params.count("vector_data"), "Cannot get data. Tensor is incorrect!");
        try {
            auto* data = params.at("vector_data_ptr").as<void*>();
            return data;
        } catch (const std::bad_cast&) {
            OPENVINO_THROW("Cannot get data. Tensor is incorrect!");
        }
    }

    void set_shape(ov::Shape shape) override {
        m_tensor->set_shape(std::move(shape));
    }

    const ov::element::Type& get_element_type() const override {
        return m_tensor->get_element_type();
    }

    const ov::Shape& get_shape() const override {
        return m_tensor->get_shape();
    }

    size_t get_size() const override {
        return m_tensor->get_size();
    }

    size_t get_byte_size() const override {
        return m_tensor->get_byte_size();
    }

    const ov::Strides& get_strides() const override {
        return m_tensor->get_strides();
    }

    const ov::AnyMap& get_properties() const override {
        return m_tensor->get_properties();
    }

    const std::string& get_device_name() const override {
        return m_tensor->get_device_name();
    }
};
```

The implementation provides a helper to get wrapped stl tensor and overrides all important methods of ov::IRemoteTensor class and recall the type dependent implementation.

The type dependent remote tensor has the next implementation:

```cpp
template <class T>
class VectorTensorImpl : public ov::IRemoteTensor {
    void update_strides() {
        if (m_element_type.bitwidth() < 8)
            return;
        auto& shape = get_shape();
        m_strides.clear();
        if (!shape.empty()) {
            m_strides.resize(shape.size());
            m_strides.back() = shape.back() == 0 ? 0 : m_element_type.size();
            std::copy(shape.rbegin(), shape.rend() - 1, m_strides.rbegin() + 1);
            std::partial_sum(m_strides.rbegin(), m_strides.rend(), m_strides.rbegin(), std::multiplies<size_t>());
        }
    }
    ov::element::Type m_element_type;
    ov::Shape m_shape;
    ov::Strides m_strides;
    std::vector<T> m_data;
    std::string m_dev_name;
    ov::AnyMap m_properties;

public:
    VectorTensorImpl(const ov::element::Type element_type, const ov::Shape& shape)
        : m_element_type{element_type},
          m_shape{shape},
          m_data(ov::shape_size(shape)),
          m_dev_name("TEMPLATE"),
          m_properties{{ov::device::full_name.name(), m_dev_name},
                       {"vector_data", m_data},
                       {"vector_data_ptr", static_cast<void*>(m_data.data())}} {
        update_strides();
    }

    const ov::element::Type& get_element_type() const override {
        return m_element_type;
    }

    const ov::Shape& get_shape() const override {
        return m_shape;
    }
    const ov::Strides& get_strides() const override {
        OPENVINO_ASSERT(m_element_type.bitwidth() >= 8,
                        "Could not get strides for types with bitwidths less then 8 bit. Tensor type: ",
                        m_element_type);
        return m_strides;
    }

    void set_shape(ov::Shape new_shape) override {
        auto old_byte_size = get_byte_size();
        OPENVINO_ASSERT(shape_size(new_shape) * get_element_type().size() <= old_byte_size,
                        "Could set new shape: ",
                        new_shape);
        m_shape = std::move(new_shape);
        update_strides();
    }

    const ov::AnyMap& get_properties() const override {
        return m_properties;
    }

    const std::string& get_device_name() const override {
        return m_dev_name;
    }
};
```

### Class Fields

The class has several fields:

- `m_element_type` - Tensor element type.
- `m_shape` - Tensor shape.
- `m_strides` - Tensor strides.
- `m_data` - Wrapped vector.
- `m_dev_name` - Device name.
- `m_properties` - Remote tensor specific properties which can be used to detect the type of the remote tensor.

### VectorTensorImpl()

The constructor of remote tensor implementation. Creates a vector with data, initialize device name and properties, updates shape, element type and strides.

### get\_element\_type()

The method returns tensor element type.

### get\_shape()

The method returns tensor shape.

### get\_strides()

The method returns tensor strides.

### set\_shape()

The method allows to set new shapes for the remote tensor.

### get\_properties()

The method returns tensor specific properties.

### get\_device\_name()

The method returns tensor specific device name.
