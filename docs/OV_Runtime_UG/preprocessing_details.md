# Preprocessing API - details {#openvino_docs_OV_Runtime_UG_Preprocessing_Details}

## Preprocessing capabilities

### Address particular input/output

If your model has only one input, then simple `PrePostProcessor::input()` will get a reference to preprocessing builder for this input (tensor, steps, model):
- C++
  @snippet snippets/ov_preprocessing.cpp ov:preprocess:input_1

- Python
    ```python
    ppp.input().preprocess().scale(50.)
    ppp.output().postprocess().convert_element_type(Type.u8)
     ```

In general, when model has multiple inputs/outputs, each one can be addressed by name
- C++
  @snippet snippets/ov_preprocessing.cpp ov:preprocess:input_name

- Python
    ```python
    ppp.input('image')
    ppp.output('result')
     ```

Or by it's index

- C++
  @snippet snippets/ov_preprocessing.cpp ov:preprocess:input_index

- Python
    ```python
    ppp.input(1)
    ppp.output(2)
     ```

### Supported preprocessing operations

#### Mean/Scale normalization

Typical data normalization includes 2 operations for each data item: subtract mean value and divide to standard deviation. This can be done with the following code:
- C++
  @snippet snippets/ov_preprocessing.cpp ov:preprocess:mean_scale

- Same in Python
    ```python
    ppp.input('input').preprocess().mean(128).scale(127)
     ```

In Computer Vision area normalization is usually done separately for R,G,B values. To do this, [layout with 'C' dimension](./layout_overview.md) shall be defined. Example:
- C++
  @snippet snippets/ov_preprocessing.cpp ov:preprocess:mean_scale_array

- Python
    ```python
    ppp.input('input').model().set_layout(Layout('NCHW')) # N=1, C=3, H=224, W=224
    # Mean/Scale has 3 values which matches with C=3
    ppp.input('input').preprocess()
            .mean([103.94, 116.78, 123.68]).scale([57.21, 57.45, 57.73])
     ```

#### Convert precision

In Computer Vision, image is represented by array of unsigned 8-but integer values (for each color), but model accepts floating point tensors

To integrate precision conversion into execution graph as a preprocessing step, just do:

- C++
  @snippet snippets/ov_preprocessing.cpp ov:preprocess:convert_element_type

- Or same in Python
    ```python
    ppp.input('input').tensor().set_element_type(Type.u8)
    ppp.input('input').preprocess().convert_element_type(Type.f32)
     ```

#### Convert layout (transpose)

Transposing of matrices/tensors is a typical operation in Deep Learning - you may have a BMP image 640x480 which is an array of `{480, 640, 3}` elements, but Deep Learning model can require input with shape `{1, 3, 480, 640}`

Using [layout](./layout_overview.md) of user's tensor and layout of original model conversion can be done implicitly

- C++
  @snippet snippets/ov_preprocessing.cpp ov:preprocess:convert_layout

- Or same in Python
    ```python
    ppp.input('input').tensor().set_layout('NHWC')  # Define layout for your tensor

    ppp.input('input').model().set_layout('NCHW')  # Define needed layout of model

    # That's all. Layout conversion will be done automatically
    print(ppp)
  ```

Or if you prefer manual transpose of axes without usage of [layout](./layout_overview.md) in your code, just do:

- C++
  @snippet snippets/ov_preprocessing.cpp ov:preprocess:convert_layout_2

- Python
    ```python
    ppp.input('input').tensor().set_shape([1, 480, 640, 3])
    # Model expects shape [1, 3, 480, 640]
    ppp.input('input').preprocess().convert_layout([0, 3, 1, 2]);
    # Transposes 0->0 3->1 1->2 2->3
  ```

It performs the same transpose, but we believe that approach using source and destination layout can be easier to read and understand

#### Resize image

Resizing of image is a typical preprocessing step for computer vision tasks. With preprocessing API this step can also be integrated into execution graph and performed on target device.

To resize the input image, it is needed to define 'H' and 'W' dimensions of [layout](./layout_overview.md)

- C++
  @snippet snippets/ov_preprocessing.cpp ov:preprocess:resize_1

- Python
    ```python
    ppp.input('input').tensor().set_shape([1, 3, 960, 1280])
    ppp.input('input').model().set_layout(Layout('??HW'))
    ppp.input('input').preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR, 480, 640)
    ```

Or in case if original model has known spatial dimensions (widht+height), target width/height can be omitted

- C++
  @snippet snippets/ov_preprocessing.cpp ov:preprocess:resize_2

- Python
    ```python
    ppp.input('input').tensor().set_shape([1, 3, 960, 1280])
    ppp.input('input').model().set_layout(Layout('??HW'))  # Model accepts {1, 3, 480, 640} shape
    ppp.input('input').preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)
    ```


#### Color conversion

Typical use case is to reverse color channels from RGB to BGR and wise versa
TBD

#### Custom operations

TBD