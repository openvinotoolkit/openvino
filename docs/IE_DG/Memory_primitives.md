Inference Engine Memory primitives {#openvino_docs_IE_DG_Memory_primitives}
=====================================================================

## Blobs

<code>InferenceEngine::Blob</code> is the main class intended for working with memory.
Using this class you can read and write memory, get information about the memory structure etc.

The right way to create <code>Blob</code> objects with a specific layout is to use constructors with <code>InferenceEngine::TensorDesc</code>.
<pre class="brush:cpp">
InferenceEngine::TensorDesc tdesc(FP32, {1, 3, 227, 227}, InferenceEngine::Layout::NCHW);
InferenceEngine::Blob::Ptr blob = InferenceEngine::make_shared_blob<float>(tdesc);
</pre>

## Layouts

<code>InferenceEngine::TensorDesc</code> is a special class that provides layout format description.

This class allows to create planar layouts using the standard formats (like <code>InferenceEngine::Layout::NCDHW</code>, <code>InferenceEngine::Layout::NCHW</code>, <code>InferenceEngine::Layout::NC</code>, <code>InferenceEngine::Layout::C</code> and etc) and also non-planar layouts using <code>InferenceEngine::BlockingDesc</code>.

In order to create a complex layout you should use <code>InferenceEngine::BlockingDesc</code> which allows to define the blocked memory with offsets and strides.

## Examples

1. You can define a blob with dimensions {N: 1, C: 25, H: 20, W: 20} and format NHWC with using next parameters:<br/>
<pre class="brush:cpp">
InferenceEngine::BlockingDesc({1, 20, 20, 25}, {0, 2, 3, 1}); // or
InferenceEngine::BlockingDesc({1, 20, 20, 25}, InferenceEngine::Layout::NHWC);
</pre>
2. If you have a memory with real dimensions {N: 1, C: 25, H: 20, W: 20} but with channels which are blocked by 8, you can define it using next parameters:<br/>
<pre class="brush:cpp">
InferenceEngine::BlockingDesc({1, 4, 20, 20, 8}, {0, 1, 2, 3, 1})
</pre>
3. Also you can set strides and offsets if layout contains it.
4. If you have a complex blob layout and you don't want to calculate the real offset to data you can use methods
<code>InferenceEngine::TensorDesc::offset(size_t l)</code> or <code>InferenceEngine::TensorDesc::offset(SizeVector v)</code>.<br/>
For example:
<pre class="brush:cpp">
InferenceEngine::BlockingDesc blk({1, 4, 20, 20, 8}, {0, 1, 2, 3, 1});
InferenceEngine::TensorDesc tdesc(FP32, {1, 25, 20, 20}, blk);
tdesc.offset(0); // = 0
tdesc.offset(1); // = 8
tdesc.offset({0, 0, 0, 2}); // = 16
tdesc.offset({0, 1, 0, 2}); // = 17
</pre>
5. If you would like to create a TensorDesc with a planar format and for N dimensions (N can be different 1, 2, 4 and etc), you can use the method
<code>InferenceEngine::TensorDesc::getLayoutByDims</code>.
<pre class="brush:cpp">
InferenceEngine::TensorDesc::getLayoutByDims({1}); // InferenceEngine::Layout::C
InferenceEngine::TensorDesc::getLayoutByDims({1, 2}); // InferenceEngine::Layout::NC
InferenceEngine::TensorDesc::getLayoutByDims({1, 2, 3, 4}); // InferenceEngine::Layout::NCHW
InferenceEngine::TensorDesc::getLayoutByDims({1, 2, 3}); // InferenceEngine::Layout::BLOCKED
InferenceEngine::TensorDesc::getLayoutByDims({1, 2, 3, 4, 5}); // InferenceEngine::Layout::NCDHW
InferenceEngine::TensorDesc::getLayoutByDims({1, 2, 3, 4, 5, ...}); // InferenceEngine::Layout::BLOCKED
</pre>