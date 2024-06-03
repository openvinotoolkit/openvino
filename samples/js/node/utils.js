const tf = require('@tensorflow/tfjs-node');

module.exports = {
  topK,
  booleanMaskAsync,
  clipByValue,
  split,
  stack,
  gather,
  connectionsNms,
  add,
  fill,
  sum,
  pool2d,
  heatmapNms,
  range,
  reshape,
  toTFTensor,
  fromTFTensor,
};

function toTFTensor(ovTensor) {
  console.log({ dataSize: ovTensor.data.length, shape: ovTensor.getShape() });

  return tf.tensor(ovTensor.data, ovTensor.getShape());
}

function fromTFTensor(ov, precision, tfTensor) {
  return new ov.Tensor(precision, tfTensor.shape, tfTensor.data);
}

/**
 * Extracts the top K elements and their indices from the given tensor along
 * the last dimension.
 * @param {tf.Tensor} tensor - The input tensor.
 * @param {number} k - The number of top elements to extract.
 * @returns {Array} - An array containing two tensors: the top K values
 * and their indices.
 */
async function topK(tensor, k) {
  const [values, indices] = tf.topk(tensor, k);

  return [values, indices];
}

/**
 * Creates a boolean mask from a tensor, based on a condition.
 * @param {tf.Tensor} tensor - The input tensor.
 * @param {tf.Tensor} mask - The boolean mask tensor.
 * @returns {tf.Tensor} - The masked tensor.
 */
async function booleanMaskAsync(tensor, mask) {
  return tf.booleanMaskAsync(tensor, mask);
}

/**
 * Performs element-wise clipping of tensor values.
 * @param {tf.Tensor} tensor - The input tensor.
 * @param {number} min - The minimum value.
 * @param {number} max - The maximum value.
 * @returns {tf.Tensor} - The clipped tensor.
 */
function clipByValue(tensor, min, max) {
  return tensor.clipByValue(min, max);
}

/**
 * Splits the input tensor into sub tensors.
 * @param {tf.Tensor} tensor - The input tensor.
 * @param {number} numOrSizeSplits - Either an integer indicating the number of
 * splits, or an array of sizes for each split.
 * @param {number} axis - The dimension along which to split.
 * @returns {Array} - An array of tensors.
 */
function split(tensor, numOrSizeSplits, axis = 0) {
  return tf.split(tensor, numOrSizeSplits, axis);
}

/**
 * Stacks a list of rank-R tensors into a rank-(R+1) tensor.
 * @param {Array} tensors - The list of tensors to stack.
 * @param {number} axis - The dimension to stack along.
 * @returns {tf.Tensor} - The stacked tensor.
 */
function stack(tensors, axis = 0) {
  return tf.stack(tensors, axis);
}

/**
 * Gathers slices from a tensor along an axis specified by indices.
 * @param {tf.Tensor} tensor - The input tensor.
 * @param {tf.Tensor} indices - The indices of the values to gather.
 * @param {number} axis - The axis along which to gather.
 * @returns {tf.Tensor} - The gathered tensor.
 */
function gather(tensor, indices, axis = 0) {
  return tf.gather(tensor, indices, axis);
}

/**
 * Performs non-maximum suppression to remove redundant keypoint connections.
 * @param {tf.Tensor} aIdx - Indices of the first set of keypoints.
 * @param {tf.Tensor} bIdx - Indices of the second set of keypoints.
 * @param {tf.Tensor} affinityScores - The scores of the connections.
 * @returns {Array} - An array containing the filtered indices and scores.
 */
async function connectionsNms(aIdx, bIdx, affinityScores) {
  const order = affinityScores.argsort().reverse();
  const sortedAffinityScores = tf.gather(affinityScores, order);
  const sortedAIdx = tf.gather(aIdx, order);
  const sortedBIdx = tf.gather(bIdx, order);

  const idx = [];
  const hasKptA = new Set();
  const hasKptB = new Set();

  for (let t = 0; t < sortedAIdx.shape[0]; t++) {
    const i = sortedAIdx.arraySync()[t];
    const j = sortedBIdx.arraySync()[t];
    if (!hasKptA.has(i) && !hasKptB.has(j)) {
      idx.push(t);
      hasKptA.add(i);
      hasKptB.add(j);
    }
  }

  const finalIdx = tf.tensor1d(idx, 'int32');

  return [
    tf.gather(sortedAIdx, finalIdx),
    tf.gather(sortedBIdx, finalIdx),
    tf.gather(sortedAffinityScores, finalIdx),
  ];
}

/**
 * Performs element-wise addition of two tensors.
 * @param {tf.Tensor} a - The first input tensor.
 * @param {tf.Tensor} b - The second input tensor.
 * @returns {tf.Tensor} - The result of element-wise addition.
 */
function add(a, b) {
  return tf.add(a, b);
}

/**
 * Fills a tensor with a scalar value.
 * @param {Array} shape - The shape of the output tensor.
 * @param {number} value - The scalar value to fill the tensor with.
 * @param {string} dtype - The data type of the output tensor.
 * @returns {tf.Tensor} - The filled tensor.
 */
function fill(shape, value, dtype = 'float32') {
  return tf.fill(shape, value, dtype);
}

/**
 * Sums the values of a tensor along the specified axes.
 * @param {tf.Tensor} tensor - The input tensor.
 * @param {Array} axes - The dimensions to reduce.
 * @returns {tf.Tensor} - The reduced tensor.
 */
function sum(tensor, axes) {
  return tf.sum(tensor, axes);
}

/**
 * 2D Pooling
 *
 * @param {tf.Tensor} A - Input 2D tensor.
 * @param {number} kernelSize - The size of the window over which we take pool.
 * @param {number} stride - The stride of the window.
 * @param {number} [padding=0] - Implicit zero paddings on both sides
 * of the input.
 * @param {string} [poolMode='max'] - Pooling mode: 'max' or 'avg'.
 * @returns {tf.Tensor} - The result of pooling.
 */
function pool2d(A, kernelSize, stride, padding = 0, poolMode = 'max') {
  // Add padding to the input tensor
  let paddedA = A.pad([[padding, padding], [padding, padding]]);

  // Reshape the input tensor to 4D for patch extraction
  const batchSize = 1;
  const channels = 1; // Assuming single-channel (grayscale) input
  const [inputHeight, inputWidth] = paddedA.shape;
  const paddedA4D =
    paddedA.reshape([batchSize, inputHeight, inputWidth, channels]);

  // Extract patches
  const patches = tf.image.extractPatches(paddedA4D, {
    sizes: [1, kernelSize, kernelSize, 1],
    strides: [1, stride, stride, 1],
    rates: [1, 1, 1, 1],
    padding: 'VALID',
  });

  // Pooling
  let pooled;
  if (poolMode === 'max') {
    pooled = patches.max(3);
  } else if (poolMode === 'avg') {
    pooled = patches.mean(3);
  }

  // Reshape pooled result to 2D
  const [outputHeight, outputWidth] = [
    Math.floor((inputHeight - kernelSize) / stride) + 1,
    Math.floor((inputWidth - kernelSize) / stride) + 1
  ];
  pooled = pooled.reshape([outputHeight, outputWidth]);

  return pooled;
}

/**
* Non-Maximum Suppression for heatmaps.
* @param {tf.Tensor} heatmaps - The original heatmaps.
* @param {tf.Tensor} pooledHeatmaps - The pooled heatmaps.
* @returns {tf.Tensor} - The heatmaps after applying NMS.
*/
function heatmapNms(heatmaps, pooledHeatmaps) {
  return heatmaps.mul(heatmaps.equal(pooledHeatmaps));
}

/**
 * Proxy function for tf.range
 *
 * @param {number} start - The start value of the range.
 * @param {number} stop - The stop value of the range (exclusive).
 * @param {number} [step=1] - The step size between values in the range.
 * @param {string} [dtype='float32'] - The data type of the output tensor.
 * @returns {tf.Tensor} A tensor with values in the specified range.
 */
function range(start, stop, step = 1, dtype = 'float32') {
  return tf.range(start, stop, step, dtype);
}

/**
* Proxy function for tf.reshape
*
* @param {tf.Tensor} tensor - The input tensor to be reshaped.
* @param {number[]} shape - The desired shape of the output tensor.
* @returns {tf.Tensor} The reshaped tensor.
*/
function reshape(tensor, shape) {
  return tf.reshape(tensor, shape);
}
