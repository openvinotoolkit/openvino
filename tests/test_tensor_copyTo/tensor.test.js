const { Tensor } = require('../../src/bindings/js/node/lib/addon.ts');

describe('Tensor class', () => {
  it('should correctly copy data to another tensor', () => {
    const tensor1 = new Tensor('float32', [2, 2]);

    const tensor2 = new Tensor('float32', [2, 2]);

    tensor1.data = new Float32Array([1, 2, 3, 4]);

    tensor1.copyTo(tensor2);

    assert.deepEqual(tensor1.getData(), tensor2.getData(), 'Data was not copied correctly.');
  });
});
