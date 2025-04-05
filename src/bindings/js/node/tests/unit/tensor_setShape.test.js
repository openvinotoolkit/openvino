import assert from 'node:assert';
import { Tensor } from '../../../lib/addon.js'; // Adjust this path if needed

describe('Tensor.setShape()', () => {
  it('should update the shape of the tensor correctly', () => {
    const tensor = new Tensor('f32', [1, 3, 1, 1]);
    const newShape = [1, 3, 224, 224];

    tensor.setShape(newShape);
    const updatedShape = tensor.getShape();

    assert.deepStrictEqual(updatedShape, newShape);
  });
});
