const ovBindings = require('bindings')('addon_openvino');
const ovCommon = require('openvinojs-common');

const { ITensor, IShape, ISession } = ovCommon;

// class ITensor {
//   getDim() {
//     throw new Error('Need to implement');
//   }

//   getData() {
//     throw new Error('Need to implement');
//   }
// }

class Tensor extends ITensor {
  #originalTensor;

  constructor(shape, data) {
    const preprocessedShape = someShapePreprocessing(shape);
    const preprocessedData = someShapePreprocessing(data);

    this.#originalTensor = new ovBindings.Tensor(preprocessedShape, preprocessedData);
  }

  getDim() {
    return this.#originalTensor.dim;
  }
} 

module.exports = { Tensor };
// module.exports = ovCommon.Factory({ Tensor, Shape, Session, Model });
