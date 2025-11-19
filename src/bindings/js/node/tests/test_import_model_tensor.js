const { Core, Tensor } = require('../lib/openvino');

test('importModel should accept a Tensor without throwing errors', () => {
    const core = new Core();

    // Minimal tensor for API checking
    const data = new Uint8Array([1, 2, 3]);
    const tensor = new Tensor('u8', data, [data.length]);

    expect(() => {
        core.importModel(tensor);
    }).not.toThrow();
});
