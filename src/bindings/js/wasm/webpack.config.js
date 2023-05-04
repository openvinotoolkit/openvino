const path = require('path');
const CopyPlugin = require('copy-webpack-plugin');

module.exports = {
  // mode: 'production',
  mode: 'development',
  entry: {
    'openvino-wasm': './lib/index.ts',
  },
  module: {
    rules: [
      {
        test: /\.ts?$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
    ],
  },
  resolve: {
    extensions: ['.ts', '.js'],
    fallback: {
      'crypto': false,
      'fs': false,
      'path': false,
    }
  },
  plugins: [
    new CopyPlugin({
      patterns: [
        { from: './bin/openvino_wasm.wasm', to: 'openvino_wasm.wasm' },
      ],
    }),
  ],
  output: {
    filename: '[name].bundle.js',
    path: path.resolve(__dirname, 'dist'),
    clean: true,
    library: {
      name: 'openvino',
      type: 'umd',
      export: 'default',
    },
    libraryTarget: 'window',
  },
};
