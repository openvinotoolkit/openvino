module.exports = {
  parserOptions: {
    ecmaVersion: 'latest'
  },
  env: {
    node: true,
    es6: true,
  },
  extends: [
    'eslint:recommended',
    '../../../src/bindings/js/.eslintrc-global.js',
  ],
  ignorePatterns: ['node_modules/', '.eslintrc.js'],
  root: true,
};
