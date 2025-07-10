const eslint = require('@eslint/js');
const globals = require('globals');
const tseslint = require('typescript-eslint');
const { defineConfig } = require('eslint/config');

const customRules = {
  'semi': ['error'],
  'no-var': ['error'],
  'max-len': ['error', { 'code': 120, 'ignoreUrls': true }],
  'eol-last': ['error'],
  'indent': ['error', 2],
  'camelcase': ['error'],
  'semi-spacing': ['error'],
  'arrow-spacing': ['error'],
  'comma-spacing': ['error'],
  'no-multi-spaces': ['error'],
  'quotes': ['error', 'single'],
  'no-trailing-spaces': ['error'],
  'space-before-blocks': ['error'],
  'newline-before-return': ['error'],
  'comma-dangle': ['error', 'always-multiline'],
  'space-before-function-paren': ['error', {
    named: 'never',
    anonymous: 'never',
    asyncArrow: 'always',
  }],
  'key-spacing': ['error', { beforeColon: false }],
  'no-multiple-empty-lines': ['error', { max: 1, maxBOF: 0, maxEOF: 0 }],
  'keyword-spacing': ['error', { overrides: { catch: { after: false } } }],
  'prefer-destructuring': ['error', { object: true, array: false }],
  'no-explicit-any': 0,
  '@typescript-eslint/no-require-imports': 0,
};

module.exports = defineConfig([
  eslint.configs.recommended,
  tseslint.configs.recommended,
  {
    ignores: ['types/', 'dist/'],
  },
  {
    files: ['**/*.{js,mjs,cjs,ts}'],
    languageOptions: {
      globals: globals.node,
      parser: tseslint.parser,
    },
    rules: customRules,
  },
  {
    files: ['**/addon.ts'],
    rules: {
      '@typescript-eslint/no-misused-new': 'off',
    },
  },
]);
