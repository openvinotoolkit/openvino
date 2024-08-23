module.exports = {
  extends: [
    'eslint:recommended',
    // TODO: Temporary disabled
    // Uncomment when will https://github.com/micromatch/micromatch/issues/264 be fixed
    // 'plugin:@typescript-eslint/recommended',
    './.eslintrc-global.js',
  ],
  ignorePatterns: ['**/*.js', 'node_modules/', 'types/', 'dist/', 'bin/'],
  root: true,
};
