module.exports = {
  type: "react-component",
  npm: {
    esModules: true,
    umd: {
      global: "AutobidsPortal",
      externals: {
        react: "React"
      }
    }
  }
};
