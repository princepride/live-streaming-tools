window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: "\\b(?:no-mathjax|tex2jax_ignore)\\b",
    processHtmlClass: "\\b(?:arithmatex|tex2jax_process)\\b"
  }
};

document$.subscribe(() => {
  MathJax.typesetPromise();
});
