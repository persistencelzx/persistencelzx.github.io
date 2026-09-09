const { readFile } = require('node:fs/promises');

const check = (condition, message) => {
  if (!condition) throw new Error(message);
};

(async () => {
  const html = await readFile('public/posts/ORION/index.html', 'utf8');

  check(/class="math (?:inline|display)"/.test(html), 'Math markup was not generated');
  check(html.includes('\\(x_s\\)'), 'LaTeX subscript was not preserved');
  check(!html.includes('x<em>s'), 'Markdown emphasis corrupted a formula');
  check(html.includes('\\begin{aligned}'), 'The methodology pipeline was not rendered as display math');
  check(html.includes('\\xrightarrow{\\text{Vision Encoder}}'), 'The methodology arrows were lost');
  check(html.includes('\\mathcal{N}(\\mu_s,\\sigma_s^2)'), 'The planner distribution was not rendered');
  check(html.includes('\\mathcal{L}_{\\mathrm{VAE}}'), 'The VAE loss was not rendered');
  check(!html.includes('$$'), 'Raw display-math delimiters leaked into HTML');
  check(!html.includes('<em>{vae}'), 'Markdown emphasis corrupted the VAE loss');
  check(/mathjax/i.test(html), 'MathJax was not loaded for the article');

  console.log('Math rendering regression check passed.');
})().catch(error => {
  console.error(error.message);
  process.exitCode = 1;
});
