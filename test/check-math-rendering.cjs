const { readFile } = require('node:fs/promises');

const check = (condition, message) => {
  if (!condition) throw new Error(message);
};

(async () => {
  const html = await readFile('public/posts/hermes/index.html', 'utf8');

  check(/class="math (?:inline|display)"/.test(html), 'Math markup was not generated');
  check(html.includes('\\mathcal{F}_t'), 'LaTeX subscript was not preserved');
  check(!html.includes('\\mathcal{F}<em>t'), 'Markdown emphasis corrupted a formula');
  check(/mathjax/i.test(html), 'MathJax was not loaded for the article');

  console.log('Math rendering regression check passed.');
})().catch(error => {
  console.error(error.message);
  process.exitCode = 1;
});
