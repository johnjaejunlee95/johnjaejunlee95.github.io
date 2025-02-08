document.querySelectorAll('pre code').forEach(block => {
  const button = document.createElement('button');
  button.innerText = '복사';
  button.className = 'copy-button';
  block.parentNode.insertBefore(button, block);

  button.addEventListener('click', () => {
    navigator.clipboard.writeText(block.textContent);
    button.innerText = '복사됨!';
    setTimeout(() => button.innerText = '복사', 2000);
  });
});