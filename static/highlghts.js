const KEYWORD = 'you';

document.addEventListener('DOMContentLoaded', () => {
    const containerEle = document.getElementById('container');
    const textarea = document.getElementById('textarea');

    const mirroredEle = document.createElement('div');
    mirroredEle.textContent = textarea.value;
    mirroredEle.classList.add('container__mirror');
    containerEle.prepend(mirroredEle);

    const textareaStyles = window.getComputedStyle(textarea);
    [
        'border',
        'boxSizing',
        'fontFamily',
        'fontSize',
        'fontWeight',
        'letterSpacing',
        'lineHeight',
        'padding',
        'textDecoration',
        'textIndent',
        'textTransform',
        'whiteSpace',
        'wordSpacing',
        'wordWrap',
    ].forEach((property) => {
        mirroredEle.style[property] = textareaStyles[property];
    });
    mirroredEle.style.borderColor = 'transparent';

    const parseValue = (v) => v.endsWith('px') ? parseInt(v.slice(0, -2), 10) : 0;
    const borderWidth = parseValue(textareaStyles.borderWidth);

    const ro = new ResizeObserver(() => {
        mirroredEle.style.width = `${textarea.clientWidth + 2 * borderWidth}px`;
        mirroredEle.style.height = `${textarea.clientHeight + 2 * borderWidth}px`;
    });
    ro.observe(textarea);

    textarea.addEventListener('scroll', () => {
        mirroredEle.scrollTop = textarea.scrollTop;
    });

    // Replace keyword
    const regexp = new RegExp(KEYWORD, 'gi');
    mirroredEle.innerHTML = textarea.value.replace(regexp, '<mark class="container__mark">$&</mark>');
});