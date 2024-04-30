const KEYWORD = 'you';


function highlightTextTest() {
    const containerEle = document.getElementById('container');
    const textarea = document.getElementById('textarea');
    
    clearHighlights();
    
    const mirroredEle = document.createElement('div');
    mirroredEle.textContent = textarea.value;
    mirroredEle.classList.add('container__mirror');
    mirroredEle.id = 'highlights'
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
}



function highlightText(innerHTML) {
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
    mirroredEle.innerHTML = innerHTML;
}


function clearHighlights() {
if (document.contains(document.getElementById("highlights"))) {
            document.getElementById("highlights").remove();
}
}


function fetchResult() {
    var entities = document.getElementById("entities");
    var query = document.getElementById("textarea").value;

    const apiUrl = '/search/' + query;
    d3.json(apiUrl).then(function(data) {
        console.log(data.result);
        highlightText(data.result['highlight']);

        appendItems(data.result.tab, entities);

    });
}


function appendItems(data, container) {
    for (const item of data) {
    console.log(item);
        const row = document.createElement('tr');
            const dotEntry = document.createElement('th');
            const dotElement = document.createElement('span');
            dotElement.classList.add("dot");
            dotElement.style.backgroundColor = item.color;
            dotEntry.appendChild(dotElement);
            row.appendChild(dotEntry);

            const descEntry = document.createElement('th');
            const descElement = document.createElement('p');
            const text = document.createTextNode(item.concept);
            descElement.appendChild(text);
            // dotElement.classList.add("dot");
            descEntry.appendChild(descElement);
            row.appendChild(descEntry);

        // row.appendChild(document.createElement("br"));
        container.appendChild(row);
    }
}