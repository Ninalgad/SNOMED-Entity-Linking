function highlightText(innerHTML) {
    const containerEle = document.getElementById('container');
    const textarea = document.getElementById('textarea');

    const mirroredEle = document.createElement('div');
    mirroredEle.textContent = textarea.value;
    mirroredEle.id = 'container__mirror';
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

    // Replace html
    mirroredEle.innerHTML = innerHTML;
}


function refresh() {
    document.getElementById("textarea").removeAttribute('readonly');

    if (document.contains(document.getElementById("container__mirror"))) {
        document.getElementById("container__mirror").remove();
    }
    if (document.contains(document.getElementById("entities"))) {
        document.getElementById("entities").remove();
        var table = document.getElementById("table");
        var entities = document.createElement('tbody');
        entities.id = 'entities'
        table.appendChild(entities);
    }
}

function fetchResult() {
    refresh();
    var entities = document.getElementById("entities");
    var textarea = document.getElementById("textarea");

    // block new input
    document.getElementById("textarea").readOnly = "true";

    // scroll back to top to avoid visual bug
    textarea.scrollTop = 0;

    const apiUrl = '/search/' + textarea.value;
    d3.json(apiUrl).then(function(data) {
        highlightText(data.result['highlight']);
        appendItems(data.result.tab, entities);
    });
}


function appendItems(data, container) {
    for (const item of data) {
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
        descEntry.appendChild(descElement);
        row.appendChild(descEntry);

        container.appendChild(row);
    }
}
