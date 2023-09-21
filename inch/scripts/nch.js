/*
Noise-cancelling headphones for the internet

(C) 2023 Thinkst Applied Research
Author: Jacob Torrey

This extension classifies all the text in every <p> on a page using a fast LZMA-based LLM-detector model.
It sets the opacity of the <p> to the confidence that the <p> is human-written, so LLM-generated content
fades away.
*/

// Sends content to the web worker in the extension to process the text
async function send_request(id, text) {
    const msg = {id: id, content: text};
    chrome.runtime.sendMessage(msg);
}

// Given a JSON msg with the ID and opacity, sets the opacity as specified.
function set_opacity(msg, _sender, _sendResp) {
    const regex = /rgba?\((\d+), (\d+), (\d+)/;
    var element = document.getElementById(msg.id);
    element.classList.add('nch');
    if (element) {
        //console.log("Setting " + element.id + " to opacity: " + msg.opacity);
        const opacity = Math.round(msg.opacity * 100) / 100;
        element.style.transition = "color 1s ease-out";
        const ecolor = window.getComputedStyle(element).color;
        const match = ecolor.match(regex);
        if (ecolor == null || match == null) {
            element.style.setProperty("--rgb", "0, 0, 0");
            element.style.color = "rgba(var(--rgb), " + opacity + ")";
        } else {
            const r = match[1];
            const g = match[2];
            const b = match[3];
            element.style.setProperty("--rgb", r + ", " + g + ", " + b);
            element.style.color = "rgba(var(--rgb), " + opacity + ")";
        }
        if (element.title == '' && opacity <= 0.60)
            element.title = "Flagged as possibly AI-generated (confidence: " + (1-opacity) + ")";
    }
}

chrome.runtime.onMessage.addListener(set_opacity);

var hoverStyle = document.createElement('style');
hoverStyle.type = 'text/css';
hoverStyle.innerHTML = 'p.nch:hover {--alpha: 255;color:rgba(var(--rgb),var(--alpha))!important;}';
document.head.appendChild(hoverStyle);

// The types of elements we'll iterate through
const selector = 'p';
const tags = document.querySelectorAll(selector);
tags.forEach(element => {
    if (element.textContent && element.textContent.length >= 150) {
        // Make sure there's an id for that element to later set the opacity
        if (element.id == "")
            element.id = 'id' + selector + Math.floor(Math.random() * 1000).toString();
        send_request(element.id, element.textContent);
    }
});