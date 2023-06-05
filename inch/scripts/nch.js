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
    var element = document.getElementById(msg.id);
    if (element) {
        //console.log("Setting " + element.id + " to opacity: " + msg.opacity);
        const opacity = Math.round(msg.opacity * 100) / 100;
        element.style.transition = "color 1s ease-out";
        element.style.color = "rgba(0, 0, 0, " + opacity + ")";
    }
}

chrome.runtime.onMessage.addListener(set_opacity);

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