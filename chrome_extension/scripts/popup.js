function analyze_toxicity() {
	chrome.tabs.executeScript(null, { file: "scripts/jquery-2.2.4.min.js" }, function() {
	    chrome.tabs.executeScript(null, { file: "scripts/content.js" });
	});
}
document.getElementById('clickme').addEventListener('click', analyze_toxicity);