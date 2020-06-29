//TO DO: need to find way to grab only relevant text. Limit to say reddit. 

alert("Analyzing comments for toxicity. This may take up to 30 seconds depending on length and number of comments.");

function unicodeToChar(text) {
	return text.replace(/\\u[\dA-F]{4}/gi, 
	      function (match) {
	           return String.fromCharCode(parseInt(match.replace(/\\u/g, ''), 16));
	      });
}

// capture all text
//var textToSend = document.body.innerText;
var realText = document.querySelectorAll("[data-test-id='comment']")

var counter = 0;
for (let i = 0; i<realText.length; i++) {
	var comments = realText[i].getElementsByTagName('p');
	for (let j = 0; j<comments.length; j++){
		counter++;
		//console.log(comments[j].innerHTML);
	}
}

var commentArray = new Array(counter);
counter=0;
for (let i = 0; i<realText.length; i++) {
	var comments = realText[i].getElementsByTagName('p');
	for (let j = 0; j<comments.length; j++){
		commentArray[counter] = comments[j].innerHTML;
		counter++;
	}
}

// Analyze and send back
console.log(typeof(api_url)==='undefined');
if(typeof(api_url) === 'undefined' ) {
    api_url = 'https://us-central1-toxicity-90.cloudfunctions.net/get_toxicity_prediction';
};

console.log(JSON.stringify(commentArray));
 fetch(api_url, {
  method: 'POST',
  body: JSON.stringify(commentArray),
  headers:{
    'Content-Type': 'application/json'
  } })
.then(data => { return data.json() })
.then(res => { 
	$.each(res, function( index, value ) {
		//value = unicodeToChar(value).replace(/\\n/g, '');
		score = value[1];
		comment_text = value[0];
		if (score>.9){
			console.log(value);
		document.body.innerHTML = document.body.innerHTML.split(comment_text).join('<span style="background-color: #AA0000;">' + comment_text + '</span>');}
	});
 })
.catch(error => console.error('Error:', error));
