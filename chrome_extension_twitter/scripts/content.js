//TO DO: need to find way to grab only relevant text. Limit to say twitter. 


function unicodeToChar(text) {
	return text.replace(/\\u[\dA-F]{4}/gi, 
	      function (match) {
	           return String.fromCharCode(parseInt(match.replace(/\\u/g, ''), 16));
	      });
}

var process_new_tweets = function() {
	if(typeof(api_url) === 'undefined' ) {
		api_url = 'https://us-central1-toxicity-90.cloudfunctions.net/get_toxicity_prediction';
	};

	var tweetText = document.querySelectorAll("[data-testid='tweet']")
	var counter = tweetText.length;
	var commentArray = new Array(counter);
	for (let i = 0; i<tweetText.length; i++) {
		var comments = tweetText[i].getElementsByTagName('span');
		var commentText = '';
		for (let j = 0; j<comments.length; j++){
			commentText = commentText.concat(' ');
			commentText = commentText.concat(comments[j].innerText);}
			commentArray[i]=commentText;
	}
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
			  console.log(value);
			  if (score>.7){
			  tweetText[index].style.backgroundColor="red";}
		  });
	   })
	  .catch(error => console.log('Error:', error));
	}
	
	setTimeout(process_new_tweets, 2000); // scan timeline 1sec after load
	setInterval(process_new_tweets, 5000); // every 5 sec lets scan all




 