# podcast-trailer-generation
Task: automatically create podcast trailers from podcast audio / transcript data to facilitate quick browsing of podcasts and promote engagement

## Instructions:
1. Clone the repo onto your local machine
2. Run the docker container image containing the model endpoint using `docker run -d --name mycontainer -p 80:80 myimage`
3. Provide sample POST request to try out model endpoint

## Approach:

My first step in this project was to research what makes a great podcast trailer, at this would provide the goal that my solution should make progress towards achieving. At the following webpage my apple, I found the outline for a podcast trailer below:

1. Introduce your show.
Start off by introducing your show and hosts to let people know what and who they’re listening to. Then provide a brief explanation of what your show is about. If you need inspiration, think back to the tagline you wrote while developing how to present your podcast. 

2. Share the highlights.
Find the moments from your show that define what the listening experience is like. If you interview guests, include your favorite quotes. If you tell stories, lead with details that help establish your narrative.

3. Make them want more.
Near the end of your highlights, build some tension or curiosity by introducing the problem in a story, providing a sudden twist, or leaving them with an unanswered question by a guest.

4. Give them a call to action.
Last but not least, give your audience a clear next step to take. Make sure listeners know when they can find new episodes, and don’t forget to mention they should subscribe. 

I decided to tackle the first of these components. Given more time, I would have approached items number 2 and 3. In order to create an introduction for a podcast, a critical component for a podcast trailer, I decided to initially approach the problem as a summarization task that aims to summarize the overall topic / content that the podcast contains. This left me with two technical approaches, abstractive or extractive summarization. Although abstractive summarization has the benefit of generating a more concise, overall summary, in the context of a trailer, it notably prevents using the actual podcast audio itself since the summary would consist of words that the podcast hosts have never spoken. Alternatively, extractive summarization enables picking out the most salient elements from a piece of text. With this in mind, I chose to favor an extractive summarization approach. However, as I thought more about it, I realized that purely summarizing the entire transcript of a podcast didn't satisfy the requirements for a podcast trailer, let alone creating an introduction for a podcast. This is because summarizing the entire podcast transcript would result in potentially including content from the end of the episode, which for some podcasts may ruin the plot or surprise element, for example. From taking a look at the sample podcast transcripts, I noticed that most podcasts include a sentence or two, usually near the beginning, that states what will be talked about in the podcast. I decided to leverage this idea to fine-tune a BERT model to specifically extract where the content of the podcsat is introduced within the full transcript. 


## Methods

One approach to this problem of creating a highlight trailer for podcasts is to treat it as a text summarization
task. Formulating the problem this way benefits us by enabling us to pre-train / fine-tune on the large array of
text summarization datasets (such as new article headlines), as this is a common task. The broader category
of datasets the better as this will enable our model to handle a larger variety of podcasts and podcast topic
areas. 
-Also can use emotion detection since a trailer is important to not only give a summary but also highlight
the most interesting or shocking part of a podcast. 

A second approach would be to perform topic modeling, extract the main topic and then find that node in a 
open-source knowledge graph. Then, use the adjacent nodes to identify keywords or in some other way
augment the process to find the most relevant span of text. This will provide the drawback of higher
latency

Another approach could be to identify the introduction of the podcast and when that stops. Serve that as the first part of the trailer. And then, use emotion speech recongition to pinpoint highlights of the episode to serve after this as "example snippets of interest from podcast"

My approach:
    -Benefit is that instead of using abstractive summarization where I'd need a larger model with an encoder and decoder, such as BART or GPT-3, I can use a smaller BERT model and perform token classification to find the start and end of the introduction.
    -For point #2 in podcast trialer structure best practices, I could either use emotion detection or standard text summarization and use news article summarization dataset (since news headlines typically are catchy, this dataset may be a good fit)
    -Also enables creating audio version of trailer, as opposed to an abstractive summarizatino approach

## Resources used
* https://arxiv.org/pdf/2110.07096.pdf
    * Method to extract intros
* https://www.damianospina.com/publication/spina-2017-extracting/spina-2017-extracting.pdf
    * Information around most effective styles of summarizing audio data

## Next steps

Leverege freely available podcast datasets, such as the Spotify Podcast Dataset. Upon looking into getting access to this dataset, I discovered you have to sign a form and then wait up to 2 weeks to be given access. Thus, unfortunately I could not make use of it. Fields of interest for this datset are the podcast description field in addition to the full podcast transcription, which could provide good training pairs for fine-tuning a summarization / description generation model. 