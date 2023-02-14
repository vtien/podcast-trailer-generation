# podcast-trailer-generation
Task: automatically create podcast trailers from podcast audio / transcript data to facilitate quick browsing of podcasts and promote engagement

## Instructions:
1. Pull the docker container for this project by running `docker pull vishaltien/podcast-intro-model:latest`
2. Run the container using `docker run -d --name app1 -p 80:80 vishaltien/podcast-intro-model:lateset`
3. Navigate to the docker url and then to the /docs endpoint to test the API

NOTE: Model performance is currently around ~80% accuracy, but final prediction does not always look great. Current hypothesis is that this is due to inference strategy, which was attempted to be replicated from paper. As a result, I also include a more direct inference strategy in the API (prediction_pieces) which extracts the tokens predicted to be in the introduction, as opposed to the most probable introduction span (prediction)

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

I decided to tackle the first of these components. Given more time, I would have approached items number 2 and 3. In order to create an introduction for a podcast, a critical component for a podcast trailer, I decided to initially approach the problem as a summarization task that aims to summarize the overall topic / content that the podcast contains. Formulating the problem this way is beneficial by enabling us to pre-train / fine-tune on the large array of text summarization datasets (such as new article headlines), as this is a common task. This left me with two technical approaches, abstractive or extractive summarization. Furthermore, I began to look into more domain-specific datasets, and found that there were podcast datasets being discussed in the literature. I intially thought of fine-tuning a generative language model, such as T5, on a publicly available podcast dataset that had a description of the podcast. One such dataset is the Spotify Podcast Dataset, but I discovered that it requires up to a 2 week waiting period to get access to the data, so I did not end up using it. I also found someone who had this exact same idea and uploaded their model at https://huggingface.co/paulowoicho/t5-podcast-summarisation. After more thought, I decided that although abstractive summarization has the benefit of generating a more concise, overall summary, in the context of a trailer, it notably prevents using the actual podcast audio itself since the summary would consist of words that the podcast hosts have never spoken. Alternatively, extractive summarization enables picking out the most salient elements from a piece of text. With this in mind, I chose to favor an extractive summarization approach. However, as I thought more about it, I realized that purely summarizing the entire transcript of a podcast didn't satisfy the requirements for a podcast trailer, let alone creating an introduction for a podcast. This is because summarizing the entire podcast transcript would result in potentially including content from the end of the episode, which for some podcasts may ruin the plot or surprise element, for example. From taking a look at the sample podcast transcripts, I noticed that most podcasts include a sentence or two, usually near the beginning, that states what will be talked about in the podcast. I decided to leverage this idea to fine-tune a BERT model to specifically extract where the content of the podcsat is introduced within the full transcript. 

**Summary of approach:**

Identify the introduction of the podcast and serve that as the first part of the trailer. Then, use emotion speech recongition to pinpoint highlights of the episode to serve after this as "example snippets of interest from podcast". In this repo, I only tackle the first component of this approach.

## Resources
| Task | Approach | Resource | Resource Link | Status |
| ---- | -------- | -------- | ------------- | ------ |
| Introduce your show | Extract introduction span from transcript | Jing et. al., 2021 | https://arxiv.org/pdf/2110.07096.pdf | Complete |
| Share the highlights | Speech emotion recognition | Wagner et. al., 2022 | https://arxiv.org/pdf/2203.07378v2.pdf | Planned (not completed) |

## Next steps

* Improve inference strategy

* Create more training pairs using augmentation / preprocessing techniques, and / or weak supervision on text summarization tasks in other domains (news article headlines)

* Parallelize prediction pipeline for faster inference since input text is split into chunks of 512 tokens. Currently, predictions are computed for each chunk one at a time. However, each prediction chunk is entirely dependent of the other, so this could be done in parallel and the most confident prediction returned at the end

* Leverege freely available podcast datasets, such as the Spotify Podcast Dataset. Upon looking into getting access to this dataset, I discovered you have to sign a form and then wait up to 2 weeks to be given access. Thus, unfortunately I could not make use of it. Fields of interest for this datset are the podcast description field in addition to the full podcast transcription, which could provide good training pairs for fine-tuning a summarization / description generation model. 