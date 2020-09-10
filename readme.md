# TWEET SENTIMENT EXTRACTION

Motive: Pick out the part of the tweet (word or phrase) that reflects the sentiment.

## Dataset Analysis:
> Conclusions drawn after performing EDA:<br/>
>1. Neutral tweets have a jaccard similarity of about 97 percent between text and selected_text. 
Comparatively, positive and negative tweets show much lower jaccard similarity.Thus, in case of neutral tweets even complete tweet can be used as selected text.<br/>
>2. URLs do not make much sense for positive and negative sentiments. They are more inclined towards the neutral side.<br/>
>3. Average length of words in the selected text is around 7. Also, selected text is always a continuous segment of words from the tweet.<br/>
>4. Also, for the best jaccard similarity, we need to extract the exact words from the tweet as selected text. Even a change of punctuation will lead to comparatively bad jaccard similarity.<br/>
>5. Symbols like continuous stars (*) are considered to be extreme emotions.
Negative and neutral tweets show high count for presence of stars. Presence of only stars and no words implies negative sentiment.<br/>

## Approaches that we find useful for solving this problem:
1. NER Approach
https://www.kaggle.com/tanulsingh077/twitter-sentiment-extaction-analysis-eda-and-model
https://www.kaggle.com/rohitsingh9990/ner-training-using-spacy-ensemble/comments
2. Q&A Approach
https://www.kaggle.com/jonathanbesomi/question-answering-starter-pack
https://www.kaggle.com/al0kharba/tensorflow-roberta-0-712
https://www.kaggle.com/cdeotte/tensorflow-roberta-0-705

While going through these notebooks, we decided to go through first some of the concepts in NLP:
## Representation of word vector (Word2vec): CBOW, Skip Gram with Negative sampling, Glove:
Problem with One Hot encoding:
* Large Dimensionality
* Closely coupled to the application, making transfer learning difficult
* No capturing of semantics

So to represent semantics(meaning), embeddings are used. 
CBOW is used to predict words given the context. 
Whereas, Skip Gram is used to predict context given the word with the help of subsampling (discard some of the words based upon their frequencies) and negative sampling (not updating all the weights in the final layer) to decrease training time.
GloVe creates a global co-occurrence matrix by estimating the probability a given word will co-occur with other words.
https://arxiv.org/pdf/1301.3781.pdf
https://arxiv.org/pdf/1310.4546.pdf
https://towardsdatascience.com/nlp-101-negative-sampling-and-glove-936c88f3bc68
http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/


With the knowledge of deep learning, we learned  about sequence models. 
Sequence-to-Sequence (or Seq2Seq) is a neural net that transforms a given sequence of elements, such as the sequence of words in a sentence, into another sequence. For this, we studied


## RNNs: 
Current output not only depends upon the current input but also on the previous inputs thereby preserving some relationships. Single neuron have the same weights and biases.
During Backpropagation, the problem of vanishing gradients still persists bcoz of multiplications in chain rule when trying to learn long term dependencies. If a sequence is long enough, they’ll have a hard time carrying information from earlier time steps to later ones.
For instance if we have a sentence like “The man who ate my pizza has purple hair”. In this case, the description of purple hair is for the man and not the pizza. So this is a long dependency.

https://www.analyticsvidhya.com/blog/2017/12/introduction-to-recurrent-neural-networks/

## LSTM: 
* Designed to tackle to short term memory
* Have gates to memorize which only relevant information: Forget gate (sigmoid function), Input gate (for adding new info involving tanh and sigmoid and their multiplication), output gates (getting useful info from current cell state).
* https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21
* https://www.analyticsvidhya.com/blog/2017/12/fundamentals-of-deep-learning-introduction-to-lstm/

## GRU
* Less computation
* Does same work as LSTM with fewer gates (Update and reset)


While exploring, we came to know that seq2seq models are particularly used for machine translation as its name suggests. Basic choice would be to use LSTM bcoz it contains gates to learn which info is important.

LSTMs and RNNs present 3 problems:
1. Sequential computation inhibits parallelization
2. No explicit modeling of long and short range dependencies
3. the probability of keeping the context from a word that is far away from the current word being processed decreases exponentially with the distance from it.

That's where Attention comes into play.


## Attention Mechanism
The attention-mechanism looks at an input sequence and decides at each step which other parts of the sequence are important while feeding into the encoder. 
for each input that the LSTM (Encoder) reads, the attention-mechanism takes into account several other inputs at the same time and decides which ones are important by attributing different weights to those inputs. The Decoder will then take as input the encoded sentence and the weights provided by the attention-mechanism.

With advancements, it was discovered that only attention mechanism are sufficient for machine translation without having to add any rnn.
https://towardsdatascience.com/transformers-141e32e69591
https://medium.com/inside-machine-learning/what-is-a-transformer-d07dd1fbec04

## Transfomers
It makes use of self attention (multihead attention) in the decoding layers and calculate the score.
Ex: The animal did not cross the street bcoz it was too wide.Positional Encoding
Another important step on the Transformer is to add positional encoding when encoding each word. Encoding the position of each word is relevant, since the position of each word is relevant to the translation.

https://towardsdatascience.com/transformers-141e32e69591
https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/
https://www.analyticsvidhya.com/blog/2019/06/understanding-transformers-nlp-state-of-the-art-models/


## BERT algorithm

* Bidirectional training of transformer which leads to have deeper sense of context
* Uses 2 training strategies: Masked LM and Next Sentence Prediction

1. MLM: 15% of the words in each sequence are replaced with a [MASK] token. The model then attempts to predict the original value of the masked words, based on the context provided by the other, non-masked, words in the sequence by adding a classification layer above the encoder. Multiplying output vector with embedding weights and gets probability of words using softmax.

2. NSP: the model receives pairs of sentences as input and learns to predict if the second sentence in the pair is the subsequent sentence in the original document. A [CLS] token is inserted at the beginning of the first sentence and a [SEP] token is inserted at the end of each sentence.

https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270

## BERT can be used in following tasks:
1. Sentiment Analysis
2. Question and answering task
3. NER

NER is used to label the entities with the help of the bert pretrained model and vocab.
In Q&A, the software receives a question regarding a text sequence and is required to mark the answer in the sequence. Using BERT, a Q&A model can be trained by learning two extra vectors that mark the beginning and the end of the answer.

How to use BERT: https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/

After learning BERT, we decided to

##Problem formulation
We formulate the task as question answering problem: given a question and a context, we train a transformer model to find the answer in the text column (the context).

We have:

Question: sentiment column (positive or negative)
Context: text column
Answer: selected_text column
