# TWEET SENTIMENT EXTRACTION

> Conclusions drawn after performing EDA:<br/>
>1. Neutral tweets have a jaccard similarity of about 97 percent between text and selected_text. 
Comparatively, positive and negative tweets show much lower jaccard similarity.Thus, in case of neutral tweets even complete tweet can be used as selected text.<br/>
>2. URLs do not make much sense for positive and negative sentiments. They are more inclined towards the neutral side.<br/>
>3. Average length of words in the selected text is around 7. Also, selected text is always a continuous segment of words from the tweet.<br/>
>4. Also, for the best jaccard similarity, we need to extract the exact words from the tweet as selected text. Even a change of punctuation will lead to comparatively bad jaccard similarity.<br/>
>5. Symbols like continuous stars (*) are considered to be extreme emotions.
Negative and neutral tweets show high count for presence of stars. Presence of only stars and no words implies negative sentiment.<br/>



