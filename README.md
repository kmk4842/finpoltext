# finpoltext
Financial text analysis for Polish (optionally other languages with adjustments to spacy model and stoplist). The basic idea is to generate textual features from company reports based on word-lists and LDA topical analysis.

## Data
Data come from the Polish Thematic Corpus of Management Reports. LDA training data for three topical areas: forecasts, operations, sales. Test set of full management reports (sd_iaaer) and letters to shareholders (la_iaaer) from top 60 companies at the Warsaw Stock Exchange.

You can find the entire corpus and more corpora at:
* Klimczak, Karol, 2022, "Tematyczny Korpus Sprawozdań z Działalności Polskich Spółek Giełdowych", https://doi.org/10.18150/YBZDYQ, RepOD.
* More corpora at https://repod.icm.edu.pl/dataverse/repod/?q=klimczak

## Program control
There are the following sections that can be turned on/off as needed with switches:
1. process raw text files using a spacy model with stoplist
2. make wordclouds for the corpus
3. calculate word counts and sentiment list word counts
4. train LDA models
4. make wordclouds for LDA topics
5. forecast probabilities of LDA topics in the same or different corpus
