#coding:utf-8
#-------------------------------
# Infers traductions based on two comparable corpora
# We perform the direct method using a dictionary of cognates
# and a bilingual dictionary
#-------------------------------

import re, sys, os, nltk
import types
import cognates

class Token:
  """Token with covered text, POS, lemma and the ID of the sentence it belongs to"""
  def __init__(self, coveredText="", pos="", lemma="", sentenceID=-1):
    self.coveredText = coveredText
    self.pos = pos
    self.lemma = lemma
    self.sentenceID = sentenceID

  def __str__(self) : 
    #return self.lemma
    return self.lemma
  def __repr__(self) : 
    return "Token(%s + %s + %s + %i)"%(self.coveredText, self.pos, self.lemma, self.sentenceID)

# Shortcuts
CORPUS_SOURCE="resources/termer_source/test" #corpus.lem"
CORPUS_TARGET="resources/termer_target/test" #corpus.lem"

COGNATES_DICO="cognates_dictionary.txt"

STOPWORDS_FR="resources/stopwords_fr.txt"
STOPWORDS_EN="resources/stopwords_en.txt"
#-------------------------------

# word == Token
# corpus == list of Token

# For French : 
# CoveredText/POS:Gender:Number/Lemma
# CoveredText/POS/Lemma
# CoveredText/Lemma
# POS = DTN | ..
# Gender = f | m | _
# Number = s | p
def iniCorpusFR(filename) : 
  corpus = []
  sentenceID = -1
  # List of stopwords and punctuation signes 
  with open(STOPWORDS_FR, "r") as f:
    content = f.read()
    stopwords = content.split("\n")
  with open(filename, "r") as f:
    for sentence in f: 
      if sentence.startswith("__") or sentence == " " : continue
      sentenceID = sentenceID + 1
      tokens = sentence.split(" ")
      for i in range(len(tokens)):
        tmp = tokens[i].split("/")
        # Filter stopwords and ponctuation
        if len(tmp) > 1 :
          if (tmp[0] not in stopwords) and (tmp[-1] not in stopwords) and (tmp[1] not in ["PREP"]) :
            #print ">" + tokens[i]
            tags = tmp[1].split(":")
            t = Token(tmp[0], tags[0], tmp[-1], sentenceID)
            #print tmp[0] + " " +tags[0]
            corpus.append(t)
  return corpus

# For English :
#CoveredText/POS/Lemma/?
#POS = CD | ...
#? = numero (correspond a quoi ??)
def iniCorpusEN(filename) : 
  corpus = []
  sentenceID = -1
  # List of stopwords and punctuation signes 
  with open(STOPWORDS_EN, "r") as f:
    content = f.read()
    stopwords = content.split("\n")
  with open(filename, "r") as f:
    for sentence in f: 
      if sentence.startswith("__") or sentence == " " : continue
      sentenceID = sentenceID + 1
      tokens = sentence.split(" ")
      for i in range(len(tokens)):
        tmp = tokens[i].split("/")
        # Filter stopwords and ponctuation
        if len(tmp) > 1 :
          if (tmp[0] not in stopwords) and (tmp[-1] not in stopwords) and (tmp[1] not in ["DT", "IN", "CC", "PRP", "TO"]) :
            #print ">" + tokens[i] 
            t = Token(tmp[0], tmp[1], tmp[-2], sentenceID)
            #print tmp[0] + " " +tmp[1]
            corpus.append(t)
  return corpus

if __name__ == "__main__":
  # Load comparable corpus : termer_{source,target}/corpus.lem
  sourceCorpus = iniCorpusFR(CORPUS_SOURCE)
  targetCorpus = iniCorpusEN(CORPUS_TARGET)

  # Compute cognates dictionary
  threshold = 5
  cognates.computeCognatesDictionary(threshold, sourceCorpus, targetCorpus, COGNATES_DICO)

  # Load bilingual dictionaries

###################################################
  # Perform direct method

## Computing context vectors for 
## nouns
## verbs (expect be, have)
## adjectives
## and adverbs
window = 7
#Chiao : tfidf
#Morin : LO(i,f)


## Transferring context vectors
# use bilingual dictionary 

## Finding candidate translations
#Cosine

###################################################

  # Print results' evaluation

