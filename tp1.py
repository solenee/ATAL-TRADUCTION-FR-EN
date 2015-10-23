#coding:utf-8

#-------------------------------
# Infers traductions based on two comparable corpora
# We perform the direct method using a dictionary of cognates
# and a bilingual dictionary
#-------------------------------

import re, sys, os, nltk
import types
import cognates
from numpy import sqrt


#-------------------------------------------------------------------------
# TYPES
#-------------------------------------------------------------------------
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


#-------------------------------------------------------------------------
# SHORTCUTS
#-------------------------------------------------------------------------
CORPUS_SOURCE="resources/termer_source/corpus.lem"
CORPUS_TARGET="resources/termer_target/corpus.lem"

BILINGUAL_DICO="resources/dicfrenelda-utf8.txt"
COGNATES_DICO="cognates_dictionary.txt"

STOPWORDS_FR="resources/stopwords_fr.txt"
STOPWORDS_EN="resources/stopwords_en.txt"

SOURCE_LANGUAGE="FR"
TARGET_LANGUAGE="EN"

JACCARD="JACCARD"
COSINE="COSINE"

#interestingPOS = {}
#interestingPOS["noun"]["fr"] = ["S", "NP"]
#interestingPOS["noun"]["en"] = ["S", "NP"]
#
#interestingPOS["adjective"]["fr"] = ["J"]
#interestingPOS["adjective"]["en"] = ["J"]
#
#interestingPOS["adverb"]["fr"] = ["D"]
#interestingPOS["adverb"]["en"] = ["D"]
#
#interestingPOS["verb"]["fr"] = ["V"]
#interestingPOS["verb"]["en"] = ["V"]

#-------------------------------------------------------------------------
# PARAMETERS
#-------------------------------------------------------------------------
SIMILARITY_FUNCTION=COSINE
#-------------------------------------------------------------------------
# METHODS
#-------------------------------------------------------------------------

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
          tmp[1] = tmp[1].split(":")[0]
          tmp[-1]= tmp[-1].split(":")[0]
          if (tmp[0] not in stopwords) and (tmp[-1] not in stopwords) and (tmp[1] in ["SBC", "ADJ", "ADJ2PAR", "ADJ1PAR", "ADV", "VCJ", "VNCNT", "VNCFF", "VPAR", "ECJ"]) :
#(tmp[1] not in ["DT", "IN", "CD", "PREP", "WDT"]) :
            #print ">" + tokens[i]
            t = Token(tmp[0], tmp[1], tmp[-1], sentenceID)
            #print tmp[0] + " " +tags[0]
            corpus.append(t)
          #else : print tmp[0]+" "+tmp[1]
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
          if (tmp[0] not in stopwords) and (tmp[-2] not in stopwords) and (tmp[1] in ["NN", "NNS", "NNP", "NNPS", "NN|JJ", "JJ", "JJR", "RB", "RBS", "VB", "VBZ", "VBN", "VBD", "VBG", "VBP",]) :
#(tmp[1] not in ["DT", "IN", "CC", "PRP", "TO"]) :
            #print ">" + tokens[i] 
            t = Token(tmp[0], tmp[1], tmp[-2], sentenceID)
            #print tmp[0] + " " +tmp[1]
            corpus.append(t)
          #else : print tmp[0]+" "+tmp[1]
  return corpus

def getBilingualDictionary(filename) :
  """ Load the bilingual dictionary """
  # TODO Load POS tags
  dico = {}
  with open(filename, "r") as f:
    for line in f: 
      if line == " " or line.startswith("-") : continue
      tmp = line.split(";")
      # Filter stopwords and ponctuation
      if len(tmp) > 5 :
        if ( tmp[2] == "TR-"+SOURCE_LANGUAGE+"-"+TARGET_LANGUAGE ) :
            #print ">" + tmp[0] 
            if (tmp[0] not in dico) : dico[tmp[0]] = []
            dico[tmp[0]].append(tmp[3])
            #print tmp[0] + " " +tmp[3]
  return dico



def inc(network, wordInFocus, wordInWindow) :
  """ Add 1 occurence of wordInWindow in wordInFocus' context vector """
  put(network, wordInFocus, wordInWindow, get(network, wordInFocus, wordInWindow) + 1 )

def put(network, wordInFocus, wordInWindow, nbOcc) : 
  """ Set the number of occurences of wordInWindow in wordInFocus' context vector to nbOcc """
  wordInFocusMap = {}
  if (wordInFocus in network) :
    wordInFocusMap = network[wordInFocus]
  else :
    network[wordInFocus] = wordInFocusMap
  # Anyway
  wordInFocusMap[wordInWindow] = nbOcc

def get(network, wordInFocus, wordInWindow) : 
  """ Return the number of occurences of wordInWindow in wordInFocus' context vector """
  if (wordInFocus not in network) :
    return 0
  wordInFocusMap = network[wordInFocus]
  if (wordInWindow not in wordInFocusMap) :
    return 0
  # otherwise (wordInFocus, wordInWindow) is in the collocation network
  return wordInFocusMap[wordInWindow]

def retrieveContextNetwork(cvsFile) : 
  """ Retrieve context vectors from a cvs file """
  # TODO
  network = {}
  return network

def saveContextNetwork(network, cvsFile) : 
  """ Save context vectors in a cvs file """
  content = []
  for word1 in network :
    for word2 in network[word1] :
      content.append(str(word1)+"\t"+str(word2)+"\t"+str(get(network, word1, word2))+"\n")
  # TODO remove the 2 unnecessary empty lines :   content.remove(sb.length()-1, sb.length());
  # Write cvs file
  with open(cvsFile, "w") as f:
    sb = ''.join(content)
    f.write(sb)
    #print sb

def computeContextVectors(corpus, windowSize) : #, POS_to_keep) :
  """ Compute the context vectors for a given pre-processed corpus """
  network = {}
  for i, word in enumerate(corpus):
    for j in range(i-windowSize, i+windowSize) :
      if (j>= 0) and (j<len(corpus)) and (j != i) :
        if corpus[j].sentenceID == word.sentenceID :
          inc( network, str(word), str(corpus[j]) )
  return network

def transfer(word, contextVector, bilingualDico) : 
  transferedVector = {}
  for word2 in contextVector :
      if word2 in bilingualDico : 
        candidates = bilingualDico[word2]
        best_candidate = candidates[0] # TODO Improve translation choice
        transferedVector[best_candidate] = contextVector[word2]
  #saveVector(word, contextVector, "CONTEXT")
  saveVector(word, transferedVector, "TRANSFERED")
  return transferedVector

def saveTransferedVector(word, vector) : 
  saveVector(word, vector, "TRANSFERED")

def saveVector(word, vector, vectorType) :
  cvsFile = vectorType+"/"+str(word)+"_"+vectorType
  content = []
  for word2 in vector :
      content.append(str(word2)+"\t"+str(vector[word2])+"\n")
  # TODO remove the 2 unnecessary empty lines :   content.remove(sb.length()-1, sb.length());
  # Write cvs file
  if content == [] : return
  with open(cvsFile, "w") as f:
    sb = ''.join(content)
    f.write(sb)
    #print sb

def getTransferedVector(word) :
  result = {} 
  vectorType = "TRANSFERED"
  cvsFile = vectorType+"/"+str(word)+"_"+vectorType
  with open(cvsFile, "r") as f:
    for line in f : 
      if line == " " : continue
      tmp = line.split("\t")
      # Filter stopwords and ponctuation
      if len(tmp) == 2 :
         print ">" + tmp[0] +" "+ tmp[1]
         result[tmp[0]] = tmp[1]
  return result

def getTestSet(filename) :
  testset = {}
  with open(filename, "r") as f:
    for line in f: 
      if line == " " : continue
      tmp = line.split(" : ")
      if len(tmp) == 2 :
        if tmp[0] not in testset : testset[tmp[0]] = []
        testset[tmp[0]].append(tmp[1])
        print tmp[0]+":"+tmp[1]
  return testset

def findCandidateTranslations(word, transferedVector, targetNetwork, nb, similarityFunction) :
  """ nb : number of candidates to find """
  print"====="
  TOP = nb
  scores = [] #list<Double> ; invariant : len(scores) <= TOP
  candidates = {} #Map< Double, list<String> >
  result = [] #Concatenation of Strings from candidates, ordered by their rank; len(translations) <= TOP
  current_min = 10000 #TODO initialize with max double
  for c in targetNetwork :
    score_c = similarity(transferedVector, targetNetwork[c], similarityFunction)
    if len(scores) < TOP :
      # add candidate
      #print "ADDING ("+c+", "+str(score_c)+")"
      if score_c not in candidates : 
        scores.append(score_c)
        candidates[score_c] = []
      # score_c is already in scores and in candidates' keyset
      candidates[score_c].append(c)
      # update current_min
      if current_min > score_c : current_min = score_c
    else :
      if score_c > current_min :
        # replace by the candidate c
        # pre : current_min is in candidates as key and in scores  
        scores.remove(current_min)
        del candidates[current_min]
        # add candidate
        #print "ADDING ("+c+", "+str(score_c)+")"
        if score_c not in candidates : 
          scores.append(score_c)
          candidates[score_c] = []
        #else score_c is already in scores and in candidates' keyset
        candidates[score_c].append(c)
        # update current_min
        current_min = min(scores)
  # rank the results
  scores.sort()

  print "len(scores)="+str(len(scores))+"\t TOP="+str(TOP)
  for s in scores :
    for w in candidates[s] : 
      print w+"> "+str(s)
      result.append(w)
#  i = 0
#  while (i<TOP) and (i<len(scores)) :
#    for w in candidates[scores[i]] : 
#      result.append(w)
#      print word+"> "+str(i)+" "+w
#    i=i+len(candidates[scores[i]])
  return result

def similarity(x, y, choice) :
  """ Cosine similarity : sigma_XiYi/ (sqrt_sigma_Xi2 * sqrt_sigma_Yi2) """
  result = 0
  Xi = [] # words
  sigma_XiYi = 0

  sigma_Xi2 = 0
  for w in x : 
    x_w = x[w]
    sigma_Xi2 = sigma_Xi2 + x_w *x_w

  sigma_Yi2 = 0
  for w in y : 
    y_w = y[w]
    sigma_Yi2 = sigma_Yi2 + y_w*y_w
    if w in x : sigma_XiYi = sigma_XiYi + x[w]*y_w
  
  if choice == COSINE : result = sigma_XiYi / ( sqrt(sigma_Xi2) * sqrt(sigma_Yi2) )
  if choice == JACCARD : result = sigma_XiYi / ( sigma_Xi2 + sigma_Yi2 - sigma_XiYi )
  return result
  
#-------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------
if __name__ == "__main__":
  # Load comparable corpus : termer_{source,target}/corpus.lem
  print ">LOADING COMPARABLE CORPORA..."
  sourceCorpus = iniCorpusFR(CORPUS_SOURCE)
  targetCorpus = iniCorpusEN(CORPUS_TARGET)

  # Compute cognates dictionary
  threshold = 5
  # TODO cognates.computeCognatesDictionary(threshold, sourceCorpus, targetCorpus, COGNATES_DICO)

  # Load bilingual dictionaries
  print ">LOADING BILINGUAL DICTIONARY..."
  bilingualDico = getBilingualDictionary(BILINGUAL_DICO)

  # Load entries to translate

###################################################
  # Perform direct method

## Computing context vectors for 
## nouns
## verbs (expect be, have)
## adjectives
## and adverbs
  print ">COMPUTING CONTEXT VECTORS..."
  windowSize = 2

  cvsFileSource = "source_network.cvs"
  POS_to_keep = []
  sourceNetwork = computeContextVectors(sourceCorpus, windowSize)
  saveContextNetwork(sourceNetwork, cvsFileSource) 

  cvsFileTarget = "target_network.cvs"
  targetNetwork = computeContextVectors(targetCorpus, windowSize)
  saveContextNetwork(targetNetwork, cvsFileTarget)

# TODO Normalize vectors
#Chiao : tfidf
#Morin : LO(i,f)

## Transferring context vectors
# use bilingual dictionary 
  print ">TRANSFERRING CONTEXT VECTORS..."
  transferedNetwork = {}
  for word in sourceNetwork : 
    transferedNetwork[word] = transfer(word, sourceNetwork[word], bilingualDico)
  
  print ">TESTING..."
  testset = getTestSet("testset.txt")
  results = {}
## Finding candidate translations
  top = 10
  tp = 0
  for word in testset :
    print ">>Candidates for '"+ word+"'"
    transferedVector = transferedNetwork[word] #getTransferedVector(word)
    print ">>>using COSINE : "
    candidates = findCandidateTranslations(word, transferedVector, targetNetwork, top, COSINE)
    found = 0
    for r in testset[word] :
      if r in candidates : 
        print r
        found = found+1
    if found > 0 :
      print r+" correct"
      tp = tp+1
    #print ">>>using JACCARD : "
    #candidates = findCandidateTranslations(word, transferedVector, targetNetwork, top, JACCARD)
#Cosine

###################################################

  # Print results' evaluation
  print "===========\ntp = "+str(tp)+"\nPrecision="+str(tp/len(testset))
  

