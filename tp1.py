#coding:utf-8

#-------------------------------
# Infers traductions based on two comparable corpora
# We perform the direct method using a dictionary of cognates
# and a bilingual dictionary
#-------------------------------

import re, sys, os #, nltk
import types
import time
import cognates
import getTestSet as test
import codecs
from math import sqrt, log, log10
from collections import Counter
from re import match

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

DIRECT="DIRECT METHOD"
CHIAO="DISTRIBUTIONAL SYMMETRY"
GOOD_DICTIONARY="GOOD DICTIONARY"
CHIAO_GOOD_DICTIONARY="CHIAO + GOOD DICTIONARY"

TFIDF="TFIDF"
LO="LO"

MOST_FREQ="most"
SAME_WEIGHT="same"
ALL_WEIGHTED="freq"

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
MIN_WORD_LENGTH=3
SIMILARITY_FUNCTION=JACCARD #COSINE #
METHOD=CHIAO
STRATEGY_DICO="TO BE DEFINED"
TOLERANCE_RATE=1.5 #When there is several candidates with the same score, we accept
NORMALIZATION=TFIDF
STRATEGY_TRANSLATE=SAME_WEIGHT #MOST_FREQ
# to process max. TOP*TOLERANCE_RATE candidates

#-------------------------------------------------------------------------
# ATTRIBUTES
#-------------------------------------------------------------------------
#N : noun, V : verb, ADJ : adjective, ADV : adverb, NADJ : N|ADJ, VADJ
TARGET_POS = {}
SOURCE_POS = {}
DICO = {}
DICO_INV = {}
TARGET_SPACE = set()
SOURCE_SPACE = set()
PIVOT_WORDS = set()
TARGET_TRANSFERRED_VECTORS = {}
SOURCE_TRANSFERRED_VECTORS = {}
TARGET_NETWORK = {}
SOURCE_NETWORK = {}
TARGET_TRANSFERRED_VECTORS_FILE = "target_transferred_vectors.csv"
SOURCE_TRANSFERRED_VECTORS_FILE = "source_transferred_vectors.csv"
TARGET_NETWORK_FILE = "target_network.csv"
SOURCE_NETWORK_FILE = "source_network.csv"

#-------------------------------------------------------------------------
# METHODS
#-------------------------------------------------------------------------

# word == Token
# corpus == list of Token

def serialize(filename, variable) : 
  with open(filename, "w") as f:
    for key in variable :
      for word in variable[key] :
        f.write(key+" ; "+word+" ; "+str(variable[key][word])+"\n")
        
def unserialize(filename, variable) :
  with open(filename, "r") as f:
    for line in f :
      content = line.split(" ; ")
      assert (len(content) == 3), "Invalid format for file "+str(filename)+str(len(content))
      put(variable, str(content[0]), str(content[1]), float(content[2]))
      #print str(content[0])


def isWord(w) :
  """ Check if a String matches word pattern : [a-zA-Z]([a-zA-Z0-9'-])* """
  res = re.match("[a-zA-Zàâéèêëîïôöûü]([a-zA-Z0-9àâéèêëîïôöûü\'_-])*", w)
  return res and (len(res.group(0)) == len(w))

def clean(word) : 
  return word.strip(",?;.:/!'")
  
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
      if sentence.startswith("__") or sentence == " " or sentence == "": continue
      sentenceID = sentenceID + 1
      tokens = sentence.split(" ")
      for i in range(len(tokens)):
        tmp = tokens[i].split("/")
        # Filter stopwords and ponctuation
        if len(tmp) > 1 :
          tmp[1] = tmp[1].split(":")[0]
          tmp[-1]= clean(tmp[-1].split(":")[0]) #lemma
          if (len(tmp[-1]) >= MIN_WORD_LENGTH) and (isWord(tmp[0])) and (isWord(tmp[1])) and (tmp[0] not in stopwords) and (tmp[-1] not in stopwords) and (tmp[1] in ["SBC", "ADJ", "ADJ2PAR", "ADJ1PAR", "ADV", "VCJ", "VNCNT", "VNCFF", "VPAR"]) :
#(tmp[1] not in ["DT", "IN", "CD", "PREP", "WDT"]) :
            #print ">" + tokens[i]
            t = Token(tmp[0], tmp[1], tmp[-1], sentenceID)
            #print tmp[0] + " " +tags[0]
            corpus.append(t)
            SOURCE_SPACE.add(str(t))
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
    stopwords = content. split("\n")
  with open(filename, "r") as f:
    for sentence in f: 
      if sentence.startswith("__") or sentence == " " or sentence == "" : continue
      sentenceID = sentenceID + 1
      tokens = sentence.split(" ")
      for i in range(len(tokens)):
        tmp = tokens[i].split("/")
        # Filter stopwords and ponctuation
        if len(tmp) > 1 :
          tmp[1] = clean(tmp[1])
          if (len(tmp[-2]) >= MIN_WORD_LENGTH) and (isWord(tmp[0])) and (isWord(tmp[-2])) and (tmp[0] not in stopwords) and (tmp[-2] not in stopwords) and (tmp[1] in ["NN", "NNS", "NNP", "NNPS", "NN|JJ", "JJ", "JJR", "VB", "VBZ", "VBN", "VBD", "VBG", "VBP",]) :
#(tmp[1] not in ["DT", "IN", "CC", "PRP", "TO"]) :
            #print ">" + tokens[i]
            t = Token(tmp[0], tmp[1], tmp[-2], sentenceID)
            #print tmp[0] + " " +tmp[1]
            corpus.append(t)
            TARGET_SPACE.add(str(t))
          #else : print tmp[0]+" "+tmp[1]
  return corpus

def getBilingualDictionary(filename) : #, dicoS_T, dicoT_S) :
  """ Load the bilingual dictionary """
  # TODO Load POS tags
  dicoS_T = DICO
  dicoT_S = DICO_INV
  with open(filename, "r") as f:
    for line in f: 
      if line == " " or line.startswith("-") : continue
      tmp = line.split(";")
      # Filter stopwords and ponctuation
      if len(tmp) > 5 :
        if ( tmp[2] == "TR-"+SOURCE_LANGUAGE+"-"+TARGET_LANGUAGE ) :
            #print ">" + tmp[0] 
            if (tmp[0] not in dicoS_T) : dicoS_T[tmp[0]] = []
            dicoS_T[tmp[0]].append(tmp[3])
            if (tmp[3] not in dicoT_S) : dicoT_S[tmp[3]] = []
            dicoT_S[tmp[3]].append(tmp[0])
            #print tmp[0] + " " +tmp[3]
    #print dicoS_T
    #print dicoT_S


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

def sum_cooc(context_i):
  def add(x,y): return x+y
  return reduce(add, context_i.values(), 0)

def normalizeLO1(vectors) :
  N = float( sum([sum_cooc(vectors[i]) for i in vectors]) ) / 2 #N=a+b+c+d
  n_vectors = {}
  for i in vectors :
    n_vectors[i] = {}
    for j in vectors[i] :
      # cooc(i,j)
      a = float(vectors[i][j])
      # cooc(i, -j)
      b = float( sum([vectors[i][k] for k in vectors[i] if not (k == j)]) )
      # cooc(-i, j)
      c = float( sum([vectors[j][k] for k in vectors[j] if not (k == i)]) )
      # cooc(-i, -j)
      d = N - (a+b+c)
      n_vectors[i][j] = log ( ( (a+0.5)*(d+0.5) ) / ( (b+0.5)*(c+0.5) ) )
  vectors = n_vectors
      
def normalizeLO(vectors) :
  n_vectors = {}
  l_keys = vectors.keys()
  def indice(key) : return l_keys.index(key)
  l_sum_cooc = [sum_cooc(vectors[i]) for i in l_keys]
  N = float( sum(l_sum_cooc) ) / 2 #N=a+b+c+d
  for i in vectors :
    n_vectors[i] = {}
    for j in vectors[i] :
      # cooc(i,j)
      a = float(vectors[i][j])
      # cooc(i, -j)
      b = float( l_sum_cooc[indice(i)] - vectors[i][j] )
      # cooc(-i, j)
      c = float( l_sum_cooc[indice(j)] - vectors[j].get(i, 0) ) #BIZZARE : Pourquoi vectors[j][i] n'existe pas tjrs??
      #print str(c)
      # cooc(-i, -j)
      d = N - (a+b+c)
      n_vectors[i][j] = log ( ( (a+0.5)*(d+0.5) ) / ( (b+0.5)*(c+0.5) ) )
  vectors = n_vectors

  
def normalizeTFIDF(vectors):
  """ Normalize context vectors using the tf*idf measure described in Chiao """
  max_cooc = [reduce(max, vectors[i].values(), 0) for i in vectors]
  MAX_OCC = float(max(max_cooc))
  cooc_i = [sum_cooc(vectors[i]) for i in vectors]
  for i_index, i in enumerate(vectors) :
    idf = 1 + log(MAX_OCC/cooc_i[i_index])
    for j in vectors[i] :
      vectors[i][j] = ( float(vectors[i][j])/MAX_OCC ) * idf

def isPivot(word, dictionary, targetFreqCounts) :
  def isInTragetSpace(w) : return targetFreqCounts[w] > 0
  return (word in dictionary) and reduce(lambda a,b : a or b, filter(isInTragetSpace, dictionary[word]), False)

def transfer(word, contextVector, bilingualDico, targetFreqCounts) : 
  transferedVector = {}
  for word2 in contextVector :
    if word2 in bilingualDico : 
      translations = [(x, targetFreqCounts[x]) for x in bilingualDico[word2] if targetFreqCounts[x] > 0]
      if len(translations) > 0 :
        PIVOT_WORDS.add(word2)
        assoc = dict(translations)
        totalCounts = sum(assoc.values())
        m = max([targetFreqCounts[x] for x in bilingualDico[word2] if targetFreqCounts[x] > 0])
        for t in assoc :
          if (STRATEGY_TRANSLATE == MOST_FREQ) : #Most_freq_only
            if assoc[t] == m : transferedVector[t] = contextVector[word2]
          elif (STRATEGY_TRANSLATE == ALL_WEIGHTED) : #all_trad_weighted_by_frequency : 
            transferedVector[t] = contextVector[word2] * ( float(assoc[t]) / totalCounts )
          elif (STRATEGY_TRANSLATE == SAME_WEIGHT) : #all_trad_same_weight :               
            transferedVector[t] = contextVector[word2]
          else : print "Unknown translation strategy"
  #saveVector(word, contextVector, "CONTEXT")
  #saveVector(word, transferedVector, "TRANSFERED")
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
      #line = line.decode('utf-8')
      if line == " " : continue
      tmp = line.split(" : ")
      if len(tmp) == 2 :
        if tmp[0] not in testset : testset[str(tmp[0])] = []
        testset[str(tmp[0])].append(tmp[1].split("\n")[0])
        #print tmp[0]+":"+tmp[1]
  return testset

def getRank(candidates, word, isScore=True) :
  rank = 0
  s_index = 0
  stop = False
  scores = sorted(candidates.keys(), reverse=isScore)
  while not stop :
    if word in candidates[scores[s_index]] :
      rank = rank + 1
      stop = True
    else : rank = rank + len(candidates[scores[s_index]])
    s_index = s_index + 1
    if s_index == len(scores) :
      rank = float('inf')
      stop = True
  return rank

def arithmeticMean(x, y) :
  if x == float('inf') : return 2*y
  if y == float('inf') : return 2*x
  return float(x+y)/2

def geometricMean(x, y) :
  if x == float('inf') : return 2*y
  if y == float('inf') : return 2*x
  return sqrt(x*y)

def harmonicMean(x, y) :
  if x == float('inf') : return 2*y
  if y == float('inf') : return 2*x
  if (x+y) < 0.00000001 : return 0
  return float(2*x*y) / (x+y)

def findCandidateTranslationsChiaoReverse(word, sourceVector, transferredTargetNetwork, nb, similarityFunction, f_transferSource, f_filter_candidates=None) :
  #print "==========="
  print word
  #list of the nb best scores found
  candidates = {}
  scores = []
  if f_filter_candidates is None :
    print "f_filter_candidates is None"
    candidates = findCandidateScores(word, sourceVector, transferredTargetNetwork, nb, similarityFunction)
    scores = sorted(candidates.keys(), reverse=True)
  else :
    candidatesKeys = DICO.get(word, [])
    if len(candidatesKeys) == 0 :
      candidatesKeys = [k for k in transferredTargetNetwork.keys() if (len(DICO_INV.get(k, []))== 0)]
    filteredTargetNetwork = {k: v for k, v in transferredTargetNetwork.iteritems() if k in candidatesKeys}
    candidates = findCandidateScores(word, sourceVector, filteredTargetNetwork, nb, similarityFunction)
    scores = sorted(candidates.keys(), reverse=True)  

  res1 = [] #Concatenation of Strings from candidates, ordered by their rank; len(translations) <= TOP
  for i in range(len(scores)) :
    if len(res1) >= nb*TOLERANCE_RATE : break
    for w in candidates[scores[i]] :
      #print w+"> "+str(s)
      res1.append(w)
      if len(res1) >= nb*TOLERANCE_RATE :
        print "----early exit"
        break
  #print res1 #
  
  d_rank = {}

  transferredSourceVector = f_transferSource(word)
  for cand in res1 :
    if not (f_filter_candidates is None) :
      #print "f_filter_candidates is not None"
      if cand not in transferredTargetNetwork :
        print str(cand)+" rejected because not in target corpus"
        continue
      #else : print str(cand)
    #else : print "f_filter_candidates is not None"
    cand_vector = TARGET_NETWORK[cand]
    cand_reverse = findCandidateScores(cand, cand_vector, SOURCE_TRANSFERRED_VECTORS, nb, similarityFunction)
    cand_rank = harmonicMean(getRank(candidates, cand), getRank(cand_reverse, word))
    if cand_rank not in d_rank : d_rank[cand_rank] = []
    d_rank[cand_rank].append(cand)

  result = []
  ranks = sorted(d_rank.keys(), reverse=False)
  # Give an ordered list of the translation candidates
  for i in range(len(ranks)) :
    for w in d_rank[ranks[i]] :
      #print w+"> "+str(s)
      result.append(w)
  #print "---"
  #print result
  return result

def findCandidateTranslationsMixMean(word, nb, similarityFunction, meanFunction, f_filter_candidates=None) :
  #list of the nb best scores found
  candidates = {}
  scores = []
  if f_filter_candidates is None :
    candidates = findCandidateScoresMix(word, nb, similarityFunction, meanFunction)
    scores = sorted(candidates.keys(), reverse=True)
  else :
    candidatesKeys = DICO.get(word, [])
    if len(candidatesKeys) == 0 :
      candidatesKeys = [k for k in targetNetwork.keys() if (len(DICO_INV.get(k, []))== 0)]
    filteredTargetNetwork = {k: v for k, v in targetNetwork.iteritems() if k in candidatesKeys}
    candidates = findCandidateScoresMix(word, nb, similarityFunction)
    scores = sorted(candidates.keys(), reverse=True)
  #print "========="
  #print scores

  result = [] #Concatenation of Strings from candidates, ordered by their rank; len(translations) <= TOP
# Give an ordered list of the translation candidates
  for i in range(len(scores)) :
    for w in candidates[scores[i]] :
      #print w+"> "+str(s)
      result.append(w)

 #  i = 0
#  while (i<TOP) and (i<len(scores)) :
#    for w in candidates[scores[i]] : 
#      result.append(w)
#      print word+"> "+str(i)+" "+w
#    i=i+len(candidates[scores[i]])

  return result

#---------------------------------------------
def findCandidateTranslationsMixChiaoSeparate(word, nb, similarityFunction) :
  print word

  # Chiao in target space 
  candidates_Trgt = findCandidateScores(word, SOURCE_TRANSFERRED_VECTORS[word], TARGET_NETWORK, nb, similarityFunction)
  scores_Trgt = sorted(candidates_Trgt.keys(), reverse=True)
  res_Trgt = [] #Concatenation of Strings from candidates, ordered by their rank; len(translations) <= TOP
  for i in range(len(scores_Trgt)) :
    #if len(res1) >= nb*TOLERANCE_RATE : break
    for w in candidates_Trgt[scores_Trgt[i]] :
      res_Trgt.append(w)  
  d_rank_Trgt = {}
  ## Re-ranking in target space
  for cand in res_Trgt :   
    cand_reverse_Trgt = findCandidateScores(cand, TARGET_NETWORK[cand], SOURCE_TRANSFERRED_VECTORS, nb, similarityFunction)
    rank_Trgt = harmonicMean(getRank(candidates_Trgt, cand), getRank(cand_reverse_Trgt, word))
    if rank_Trgt not in d_rank_Trgt : d_rank_Trgt[rank_Trgt] = []
    d_rank_Trgt[rank_Trgt].append(cand)
  #print res1 #
  res1 = []
  ranksTrgt = sorted(d_rank_Trgt.keys(), reverse=False)
  # Give an ordered list of the translation candidates
  for i in range(len(ranksTrgt)) :
    for w in d_rank_Trgt[ranksTrgt[i]] :
      #print w+"> "+str(s)
      res1.append(w)
  print res1

  # Chiao in source space
  candidates_Src = findCandidateScores(word, SOURCE_NETWORK[word], TARGET_TRANSFERRED_VECTORS, nb, similarityFunction)
  scores_Src = sorted(candidates_Src.keys(), reverse=True)
  res_Src = [] #Concatenation of Strings from candidates, ordered by their rank; len(translations) <= TOP
  for i in range(len(scores_Src)) :
    #if len(res1) >= nb*TOLERANCE_RATE : break
    for w in candidates_Src[scores_Src[i]] :
      res_Src.append(w)
  #print res_Src #
  d_rank_Src = {}
  ## Re-ranking in source space
  for cand in res_Src :   
    cand_reverse_Src = findCandidateScores(cand, TARGET_TRANSFERRED_VECTORS[cand], SOURCE_NETWORK, nb, similarityFunction)
    rank_Src = harmonicMean(getRank(candidates_Src, cand), getRank(cand_reverse_Src, word))
    if rank_Src not in d_rank_Src : d_rank_Src[rank_Src] = []
    d_rank_Src[rank_Src].append(cand)
  res2 = []
  ranksSrc = sorted(d_rank_Src.keys(), reverse=False)
  # Give an ordered list of the translation candidates
  for i in range(len(ranksSrc)) :
    for w in d_rank_Src[ranksSrc[i]] :
      #print w+"> "+str(s)
      res2.append(w)
  print res2

  #Harmonic mean of both ranks 
  d_rank = {}
  for cand in set(res1) | set(res2) :
    cand_rank = harmonicMean(getRank(d_rank_Src, cand, False), getRank(d_rank_Trgt, cand, False))
    if cand_rank not in d_rank : d_rank[cand_rank] = []
    d_rank[cand_rank].append(cand)
  result = []
  ranks = sorted(d_rank.keys(), reverse=False)
  # Give an ordered list of the translation candidates
  for i in range(len(ranks)) :
    for w in d_rank[ranks[i]] :
      #print w+"> "+str(s)
      result.append(w)
  print result
  print "---"
  return result                 
#---------------------------------------------

#---------------------------------------------
def findCandidateTranslationsMixChiao(word, nb, similarityFunction) :
  # Mixed direct method
  print word
  candidates = findCandidateScores(word, SOURCE_TRANSFERRED_VECTORS[word], TARGET_NETWORK, nb, similarityFunction)
  scores = sorted(candidates.keys(), reverse=True)
  res1 = [] #Concatenation of Strings from candidates, ordered by their rank; len(translations) <= TOP
  for i in range(len(scores)) :
    #if len(res1) >= nb*TOLERANCE_RATE : break
    for w in candidates[scores[i]] :
      #print w+"> "+str(s)
      res1.append(w)
      #if len(res1) >= nb*TOLERANCE_RATE :
        #print "----early exit"
        #break
  #print res1 #
  
  candidatesReverse = findCandidateScores(word, SOURCE_NETWORK[word], TARGET_TRANSFERRED_VECTORS, nb, similarityFunction)
  scoresReverse = sorted(candidatesReverse.keys(), reverse=True)
  res2 = [] #Concatenation of Strings from candidates, ordered by their rank; len(translations) <= TOP
  for i in range(len(scoresReverse)) :
    #if len(res1) >= nb*TOLERANCE_RATE : break
    for w in candidatesReverse[scoresReverse[i]] :
      #print w+"> "+str(s)
      res2.append(w)
      #if len(res1) >= nb*TOLERANCE_RATE :
        #print "----early exit"
        #break
  #print res2 #

  mixedDirectRanks = {}
  d_rankD = {}
  for cand in set(res1) | set(res2) :
    cand_rank = harmonicMean(getRank(candidates, cand), getRank(candidatesReverse, cand))
    if cand_rank not in d_rankD : d_rankD[cand_rank] = []
    d_rankD[cand_rank].append(cand)
    mixedDirectRanks[cand] = cand_rank
  resultD = []
  ranksD = sorted(d_rankD.keys(), reverse=False)
  # Give an ordered list of the translation candidates
  for i in range(len(ranksD)) :
    for w in d_rankD[ranksD[i]] :
      #print w+"> "+str(s)
      resultD.append(w)
  print resultD
  #print "---"
  

  #Mixed reverse
  d_rankF = {}
  for cand in resultD :
    cand_reverse1 = findCandidateScores(cand, TARGET_TRANSFERRED_VECTORS[cand], SOURCE_NETWORK, nb, similarityFunction)
    cand_reverse2 = findCandidateScores(cand, TARGET_NETWORK[cand], SOURCE_TRANSFERRED_VECTORS, nb, similarityFunction)
    #Mixed reverse rank
    rankReverse = harmonicMean(getRank(cand_reverse1, cand), getRank(cand_reverse2, word))
    #Final rank
    cand_rank = harmonicMean(mixedDirectRanks[cand], rankReverse)
    if cand_rank not in d_rankF : d_rankF[cand_rank] = []
    d_rankF[cand_rank].append(cand)
    
  resultF = []
  ranksF = sorted(d_rankF.keys(), reverse=False)
  # Give an ordered list of the translation candidates
  for i in range(len(ranksF)) :
    for w in d_rankF[ranksF[i]] :
      #print w+"> "+str(s)
      resultF.append(w)
  print resultF
  print "---"

  #Final re-ranking
  return resultF  
#---------------------------------------------
def findCandidateTranslationsMixBase(word, nb, similarityFunction) :
  #print word
  candidates = findCandidateScores(word, SOURCE_TRANSFERRED_VECTORS[word], TARGET_NETWORK, nb, similarityFunction)
  scores = sorted(candidates.keys(), reverse=True)
  res1 = [] #Concatenation of Strings from candidates, ordered by their rank; len(translations) <= TOP
  for i in range(len(scores)) :
    #if len(res1) >= nb*TOLERANCE_RATE : break
    for w in candidates[scores[i]] :
      #print w+"> "+str(s)
      res1.append(w)
      #if len(res1) >= nb*TOLERANCE_RATE :
        #print "----early exit"
        #break
  #print res1 #
  
  candidatesReverse = findCandidateScores(word, SOURCE_NETWORK[word], TARGET_TRANSFERRED_VECTORS, nb, similarityFunction)
  scoresReverse = sorted(candidatesReverse.keys(), reverse=True)
  res2 = [] #Concatenation of Strings from candidates, ordered by their rank; len(translations) <= TOP
  for i in range(len(scoresReverse)) :
    #if len(res1) >= nb*TOLERANCE_RATE : break
    for w in candidatesReverse[scoresReverse[i]] :
      #print w+"> "+str(s)
      res2.append(w)
      #if len(res1) >= nb*TOLERANCE_RATE :
        #print "----early exit"
        #break
  #print res2 #
  
  d_rank = {}
  for cand in set(res1) | set(res2) :
    cand_rank = harmonicMean(getRank(candidates, cand), getRank(candidatesReverse, cand))
    if cand_rank not in d_rank : d_rank[cand_rank] = []
    d_rank[cand_rank].append(cand)
  result = []
  ranks = sorted(d_rank.keys(), reverse=False)
  # Give an ordered list of the translation candidates
  for i in range(len(ranks)) :
    for w in d_rank[ranks[i]] :
      #print w+"> "+str(s)
      result.append(w)
  #print result
  #print "---"
  return result                                    

  
def findCandidateTranslationsChiao(word, transferedVector, targetNetwork, nb, similarityFunction, f_transferTarget=None, f_filter_candidates=None) :
  #print "==========="
  #print word
  #list of the nb best scores found
  candidates = {}
  scores = []
  if f_filter_candidates is None :
    #print "f_filter_candidates is None"
    candidates = findCandidateScores(word, transferedVector, targetNetwork, nb, similarityFunction)
    scores = sorted(candidates.keys(), reverse=True)
  else :
    candidatesKeys = DICO.get(word, [])
    if len(candidatesKeys) == 0 :
      candidatesKeys = [k for k in targetNetwork.keys() if (len(DICO_INV.get(k, []))== 0)]
    filteredTargetNetwork = {k: v for k, v in targetNetwork.iteritems() if k in candidatesKeys}
    candidates = findCandidateScores(word, transferedVector, filteredTargetNetwork, nb, similarityFunction)
    scores = sorted(candidates.keys(), reverse=True)  

  res1 = [] #Concatenation of Strings from candidates, ordered by their rank; len(translations) <= TOP
  for i in range(len(scores)) :
    if len(res1) >= nb*TOLERANCE_RATE : break
    for w in candidates[scores[i]] :
      #print w+"> "+str(s)
      res1.append(w)
      if len(res1) >= nb*TOLERANCE_RATE :
        print "----early exit"
        break
  #print res1 #
  
  d_rank = {}
  
  for cand in res1 :
    if not (f_filter_candidates is None) :
      #print "f_filter_candidates is not None"
      if cand not in targetNetwork :
        print str(cand)+" rejected because not in target corpus"
        continue
      #else : print str(cand)
    #else : print "f_filter_candidates is not None"
    if cand in TARGET_TRANSFERRED_VECTORS :
      cand_transferedVector = TARGET_TRANSFERRED_VECTORS[cand]
    else :
      #print "transfer::"
      raise RuntimeError("f_transferTarget is not defined in findCandidateTranslationsChiao")
      cand_transferedVector = f_transferTarget(cand)
      TARGET_TRANSFERRED_VECTORS[cand] = cand_transferedVector
    cand_reverse = findCandidateScores(cand, cand_transferedVector, SOURCE_NETWORK, nb, similarityFunction)
    cand_rank = harmonicMean(getRank(candidates, cand), getRank(cand_reverse, word))
    if cand_rank not in d_rank : d_rank[cand_rank] = []
    d_rank[cand_rank].append(cand)

  result = []
  ranks = sorted(d_rank.keys(), reverse=False)
  # Give an ordered list of the translation candidates
  for i in range(len(ranks)) :
    for w in d_rank[ranks[i]] :
      #print w+"> "+str(s)
      result.append(w)
  #print "---"
  #print result
  return result

def findCandidateTranslations(word, transferedVector, targetNetwork, nb, similarityFunction, f_filter_candidates=None) :
  #list of the nb best scores found
  candidates = {}
  scores = []
  if f_filter_candidates is None :
    candidates = findCandidateScores(word, transferedVector, targetNetwork, nb, similarityFunction)
    scores = sorted(candidates.keys(), reverse=True)
  else :
    candidatesKeys = DICO.get(word, [])
    if len(candidatesKeys) == 0 :
      candidatesKeys = [k for k in targetNetwork.keys() if (len(DICO_INV.get(k, []))== 0)]
    filteredTargetNetwork = {k: v for k, v in targetNetwork.iteritems() if k in candidatesKeys}
    candidates = findCandidateScores(word, transferedVector, filteredTargetNetwork, nb, similarityFunction)
    scores = sorted(candidates.keys(), reverse=True)
  #print "========="
  #print scores

  result = [] #Concatenation of Strings from candidates, ordered by their rank; len(translations) <= TOP
# Give an ordered list of the translation candidates
  for i in range(len(scores)) :
    for w in candidates[scores[i]] :
      #print w+"> "+str(s)
      result.append(w)

 #  i = 0
#  while (i<TOP) and (i<len(scores)) :
#    for w in candidates[scores[i]] : 
#      result.append(w)
#      print word+"> "+str(i)+" "+w
#    i=i+len(candidates[scores[i]])

  return result
  
def findCandidateScoresMix(word, nb, similarityFunction, meanFunction) :
  """ nb : number of candidates scores to find """
  #print"====="
  TOP = nb
  scores = [] #list<Double> ; invariant : len(scores) <= TOP
  candidates = {} #Map< Double, list<String> >
  result = [] #Concatenation of Strings from candidates, ordered by their rank; len(translations) <= TOP
  rank_results = [] #Concatenation of couple - rank
  current_min = 10000 #TODO initialize with max double
  for c in TARGET_NETWORK :
    score_c = meanFunction(similarity(SOURCE_TRANSFERRED_VECTORS[word], TARGET_NETWORK[c], similarityFunction), similarity(SOURCE_NETWORK[word], TARGET_TRANSFERRED_VECTORS[c], similarityFunction))
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
        #print "ADDING ("+c+", "+str(score_c)+")"TARGET_NETWORK
        if score_c not in candidates : 
          scores.append(score_c)
          candidates[score_c] = []
        #else score_c is already in scores and in candidates' keyset
        candidates[score_c].append(c)
        # update current_min
        current_min = min(scores)
  # rank the results
  return candidates

def findCandidateScores(word, transferedVector, targetNetwork, nb, similarityFunction) :
  """ nb : number of candidates scores to find """
  #print"====="
  TOP = nb
  scores = [] #list<Double> ; invariant : len(scores) <= TOP
  candidates = {} #Map< Double, list<String> >
  result = [] #Concatenation of Strings from candidates, ordered by their rank; len(translations) <= TOP
  rank_results = [] #Concatenation of couple - rank
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
  return candidates

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
  if choice == JACCARD : result = sigma_XiYi / ( sigma_Xi2 + sigma_Yi2 - sigma_XiYi ) #CF Chiao et al.
  else : result = 0.5 * ((sigma_XiYi / ( sqrt(sigma_Xi2) * sqrt(sigma_Yi2) )) + (sigma_XiYi / ( sigma_Xi2 + sigma_Yi2 - sigma_XiYi )))
  return result

def makeTest(top_list, testset, transferedNetwork, targetNetwork, f_transferTarget) :
  candidates = {} #Map< String, List<String> >
  unknownSourceWords = set()
  for word in testset :
    #print ">>Candidates for '"+ word+"'"
    if word not in transferedNetwork :
      print word+" not in source corpus"
      unknownSourceWords.add(word)
      candidates[word] = []
    else :
      transferedVector = transferedNetwork[word] #getTransferedVector(word)
      #Base      candidates[word] = findCandidateTranslations(word, transferedVector, targetNetwork, max(top_list), SIMILARITY_FUNCTION)
      #Improvement 1 : Chiao      candidates[word] = findCandidateTranslationsChiao(word, transferedVector, targetNetwork, max(top_list), SIMILARITY_FUNCTION, f_transferTarget)
      #Improvement 2 : Good dictionary      candidates[word] = findCandidateTranslations(word, transferedVector, targetNetwork, max(top_list), SIMILARITY_FUNCTION, 'Yes')
      #Improvement 3 : Good dictionary + Chiao      candidates[word] = findCandidateTranslationsChiao(word, transferedVector, targetNetwork, max(top_list), SIMILARITY_FUNCTION, f_transferTarget, 'Yes')
      #Improvement 5 : Chiao reverse      candidates[word] = findCandidateTranslationsChiaoReverse(word, transferedVector, targetNetwork, max(top_list), SIMILARITY_FUNCTION, f_transferTarget)

  targetWords = map(Token.__str__, targetCorpus)
  testsetSize = len(testset) - len(unknownSourceWords)
  for top in top_list :
    tp = 0
    print "==========="
    print "TOP "+str(top)
    for word in testset :
      found = 0
      mistake = False
      for r in testset[word] :
        if r in targetWords : mistake = True
        if ( len(candidates[word]) > 0 ) and r in [candidates[word][i] for i in range( min([top, len(candidates[word])]) ) ] : 
          #print "====================" + r
          found = found+1
      if found > 0 :
        #print word
        tp = tp+1
      #else :
        #if not mistake :
          #print str(word) + " couldn't be found"
        #else :
          #print str(word)+" : "
          #print candidates[word]
      
    ###################################################
    # Print results' evaluation
    precision = float(tp) / len(testset)
    realPrecision = float(tp) / testsetSize
    print "tp = "+str(tp)+" /"+str(testsetSize)
    print "Precision = "+str(precision)
    print "Real precision = "+str(realPrecision)


def performTest(top_list, testset) :
  averageMAP = 0
  averageMAP_recall = 0
  averageMAP_best = 0
  candidates = {} #Map< String, List<String> >
  unknownSourceWords = set()
  for word in testset :
    #print ">>Candidates for '"+ word+"'"
    if word not in SOURCE_NETWORK :
      print word+" not in source corpus"
      unknownSourceWords.add(word)
      candidates[word] = []
    else :
      #transferedVector = transferedNetwork[word] #getTransferedVector(word)
      #Base                 candidates[word] = findCandidateTranslations(word, SOURCE_TRANSFERRED_VECTORS[word], TARGET_NETWORK, max(top_list), SIMILARITY_FUNCTION)
      #Improvement 1 : Chiao             candidates[word] = findCandidateTranslationsChiao(word, SOURCE_TRANSFERRED_VECTORS[word], TARGET_NETWORK, max(top_list), SIMILARITY_FUNCTION)
      #Improvement 2 : Good dictionary      candidates[word] = findCandidateTranslations(word, SOURCE_TRANSFERRED_VECTORS[word], TARGET_NETWORK, max(top_list), SIMILARITY_FUNCTION, 'Yes')
      #Improvement 3 : Good dictionary + Chiao      candidates[word] = findCandidateTranslationsChiao(word, SOURCE_TRANSFERRED_VECTORS[word], TARGET_NETWORK, max(top_list), SIMILARITY_FUNCTION, 'Yes')
      #Improvement 4 : Base reverse          candidates[word] = findCandidateTranslations(word, SOURCE_NETWORK[word], TARGET_TRANSFERRED_VECTORS, max(top_list), SIMILARITY_FUNCTION)
      #Improvement 5 : Chiao reverse      candidates[word] = findCandidateTranslationsChiaoReverse(word, SOURCE_NETWORK[word], TARGET_TRANSFERRED_VECTORS, max(top_list), SIMILARITY_FUNCTION)
      #Improvement 6 : Mix Base              candidates[word] = findCandidateTranslationsMixBase(word, max(top_list), SIMILARITY_FUNCTION)
      #Improvement 7 : Mix Arithmetic Mean            
      candidates[word] = findCandidateTranslationsMixMean(word, max(top_list), SIMILARITY_FUNCTION, arithmeticMean)
      #Improvement 8 : Mix Harmonic Mean           candidates[word] = findCandidateTranslationsMixMean(word, max(top_list), SIMILARITY_FUNCTION, harmonicMean)
      #Improvement 9 : Mix Chiao REFAIRE      candidates[word] = findCandidateTranslationsMixChiao(word, max(top_list), SIMILARITY_FUNCTION)
      #Improvement 10 : Mix Chiao separate : each 2-space rank is computed separately and we re-rank      candidates[word] = findCandidateTranslationsMixChiaoSeparate(word, max(top_list), SIMILARITY_FUNCTION)
      
      
  targetWords = TARGET_NETWORK.keys()
  testsetSize = len(testset) - len(unknownSourceWords)
  for top in top_list :
    tp = 0
    print "==========="
    print "TOP "+str(top)
    for word in testset :
      wordMAP = 0
      wordMAP_recall = 0
      wordMAP_best = 0
      found = 0
      mistake = False
      for r in testset[word] :
        if r in targetWords : mistake = True
        if ( len(candidates[word]) > 0 ) and r in [candidates[word][i] for i in range( min([top, len(candidates[word])]) ) ] : 
          #print "====================" + r 
          found = found+1
          wordMAP = wordMAP + ( 1.0 / (candidates[word].index(r)+1) )
          wordMAP_best = max(wordMAP_best, ( 1.0 / (candidates[word].index(r)+1) ))
      if found > 0 :
        wordMAP = float(wordMAP) / len(testset[word])
        wordMAP_recall = float(wordMAP) / found
        #wordMAP_best = wordMAP_best
        #print word + "\t"+ str(wordMAP)
        tp = tp+1
      averageMAP = averageMAP + wordMAP
      averageMAP_recall = averageMAP_recall + wordMAP_recall
      averageMAP_best = averageMAP_best + wordMAP_best
      #else :
        #if not mistake :
          #print str(word) + " couldn't be found"
        #else :
          #print str(word)+" : "
          #print candidates[word]
      
    ###################################################
    # Print results' evaluation
    precision = float(tp) / len(testset)
    realPrecision = float(tp) / testsetSize
    averageMAP = float(averageMAP) / testsetSize
    averageMAP_recall = float(averageMAP_recall) / testsetSize
    averageMAP_best = float(averageMAP_best) / testsetSize
    print "tp = "+str(tp)+" /"+str(testsetSize)
    print "Precision = "+str(precision)
    print "Real precision = "+str(realPrecision)
    print "MAP (classic) = "+str(averageMAP)
    print "MAP (recall) = "+str(averageMAP_recall)
    print "MAP (best) = "+str(averageMAP_best)

#-------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------
def bof() : 
  print ">LOADING COMPARABLE CORPORA..."
  #sourceCorpus = iniCorpusFR(CORPUS_SOURCE)
  corpus = []
  sentenceID = -1
  # List of stopwords and punctuation signes 
  with open(STOPWORDS_FR, "r") as f:
    content = f.read()
    stopwords = content.split("\n")
  with open("test.txt", "r") as f:
    for sentence in f.split("."): 
      if sentence.startswith("__") or sentence == " " : continue
      sentenceID = sentenceID + 1
      tokens = sentence.split(" ")
      for i in range(len(tokens)):
        tmp = tokens[i]
        # Filter stopwords and ponctuation
        if (tokens[i] not in stopwords) :
          #print ">" + tokens[i]
          t = Token("", tokens[i], "", sentenceID)
          #print tmp[0] + " " +tags[0]
          corpus.append(t)
          SOURCE_SPACE.add(str(t))
          #else : print tmp[0]+" "+tmp[1]
  print ">COMPUTING CONTEXT VECTORS..."
  start_time = time.time()
  
  windowSize = 2

  cvsFileSource = "test_source_network.cvs"
  POS_to_keep = []
  sourceNetwork = computeContextVectors(corpus, windowSize)
  saveContextNetwork(sourceNetwork, cvsFileSource)

if __name__ == "__main__":
  #--testset = getTestSet("testset.cvs")
  #--print testset
  # Load comparable corpus : termer_{source,target}/corpus.lem
  print ">LOADING COMPARABLE CORPORA..."
  sourceCorpus = iniCorpusFR(CORPUS_SOURCE)
  targetCorpus = iniCorpusEN(CORPUS_TARGET)

  # Compute cognates dictionary
  threshold = 5
  # TODO cognates.computeCognatesDictionary(threshold, sourceCorpus, targetCorpus, COGNATES_DICO)

  # Load bilingual dictionaries
  print ">LOADING BILINGUAL DICTIONARY..."
  getBilingualDictionary(BILINGUAL_DICO) #, bilingualDico, bilingualDico_inv)
  bilingualDico = DICO
  bilingualDico_inv = DICO_INV


  # Load entries to translate

###################################################
  # Perform direct method

## Computing context vectors for 
## nouns
## verbs (expect be, have)
## adjectives
## and adverbs
  print ">COMPUTING AND NORMALIZING ("+NORMALIZATION+") CONTEXT VECTORS..."
  start_time = time.time()
  
  windowSize = 2

  POS_to_keep = []
  if (os.path.isfile(SOURCE_NETWORK_FILE)) : unserialize(SOURCE_NETWORK_FILE, SOURCE_NETWORK)
  else :
    SOURCE_NETWORK = computeContextVectors(sourceCorpus, windowSize)
    if (NORMALIZATION == TFIDF) : normalizeTFIDF(SOURCE_NETWORK)
    elif (NORMALIZATION == LO) : normalizeLO1(SOURCE_NETWORK)
    #else : NONE
    serialize(SOURCE_NETWORK_FILE, SOURCE_NETWORK)
    
  if (os.path.isfile(TARGET_NETWORK_FILE)) : unserialize(TARGET_NETWORK_FILE, TARGET_NETWORK)
  else : 
    TARGET_NETWORK = computeContextVectors(targetCorpus, windowSize)
    if (NORMALIZATION == TFIDF) : normalizeTFIDF(TARGET_NETWORK)
    elif (NORMALIZATION == LO) : normalizeLO1(TARGET_NETWORK)
    #else : NONE
    serialize(TARGET_NETWORK_FILE, TARGET_NETWORK)
  
  elapsed_time = time.time() - start_time
  print str(elapsed_time)

##def targetToSource() :
#### TargetToSource
#### Transferring context vectors
### use bilingual dictionary 
##  print ">TRANSFERRING CONTEXT VECTORS..."
##  start_time = time.time()
##
##  sourceFreqCounts = Counter(map(Token.__str__, sourceCorpus))
##  transferedNetwork = {}
##  for word in targetNetwork : 
##    transferedNetwork[word] = transfer(word, targetNetwork[word], bilingualDico_inv, sourceFreqCounts)
##
##  print "Source space :"+str(len(SOURCE_SPACE))+", Pivot words : "
##  print "Target space :"+str(len(TARGET_SPACE))+", Pivot words : "+str(len(PIVOT_WORDS_SOURCE))
##
##  targetFreqCounts = Counter(map(Token.__str__, targetCorpus))
##  for word in sourceNetwork : 
##    SOURCE_TRANSFERRED_VECTORS[word] = transfer(word, sourceNetwork[word], bilingualDico, targetFreqCounts)
##  
###Test  
##  print ">TESTING..."
##    
##  #testset = test.getTestSet()
##  testset = getTestSet("testset.cvs")
##  #print testset
##  results = {}
#### Finding candidate translations
##  start_time = time.time()
##  def ftransferSource(word) : return transfer(word, sourceNetwork[word], bilingualDico, Counter(map(Token.__str__, targetCorpus)))
##  #makeTest([40], testset, transferedNetwork, targetNetwork, ftransferTarget)
##  makeTest([30, 20, 10, 5, 1], testset, sourceNetwork, transferedNetwork, ftransferSource) # 90, 80, 70, 60, 50, 40, 
##  elapsed_time = time.time() - start_time
##  print str(elapsed_time)
##

  
#def sourceToTarget() :
## Transferring context vectors
# use bilingual dictionary 
  print ">TRANSFERRING CONTEXT VECTORS..."
  start_time = time.time()

  if (os.path.isfile(SOURCE_TRANSFERRED_VECTORS_FILE)) : unserialize(SOURCE_TRANSFERRED_VECTORS_FILE, SOURCE_TRANSFERRED_VECTORS)
  else : 
    SOURCE_TRANSFERRED_VECTORS = {}
    targetFreqCounts = Counter(map(Token.__str__, targetCorpus))
    for word in SOURCE_NETWORK : 
      SOURCE_TRANSFERRED_VECTORS[word] = transfer(word, SOURCE_NETWORK[word], bilingualDico, targetFreqCounts)
    serialize(SOURCE_TRANSFERRED_VECTORS_FILE, SOURCE_TRANSFERRED_VECTORS)
  print "Source space :"+str(len(SOURCE_SPACE))+", Pivot words : "+str(len(PIVOT_WORDS))
  print "Target space :"+str(len(TARGET_SPACE))+", Pivot words : " #PIVOT_WORDS_TARGET
  PIVOT_WORDS = set()
  
#  TARGET_TRANSFERRED_VECTORS_FILE = "target_transferred_vectors.csv"
#  SOURCE_TRANSFERRED_VECTORS_FILE = "source_transferred_vectors.csv"
#  TARGET_NETWORK_FILE = "target_network.csv"
#  SOURCE_NETWORK_FILE = "source_network.csv"

  if (os.path.isfile(TARGET_TRANSFERRED_VECTORS_FILE)) : unserialize(TARGET_TRANSFERRED_VECTORS_FILE, TARGET_TRANSFERRED_VECTORS)
  else : 
    TARGET_TRANSFERRED_VECTORS = {}
    sourceFreqCounts = Counter(map(Token.__str__, sourceCorpus))
    for word in TARGET_NETWORK : 
      TARGET_TRANSFERRED_VECTORS[word] = transfer(word, TARGET_NETWORK[word], bilingualDico_inv, sourceFreqCounts)
    serialize(TARGET_TRANSFERRED_VECTORS_FILE, TARGET_TRANSFERRED_VECTORS)
  print "Source space :"+str(len(SOURCE_SPACE))+", Pivot words : "
  print "Target space :"+str(len(TARGET_SPACE))+", Pivot words : "+str(len(PIVOT_WORDS))

#Test  
  print ">TESTING..."
    
  #testset = test.getTestSet()
  testset = getTestSet("testset.cvs")
  #print testset
  results = {}
## Finding candidate translations
  start_time = time.time()
  #def ftransferTarget(word) : return transfer(word, targetNetwork[word], bilingualDico_inv, Counter(map(Token.__str__, sourceCorpus)))
  #makeTest([40], testset, transferedNetwork, targetNetwork, ftransferTarget)
  #makeTest([30, 20, 10, 5, 1], testset, transferedNetwork, targetNetwork, ftransferTarget) # 
  performTest([90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 1], testset) # 
  elapsed_time = time.time() - start_time
  print str(elapsed_time)

def next() : 
  start_time = time.time()
  makeTest(20, testset, transferedNetwork, targetNetwork)
  elapsed_time = time.time() - start_time
  print str(elapsed_time)
  
  start_time = time.time()
  makeTest(10, testset, transferedNetwork, targetNetwork)
  elapsed_time = time.time() - start_time
  print str(elapsed_time)


