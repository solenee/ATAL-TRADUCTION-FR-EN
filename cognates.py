#coding:utf-8
#-------------------------------
# Compute a dictionary of cognates based on two comparable corpora
# Write the dictionary as a file
#-------------------------------
import re, sys, os #, nltk
#1 import itertools
from types import *

def hammingDistance(str1, str2):
  """Compute the number of differences between equal length strings str1 and str2"""
  if len(str1) != len(str2):
    return sys.maxint #raise ValueError("Undefined for sequences of unequal length")
  else: 
    return sum(ch1 != ch2 for ch1, ch2 in zip(str1, str2))
  #1 return sum(itertools.imap(str.__ne__, str1, str2))


def computeCognatesDictionary(threshold, words_source, words_target, dictionaryFileName):
  """Compute a file containing the cognates find in the lists of words words_source and words_target"""
  myCognates = {}
  # Look for cognates
  for i in range(len(words_source)):
    word = words_source[i]
    best_candidate = ""
    best_score = threshold
    for j in range(len(words_target)):
      d = hammingDistance(str(word), str(words_target[j]))
      if (d < best_score):
        best_candidate = words_target[j]
        best_score = d
    if (best_candidate != ""):
      cognate = best_candidate
      myCognates[word] = cognate
  #filter false cognates
  #write dictionary file
  with open(dictionaryFileName, "w") as f:
    for k, v in myCognates.items():
      print "%s %s" % (k, v)





