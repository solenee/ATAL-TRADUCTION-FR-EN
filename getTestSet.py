#coding:utf-8

#-------------------------------
# Parse ts.xml file to save the
# test set into a CVS file
# separatation : " : "
#-------------------------------

import re, sys, os
import xml.etree.ElementTree as ET
import codecs

def isValid(tradNode) :
  return tradNode.get('valid') == 'yes'

def isTarget(langNode) :
  return langNode.get('type') == 'target'

def getSource(tradNode) :
  def isSource(langNode) :
    return langNode.get('type') == 'source'
  return filter(isSource, tradNode.iter('LANG'))[0].find('LEM').text

def getLEM(langNode) :
  return langNode.find('LEM').text

xmlFile = 'ts.xml'
cvsChar = " : "
cvsFile = 'testset.cvs'
tree = ET.parse(xmlFile)
root = tree.getroot()
content = []

if __name__ == "__main__":
  with codecs.open(cvsFile, "w", "utf-8") as f:
    for tradNode in filter(isValid ,root.iter('TRAD')):
      source = getSource(tradNode)
      trads = map(getLEM, filter(isTarget, tradNode.iter('LANG')))
      for i in range(len(trads)) :
        f.write(source+cvsChar+trads[i]+"\n")
        print source+cvsChar+trads[i]
  
def getTestSet() :
  testset = {}
  for tradNode in filter(isValid ,root.iter('TRAD')):
    source = getSource(tradNode)
    testset[source] = []
    #print source
    trads = map(getLEM, filter(isTarget, tradNode.iter('LANG')))
    for i in range(len(trads)) :
      testset[source].append(trads[i])
      #print source+cvsChar+trads[i]
  return testset
    
