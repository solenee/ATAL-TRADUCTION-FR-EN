#-------------------------------
# Defines some useful types for corpora analysis
#-------------------------------

class Token:
  """Token with covered text, POS, lemma and the ID of the sentence it belongs to"""
  def __init__(self, coveredText="", pos="", lemma="", sentenceID=-1):
    self.coveredText = coveredText
    self.pos = pos
    self.lemma = lemma
    self.sentenceID = sentenceID

  def __str__(self) : return self.lemma
  def __repr__(self) : 
    return "Token(%s + %s + %s + %i)"%(self.coverdText, self.pos, self.lemma, self.sentenceID)

if __name__ == "__main__":
  # Test
  t = Token("Hello", "NN", "hello")
  print t
