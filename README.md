# ATAL-TRADUCTION-FR-EN
SUBJECT=Alignement de chaînes et de textes

GOAL=Implement automatic traduction from french into english

WORK TO DO
CM1-1/ Identifier les transfuges et les cognats présents dans le corpus comparable
CM1-2/ Evaluation du travail effectué
CM1-2/a- Nombre de transfuges et de cognats identifiés
CM1-2/b- Qualité des transfuges et des cognats identifiés : precision, recall, F-measure 2PR/(P+R) {good: 1, bad:0)

FORMULES
- Indice de ressemblance CM1,17 (Débili and Sammouda, 1992)
- Degré de comparabilité d’un corpus comparable CM2,9

NOTIONS UTILES 
- Genre
-Type de discours
- Méthode directe : CM2,10 "You shall know a word by the company it keeps" Firth (1957)

DEFINITIONS
- alignement de textes : mettre en correspondance automatiquement des
unités aux sens identiques à travers l’exploitation de corpus monolingues ou bilingues
=> Applications : traduction automatique, aide à la ~, RI monolingue et multi-lingue, extraction de lexiques
- corpus : ensemble de documents présentant une certaine homogénéité (période, langue parlée ou écrite, genre, type de discours,...)
=> type : { monolingues, bilingues {parallèles, comparables}, multimodaux }
"bilingues parallèles" = textes en relation de traduction 
"bilingues comparables" = pas de relation de traduction mais les textes abordent une même thématique sur une même période : ils sont 'comparables'
"multimodaux" = constitués de documents de nature différentes : texte électronique, texte manuscrit, parole, vidéo

- Indices d'alignement : {transfuges, cognats}
* cognats = {
véritables cognats : reconnaissance (FR) et recognition (EN)
faux amis : blesser (EN) et bless (EN)
cognats partiels dépendant du contexte : facteur (FR) et factor/mailman (EN)
cognats génétiques partageant des origines communes : chef (FR) et head (EN)
mots non reliés : glace (FR) et ice (EN)
}

- degré de comparabilité d’un corpus comparable P = l'espérance de trouver la traduction d’un mot du vocabulaire source (respectivement cible) dans le vocabulaire cible (respectivement source)
