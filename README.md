# Content

Dieses Projekt soll alle Bausteine beinhalten, die für die Datenanalyse und das 
Preprocessing der Daten vor der Anwendung von ML-Modellen benötigt werden. Dazu 
gehören:
- Deskriptive Datenanalyse 
- Korrelationsanalyse
- Skalierung der Daten
- Encoding

Korrelationsanalyse

Vorgehen: 
1) Schritt: Korrelationen zwischen allen kardinalen Features ermitteln
gibt Aufschluss darüber, ob, in welchem Maße und in welcher 
   Richtung Features miteinander zusammenhängen

2) Schritt: Nominale Features betrachten
Soll nur geprüft werden, ob die Verteilungen voneinander abhängen 
   können auch ordinale Features mit einbezogen werden
   
3) Schritt: Metrisch vs. nominal / ordinal
Plot und Vergleich der Mittelwerte je Kategorie - gibt 
   es Unterschiede und sind sie gerichtet?

Datentypen: dichotom, nominal, ordinal, kardinal

dichotom - dichotom: yule
nominal - nominal: cramers_v
nominal - ordinal:
ordinal - ordinal: kendals_tau
nominal - kardinal:
ordinal - kardinal: 
kartdinal - kartdinal: pearson