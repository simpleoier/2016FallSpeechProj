- The lexicon file 'lexicon.txt' is plain ASCII; the coding is SAMPA.
- Each character represents one phone, apart from ":", 
  which characterizes lengthening of the left adjacent vowel, 
  e.g. "a:", "e:", "u:", etc.
- Syllable boundaries are denoted with "|".
- The position of the word stress is marked with "'".
- Glottal stop at vocalic word onset is marked with "?";
  this can be found within compounds a s well.
- If your ASR system cannot deal with these phenomena, 
  the three characters "?", "|", and "'" can be deleted.
- Lexicon entries with the character "*" as first letter denote non-words/fragments.
- "**" in the transcription denotes verbal noise that cannot be transcribed. 
