This directory provided the given labels for the emotion recognition challenge.

The labels are structured in textfiles, where each line represents one given utterance/chunk.

The structure is:

<name of utterance> <LABELTYPE (A,E,N,P,R)> <confidence>
so e.g.:
Ohm_01_015_00 N 1

Means that the utterance Ohm_01_015_00 was classified to 100% as being the emotion N. The confidence is a factor resulting from that the given participants in the dataset could not always accurately define an emotion, leading to confidence scores smaller than 100%.

YOU DO NOT NEED TO TAKE CARE OF THE CONFIDENCE