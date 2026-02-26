#!/bin/sh

echo "Splitting master file reference_candidate.tsv into reference.bio and candidate.bio ..."

cut -f 1-5 reference_candidate.tsv | perl -lne 'if (/^\t/) {print ""} else {print}' > reference.bio
cut -f 6-10 reference_candidate.tsv | perl -lne 'if (/^\t/) {print ""} else {print}' > candidate.bio
