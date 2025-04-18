# PMDHAN Peptide Sequence Alignment Data

## File Description
This repository contains Peptide-Peptide assocition data processed using the Smith-Waterman algorithm:

- `pep-pep.dat`: Peptide sequence alignment data file containing local alignment results

## Download Instructions
1. Access the data file via Google Drive:
https://drive.google.com/file/d/13Z4_Ry36UKOIrDzc-SjtoLSvhS_33kwL/view?usp=drive_link

2. Click the download button (↓) in the top-right corner
3. The downloaded file contains:
PMDHAN/
└── pep-pep.dat

## About Smith-Waterman Algorithm
The Smith-Waterman is a dynamic programming method for **local sequence alignment** of biological sequences (DNA/Protein(Peptide)).

### Key Features:
- **Local alignment**: Finds best matching subregions while ignoring dissimilar parts
- **Scoring system**: 
- Uses substitution matrices (e.g., BLOSUM, PAM)
- Incorporates gap penalties for insertions/deletions
- **Backtracking**: Alignment reconstruction starts from highest scoring cell

### Typical Applications:
- Identifying conserved peptide motifs
- Finding local gene similarities
- Detecting functional domains

### Example Alignment:
| Position | 1 | 2 | 3 | 4 | 5 |
|----------|---|---|---|---|---|
| Query:   | A | C | C | T | A |
|          | - |   | - | - | - |
| Target:  | A | G | C | T | A |
