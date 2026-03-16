# AroMIP: A predictor of membrane insertion by aromatic-centered motifs of intrinsically disordered proteins

AroMIP scans a given protein sequence for 9-residue fragments centered on an aromatic residue (F, W, Y), and predicts whether the aromatic sidechain in each fragment inserts into the acyl chain region of membranes. The prediction is based on a mathematical model, where each flanking residue contributes a multiplicative factor to the membrane insertion score of the central aromatic residue. The model was trained on the intrinsically disordered regions of the human proteome.

It is reported in the following publication:
F. N. K. Muhammedkutty, and H.-X. Zhou (2026). "A membrane-insertion code for intrinsically disordered proteins." Biorxiv

### codes
Python codes for training and testing is available as:
Training_codes/AroMIP_training_codes.ipynb

The file aromip.js contains the javascript code for AroMIP here. To run, use the following command: 

```
node aromip.js RRNKFGINRTTGNWRGMLQRDLYSGLN
```

where “RRNKFGINRTTGNWRGMLQRDLYSGLN” is the protein sequence in one-letter representation.
