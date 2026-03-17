# AroMIP: A predictor of membrane insertion by aromatic-centered motifs of intrinsically disordered proteins

This repository contains source data and codes for the AroMIP project. AroMIP scans a given protein sequence for 9-residue fragments centered on an aromatic residue (F, W, Y), and predicts whether the aromatic sidechain in each fragment inserts into the acyl chain region of membranes. The prediction is based on a mathematical model, where each flanking residue contributes a multiplicative factor to the membrane insertion score of the central aromatic residue. The model was trained on the intrinsically disordered regions of the human proteome.

The AroMIP method is described in:

Fidha Kunnath Muhammedkutty and Huan-Xiang Zhou (2026), A membrane insertion code for intrinsically disordered proteins, bioRxiv.

### source data

Source data for the above publication are found in the folder "Source Data"

### codes

Codes are found in the folder "Codes". The file aromip.js contains the javascript code for AroMIP prediction. To run, use the following command:

node aromip.js RRNKFGINRTTGNWRGMLQRDLYSGLN

where “RRNKFGINRTTGNWRGMLQRDLYSGLN” is the protein sequence in one-letter representation. AroMIP is also available as a web server at
https://zhougroup-uic.github.io/AroMIP/

Python codes for training and testing are in the subdirectory Training_codes under the name AroMIP_training_codes.ipynb.
