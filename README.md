# Hindi–English Neural Translator (MBART)

A Transformer-based Neural Machine Translation (NMT) system that translates between English and Hindi using the mBART-large-50 model, fine-tuned for high accuracy and fluency. Built with Hugging Face Transformers, PyTorch, and Pandas.

---

## Overview
This project fine-tunes Facebook’s `facebook/mbart-large-50-one-to-many-mmt` for translating English to Hindi. It handles preprocessing steps like lowercasing, punctuation/digit stripping, sentence length filtering, and BPE tokenization. Training employs Adam with warm-up and linear decay, early stopping on validation loss, and beam search decoding to generate fluent, high-accuracy translations.

---

## Dataset
- **Source**: *Hindi–English Truncated Corpus* (Kaggle, by umasrikakollu72)  
- **Format**: CSV with sentence-aligned English–Hindi pairs (`english_sentence`, `hindi_sentence`)  
- **Preprocessing Steps**:
  - Lowercase text, remove punctuation/digits  
  - Filter by sentence length  
  - Add custom tokens (`START_`, `_END`) to Hindi sentences  
  - Use only first 20,000 entries, filter for consistent style (e.g. source = “ted”)

---

## Model & Architecture
- **Pre-trained Base**: `facebook/mbart-large-50-one-to-many-mmt` from Hugging Face  
- **Framework**: Hugging Face Transformers + PyTorch  
- **Fine-tuning**:
  - Optimizer: **Adam** with learning-rate warm-up and linear decay  
  - Early stopping based on validation loss  
  - Beam-search decoding for inference  

---
