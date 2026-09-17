# Multimodal Emotion Recognition

A deep learning system for recognizing human emotions by jointly modeling **text, audio, and visual information**.

The project explores multiple unimodal and multimodal learning approaches on the **CMU-MOSEI** dataset, including **BERT-based text modeling, GloVe representations, audio-visual fusion, and tri-modal early fusion using PyTorch**.

The objective is to understand how combining linguistic, acoustic, and visual signals can improve emotion recognition compared with relying on a single modality.

---

## Project Overview

Human emotion is inherently multimodal.

The meaning of a spoken sentence alone may not fully capture a person's emotional state. Tone of voice, facial expressions, and linguistic context can each contribute different signals.

For example:

```text
Text:   "Yeah, that's exactly what I wanted."
Audio:  sarcastic tone
Visual: negative facial expression
```

Analyzing only the text may incorrectly interpret the statement as positive.

This project therefore models three complementary modalities:

- **Text** — linguistic information from spoken sentences
- **Audio** — acoustic and vocal characteristics
- **Visual** — facial and visual behavioral features

These representations are evaluated independently and through **early-fusion architectures** for multi-label emotion recognition.

---

## Key Highlights

- Developed a **multimodal deep learning pipeline** combining text, audio, and visual signals for emotion recognition.
- Fine-tuned **BERT (`bert-base-uncased`)** for six-label emotion modeling and feature extraction.
- Explored both **BERT and GloVe** representations for linguistic information.
- Implemented **audio + visual** and **audio + visual + text** early-fusion architectures using PyTorch.
- Constructed an **877-dimensional tri-modal representation** combining BERT, acoustic, and visual features.
- Trained emotion-specific neural classifiers for **six emotion categories**.
- Evaluated models using **accuracy, precision, recall, classification reports, and balanced accuracy** to account for class imbalance.
- Compared how different combinations of modalities influence emotion-recognition performance.

---

## Emotion Categories

The system evaluates six emotion labels:

```text
Anger
Disgust
Fear
Happiness
Sadness
Surprise
```

Because a single video segment can exhibit multiple emotional characteristics, the problem is treated as a **multi-label emotion-recognition task**.

---

## Dataset

The project uses the **CMU Multimodal Opinion Sentiment and Emotion Intensity (CMU-MOSEI)** dataset.

CMU-MOSEI provides aligned linguistic, acoustic, and visual information extracted from video segments.

### Modalities

#### Text

Linguistic information includes:

- timestamped words
- GloVe word vectors
- contextual BERT representations

#### Audio

Acoustic information captures characteristics of speech such as vocal behavior and prosody.

The project works with CMU-MOSEI acoustic feature representations, including **COVAREP features**.

#### Visual

Visual information captures facial and behavioral signals.

CMU-MOSEI provides visual representations including features derived from:

- OpenFace
- FACET

---

## Multimodal Architecture

The overall architecture can be summarized as:

```text
                         Video Segment
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
           Text             Audio            Visual
             │                │                │
             ▼                ▼                ▼
            BERT          Acoustic          Visual
         Embeddings       Features         Features
             │                │                │
             │                │                │
             └────────────┬───┴────────────────┘
                          │
                          ▼
                     Early Fusion
                          │
                          ▼
               877-Dimensional Vector
                          │
                          ▼
                 Feed-Forward Network
                          │
                          ▼
                 Emotion Prediction
                          │
         ┌────────────────┼────────────────────┐
         ▼                ▼                    ▼
       Anger          Happiness              Fear
       Disgust        Sadness               Surprise
```

---

# Technical Approach

## 1. Text Representation with BERT

The text branch uses **BERT (`bert-base-uncased`)** to learn contextual representations of spoken language.

Unlike static word embeddings, BERT generates representations based on the surrounding context of each word.

For example:

```text
"I am fine."
```

may have a different emotional meaning depending on the words and context surrounding it.

A custom multi-label BERT classification architecture was implemented using PyTorch and Hugging Face Transformers.

The model consists of:

```text
Input Sentence
      │
      ▼
BERT Tokenization
      │
      ▼
bert-base-uncased
      │
      ▼
Contextual Representation
      │
      ▼
Dropout
      │
      ▼
Linear Classification Layer
      │
      ▼
Six Emotion Outputs
```

The model predicts six emotion labels and uses:

```text
BCEWithLogitsLoss
```

for multi-label training.

BERT's **768-dimensional hidden representation** is also extracted for downstream multimodal fusion.

---

## 2. GloVe Text Representation

In addition to contextual BERT embeddings, the project explores **GloVe word embeddings** as an alternative linguistic representation.

This provides a useful comparison between:

```text
Static Word Representation
        vs.
Contextual Transformer Representation
```

GloVe provides fixed vector representations for words, whereas BERT generates contextual embeddings based on the full sentence.

These experiments help evaluate the impact of representation quality on emotion-recognition performance.

---

## 3. Audio Feature Processing

The audio branch represents characteristics of the speaker's voice.

The CMU-MOSEI acoustic features used in the project include **COVAREP-based representations**.

Acoustic features capture information associated with:

- speech characteristics
- vocal intensity
- pitch-related behavior
- prosodic patterns
- voice quality

The audio pipeline can be summarized as:

```text
Video
  │
  ▼
Audio Signal
  │
  ▼
Acoustic Feature Extraction
  │
  ▼
Feature Normalization
  │
  ▼
Audio Representation
```

In the audio-visual experiments, the audio representation contains **74 acoustic features** per aligned timestep.

---

## 4. Visual Feature Processing

The visual modality provides information about facial expressions and visible behavioral signals.

The project uses visual representations derived from the CMU-MOSEI feature set.

These include facial-analysis features generated using tools such as **OpenFace** and **FACET**.

The visual pipeline follows:

```text
Video Frames
     │
     ▼
Facial Feature Extraction
     │
     ▼
Visual Feature Representation
     │
     ▼
Feature Normalization
```

The audio-visual experiments use **35 visual features** per aligned timestep.

---

## 5. Audio + Visual Early Fusion

The project first investigates multimodal learning without textual information by combining:

```text
Audio Features + Visual Features
```

The two modalities are aligned and fused before classification.

```text
Audio Representation
        │
        ├─────────────┐
        │             │
        │             ▼
        │        Feature Fusion
        │             │
        │             ▼
Visual Representation
                      │
                      ▼
              Neural Classifier
                      │
                      ▼
              Emotion Prediction
```

This experiment evaluates how effectively non-verbal signals alone can identify emotional states.

---

## 6. Text + Audio + Visual Early Fusion

The main multimodal approach combines all three modalities.

The fused feature representation consists of:

```text
BERT Text Features      = 768 dimensions
Audio Features          = 74 dimensions
Visual Features         = 35 dimensions
                          ─────────────
Combined Representation = 877 dimensions
```

The resulting feature vector is passed through a feed-forward neural network.

### Neural Network Architecture

```text
877 Input Features
        │
        ▼
Dense Layer — 128
        │
       ReLU
        │
     Dropout
        │
        ▼
Dense Layer — 64
        │
       ReLU
        │
     Dropout
        │
        ▼
Dense Layer — 32
        │
       ReLU
        │
        ▼
Binary Emotion Output
```

Separate emotion classifiers are evaluated for each target emotion.

This architecture allows the network to jointly learn from:

- semantic meaning from text
- vocal characteristics from audio
- facial and behavioral signals from video

---

## 7. Emotion Classification

The six emotion labels are evaluated independently as binary classification tasks.

For each emotion:

```text
Multimodal Features
        │
        ▼
Neural Network
        │
        ▼
Probability Score
        │
        ▼
Emotion Present / Not Present
```

The approach supports multi-label prediction because a segment can contain more than one emotional signal.

---

## 8. Model Evaluation

Emotion datasets can be highly imbalanced.

For example, some emotions may occur considerably less frequently than others.

Because of this, relying exclusively on raw accuracy can be misleading.

The project evaluates models using:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1-oriented classification reports
- Confusion matrices

**Balanced accuracy** is particularly useful because it accounts for performance across both positive and negative classes.

---

## Experimental Results

The tri-modal **text + audio + visual early-fusion model** was evaluated independently across the six emotion categories.

| Emotion | Accuracy | Balanced Accuracy |
|---|---:|---:|
| Happiness | 65% | 65.1% |
| Sadness | 60% | 61.6% |
| Anger | 63% | 64.5% |
| Disgust | 70% | 68.9% |
| Surprise | 61% | 57.9% |
| Fear | 80% | 61.0% |

The difference between raw accuracy and balanced accuracy for certain emotions highlights the impact of **class imbalance** and the importance of evaluating multimodal emotion-recognition systems with metrics beyond accuracy alone.

The experiments demonstrate that emotion recognition is highly dependent on both the emotion category and the information available across modalities.

---

## End-to-End Pipeline

```text
                         CMU-MOSEI
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
        Text                Audio               Video
          │                   │                   │
          ▼                   ▼                   ▼
        BERT               COVAREP          Visual Features
          │                   │                   │
          ▼                   ▼                   ▼
  768-D Representation   74-D Features       35-D Features
          │                   │                   │
          └───────────────────┼───────────────────┘
                              │
                              ▼
                         Early Fusion
                              │
                              ▼
                    877-D Representation
                              │
                              ▼
                  Feed-Forward Neural Network
                              │
                              ▼
                 Emotion-Specific Classifiers
                              │
                              ▼
      Anger • Disgust • Fear • Happy • Sad • Surprise
```

---

## Tech Stack

### Programming

- Python

### Machine Learning & Deep Learning

- PyTorch
- Hugging Face Transformers
- BERT
- GloVe
- scikit-learn

### Multimodal Processing

- CMU-MOSEI
- COVAREP acoustic features
- OpenFace visual features
- FACET visual features

### Data Processing

- NumPy
- pandas
- h5py

### Evaluation & Visualization

- scikit-learn metrics
- Matplotlib
- Seaborn

### Development

- Jupyter Notebook
- Google Colab
- GPU-accelerated PyTorch training

---

## Repository Structure

```text
Multimodal-Emotion-Recognition/
│
├── Source Code Files/
│   │
│   ├── BERT Emo - 2.ipynb
│   │   └── BERT-based text emotion modeling and feature extraction
│   │
│   ├── rev glove.ipynb
│   │   └── GloVe-based text representation experiments
│   │
│   ├── Early Fusion Diss Audio+Video - PyTorch - 2.ipynb
│   │   └── Audio + visual multimodal early-fusion experiments
│   │
│   ├── Early Fusion Diss Audio+Video+Text.ipynb
│   │   └── Tri-modal text + audio + visual early-fusion model
│   │
│   └── Modify Scores.ipynb
│       └── Label and score preprocessing
│
├── Dissertation/
│   └── Supporting project files
│
├── cmu_mosei.py
│   └── CMU-MOSEI feature and label resource definitions
│
├── raw sentence.xlsx
│   └── Text data used during preprocessing and experimentation
│
├── aa497_full_text.docx
│   └── Project documentation
│
└── README.md
```

---

## Running the Project

The original experiments were developed using **Google Colab** and use Google Drive paths for datasets and generated features.

### 1. Clone the Repository

```bash
git clone https://github.com/ayushiiamin/Multimodal-Emotion-Recognition.git
cd Multimodal-Emotion-Recognition
```

### 2. Install Dependencies

A typical environment requires:

```bash
pip install torch torchvision transformers
pip install numpy pandas scikit-learn h5py
pip install matplotlib seaborn
```

### 3. Obtain CMU-MOSEI Features

The repository includes `cmu_mosei.py`, which references CMU-MOSEI resources for:

- timestamped words
- GloVe vectors
- COVAREP acoustic features
- OpenFace visual features
- FACET visual features
- emotion labels

Download and preprocess the required modalities before running the fusion notebooks.

### 4. Update Dataset Paths

The original notebooks contain Google Drive paths such as:

```text
/content/drive/MyDrive/Dissertation/...
```

Update these paths to match the location of the dataset on your system or Google Drive.

### 5. Run the Experiments

The notebooks can be executed independently depending on the modality configuration being evaluated:

```text
BERT Emo - 2.ipynb
    → text / BERT experiments

rev glove.ipynb
    → GloVe text experiments

Early Fusion Diss Audio+Video - PyTorch - 2.ipynb
    → audio + visual fusion

Early Fusion Diss Audio+Video+Text.ipynb
    → text + audio + visual fusion
```

---

## Challenges & Learnings

Multimodal emotion recognition introduces several challenges:

### Class Imbalance

Some emotions occur substantially more often than others, making raw accuracy insufficient for evaluating model quality.

This motivated the use of **balanced accuracy, precision, recall, and confusion matrices**.

### Multimodal Alignment

Text, audio, and visual signals must represent the same portion of a video for meaningful fusion.

Maintaining alignment between modalities is therefore essential.

### Representation Selection

Different modalities require different representation strategies:

```text
Text   → contextual transformer embeddings
Audio  → acoustic features
Visual → facial and behavioral features
```

Combining these heterogeneous representations into a useful shared feature space is a central challenge in multimodal machine learning.

### Fusion Strategy

The project explores **early fusion**, where features from multiple modalities are combined before classification.

More advanced architectures could instead learn cross-modal interactions directly.

---

## Future Improvements

Potential extensions include:

- replacing hand-engineered acoustic and visual representations with learned encoders
- using transformer-based audio and vision models
- applying cross-modal attention
- experimenting with late fusion and hybrid fusion strategies
- addressing class imbalance using weighted or focal losses
- performing hyperparameter optimization
- adding model calibration
- building a unified training and evaluation pipeline
- packaging the preprocessing workflow into reusable Python modules
- evaluating modern multimodal foundation models

A modern version could use an architecture such as:

```text
Text Transformer ───────┐
                        │
Audio Transformer ──────┼──► Cross-Modal Attention ─► Classifier
                        │
Vision Transformer ─────┘
```

---

## Skills Demonstrated

- Multimodal Machine Learning
- Deep Learning
- Natural Language Processing
- Emotion Recognition
- BERT / Transformers
- PyTorch
- Feature Engineering
- Early Fusion
- Multi-Label Classification
- Audio Processing
- Visual Feature Analysis
- Model Evaluation
- Class-Imbalance Analysis
- Data Preprocessing
- Experimental ML Design
