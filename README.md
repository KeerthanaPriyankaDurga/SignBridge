# SIGN BRIDGE
## Indian Sign Language Recognition System

### Project Goal
Recognize English fingerspelling, digits, and isolated Indian Sign Language word gestures through a webcam.

### Core Features
- English alphabet recognition: A-Z
- Digit recognition: 0-9
- Isolated ISL word-gesture recognition
- Combine recognized letters into words
- Combine recognized words into sentences
- Editable text output
- English/Telugu text-to-speech support
- Real evaluation metrics and confusion matrix

### Planned Datasets
1. English alphabet and number images: existing Mendeley dataset.
2. ISL word videos: Amrita SLR dataset, pending download and audit.
3. Marathi data: preserve separately; exclude from initial training.

### Rules
- Audit labels, duplicates, quality, and license before training.
- Keep train, validation, and test participants separated where possible.
- Never report fabricated evaluation metrics.
- Keep original datasets unchanged.
- Finalize model architecture after inspecting the actual datasets.

### Current Status
- Project folders created.
- Dataset audit started.
- Amrita SLR download pending.
- Training and webcam integration not started.
