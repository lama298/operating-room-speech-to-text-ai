

\# AI-Powered Operating Room Speech-to-Text \& Clinical Documentation System



An end-to-end AI prototype developed as part of a Master's thesis in Artificial Intelligence.



The system explores how speech recognition and natural language processing can transform spoken operating room information into structured clinical documentation.



\## Overview



The project combines automatic speech recognition, semantic information extraction, structured value parsing, database integration, and report generation within a desktop application.



\*\*Main capabilities\*\*

\- Live microphone recording with voice activity detection (VAD)

\- Speech-to-text transcription using OpenAI Whisper

\- Semantic clinical field detection using Sentence Transformers

\- Hybrid extraction and validation of clinical information

\- MongoDB integration for structured data persistence

\- Automated clinical report generation

\- Tkinter-based desktop interface



\## System Architecture



```text

Microphone / Operating Room Audio

&#x20;               |

&#x20;               v

&#x20;   Voice Activity Detection

&#x20;               |

&#x20;               v

&#x20;     Audio Preprocessing

&#x20;               |

&#x20;               v

&#x20;      Whisper Transcription

&#x20;               |

&#x20;               v

&#x20;   Clinical Information Extraction

&#x20;     | Semantic Field Detection

&#x20;     | Structured Value Parsing

&#x20;     | Validation and Mapping

&#x20;               |

&#x20;               v

&#x20;      Structured Clinical Data

&#x20;               |

&#x20;               v

&#x20;        MongoDB Storage

&#x20;               |

&#x20;               v

&#x20;      Clinical Report Generation

```



\## Technologies



\- Python

\- PyTorch

\- OpenAI Whisper

\- Sentence Transformers

\- Natural Language Processing

\- Silero VAD

\- MongoDB

\- Tkinter



\## Evaluation



The final independent evaluation used 20 previously unseen recordings.



| Metric | Result |

|---|---|

| Word Error Rate (WER) | 4.62% |

| Information Extraction Precision | 100.00% |

| Information Extraction Recall | 86.11% |

| Information Extraction F1-score | 92.54% |

| Value Accuracy | 77.78% |

| Case Accuracy | 70.00% |



These results reflect performance on a limited research evaluation dataset and do not establish clinical readiness.



\## Installation



Clone the repository:



```bash

git clone https://github.com/YOUR\_USERNAME/operating-room-speech-to-text-ai.git

cd operating-room-speech-to-text-ai

```



Create the Conda environment:



```bash

conda env create -f environment.yml

```



Activate the environment using the name defined in `environment.yml`.



A local MongoDB instance may be required for database persistence. Model downloads and microphone permissions may also be needed.



\## Running the Desktop Application



From the project root:



```bash

python -m src.live\_surgery\_demo\_app\_v1

```



This launches the OR Documentation Assistant desktop interface.



\## Privacy and Data Availability



Original audio recordings, research datasets, patient-related test records, and generated clinical reports are intentionally excluded from this public repository.



The repository is intended to demonstrate the software architecture and implementation without distributing private research materials.



\## Limitations



This system is a research prototype and is not intended for clinical decision-making or deployment in real operating rooms without further validation, security review, and regulatory assessment.



\## Author



Master's Thesis in Artificial Intelligence



\*\*Research areas:\*\* Speech Recognition, Natural Language Processing, Information Extraction, Healthcare AI, and End-to-End AI Engineering.



