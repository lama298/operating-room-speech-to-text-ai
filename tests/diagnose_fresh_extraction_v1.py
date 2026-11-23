"""Save actual semantic rankings and extraction decisions without rerunning ASR."""
import csv
import json
from pathlib import Path

def main():
    from config.module7.clinical_extraction_pipeline_v1 import ClinicalExtractionPipelineV1
    root = Path(__file__).resolve().parents[1]
    path = root / 'outputs/fresh_blind_test_evaluation/fresh_blind_test_results.csv'
    pipeline = ClinicalExtractionPipelineV1()
    records = []
    with path.open(encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            transcript = row['predicted_transcript']
            clauses = pipeline.segmenter.segment(transcript)
            records.append({
                'filename': row['filename'],
                'transcript': transcript,
                'clauses': [dict(clause=c, semantic=pipeline.detector.predict(c, top_k=10)) for c in clauses],
                'extractions': pipeline.process_transcript(transcript, top_k=3),
            })
    output = root / 'outputs/fresh_blind_diagnostics/extraction_trace.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Saved semantic rankings and extraction decisions:', output)

if __name__ == '__main__':
    main()
