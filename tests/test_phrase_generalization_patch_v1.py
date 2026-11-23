"""Parser and explicit-recovery tests; no model download or ASR required."""
import csv
import importlib.util
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'config' / 'module7'
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

PARSER = load('value_parser', MODULE / 'value_parser.py')
SEGMENTER = load('clause_segmenter_v1', MODULE / 'clause_segmenter_v1.py')
with (MODULE / 'field_sources_v2.csv').open(encoding='utf-8-sig') as f:
    META = {r['field']: r for r in csv.DictReader(f)}

class PhraseTests(unittest.TestCase):
    def value(self, field, text):
        result = PARSER.parse_and_validate(text, META[field])
        return result['converted_value'] if result['accepted'] else None

    def test_generalized_positives(self):
        cases = [
            ('preoperative_diagnosis', 'The diagnosis recorded before the operation was a renal stone.', 'renal stone'),
            ('postoperative_diagnosis', 'The diagnosis confirmed after the procedure is a lipoma.', 'lipoma'),
            ('known_allergies', 'The patient reported an allergy to iodine and adhesive tape.', ['iodine', 'adhesive tape']),
            ('patient_name', 'Record Example Patient as the patient name.', 'Example Patient'),
            ('patient_name', 'Enter ExamplePatient as the patient name.', 'ExamplePatient'),
            ('surgeon_1', 'The primary surgeon for the operation was Dr. Hani Saleh.', 'Hani Saleh'),
            ('anesthesiologist', 'Care was provided by the anesthesiologist, Doctor Ruba Amin.', 'Ruba Amin'),
            ('procedure_site', 'The surgical site is the elbow and the laterality is right.', 'elbow'),
            ('incision_time', 'Skin incision took place at 3.15 in the afternoon.', '15:15'),
            ('surgery_end_time', 'The operation was completed at 4:50 in the afternoon.', '16:50'),
            ('incision_time', 'The incision was made at 12:10 in the morning.', '00:10'),
            ('surgery_end_time', 'The surgery was completed at 12:20 in the afternoon.', '12:20'),
            ('transfer_destination', 'The patient was sent to the surgical ward.', 'floor'),
            ('procedure_name', 'The planned procedure is tendon repair.', 'tendon repair'),
            ('preoperative_diagnosis', 'The preoperative diagnosis is acute appendicitis.', 'acute appendicitis'),
            ('surgeon_1', 'The primary surgeon is Dr. Sami Hatem.', 'Sami Hatem'),
            ('surgeon_1', 'Dr. Sami Hatem is the primary surgeon, Dr. Ruba Amin is the second surgeon.', 'Sami Hatem'),
            ('known_allergies', 'The patient is allergic to penicillin.', ['penicillin']),
            ('surgery_end_time', 'Surgery ended at 10:20.', '10:20'),
        ]
        for field, text, expected in cases:
            with self.subTest(field=field, text=text):
                self.assertEqual(self.value(field, text), expected)

    def test_event_context_and_boundaries(self):
        device = 'During the operation, the suction unit stopped functioning and was replaced.'
        burn = 'An inadvertent thermal burn occurred in the operative field and was treated.'
        for text in [device, burn]:
            self.assertEqual(self.value('unexpected_event', text), [text])
        self.assertEqual(self.value('complications', burn), [burn])
        for field, text in [
            ('complications', device),
            ('complications', 'No accidental skin burn occurred during surgery.'),
            ('unexpected_event', 'A possible accidental burn could occur during surgery.'),
            ('unexpected_event', 'The suction unit stopped drawing fluid in the workshop.'),
            ('known_allergies', 'The patient denies an allergy to iodine.'),
            ('known_allergies', 'The patient reports no allergy to iodine.'),
            ('patient_name', 'Please enter the patient ID.'),
            ('procedure_name', 'Blood loss during this procedure was estimated at 40 milliliters.'),
            ('procedure_description', 'The operation was completed at 6.15 in the afternoon.'),
            ('incision_time', 'The patient entered the operating room at 08:15.'),
            ('surgery_end_time', 'The operation was completed at 25:10.'),
            ('other_destination', 'The patient was transferred to the medical ward.'),
        ]:
            with self.subTest(field=field, text=text):
                self.assertIsNone(self.value(field, text))

    def test_site_segmentation(self):
        clauses = SEGMENTER.ClauseSegmenterV1().segment(
            'The surgical site is the elbow and the laterality is right.')
        self.assertEqual(len(clauses), 2)
        self.assertEqual(self.value('procedure_site', clauses[0]), 'elbow')
        self.assertEqual(self.value('laterality', clauses[1]), 'right')

    def test_explicit_recovery_without_model(self):
        # Stub only the import: this checks the explicit-label method, not
        # embedding accuracy or end-to-end BioLORD performance.
        original = sys.modules.get('semantic_detector_v2')
        stub = types.ModuleType('semantic_detector_v2')
        stub.SemanticFieldDetectorV2 = type('UnusedDetector', (), {})
        sys.modules['semantic_detector_v2'] = stub
        try:
            pipeline_module = load('phrase_patch_pipeline', MODULE / 'clinical_extraction_pipeline_v1.py')
        finally:
            if original is None:
                sys.modules.pop('semantic_detector_v2', None)
            else:
                sys.modules['semantic_detector_v2'] = original
        pipeline = pipeline_module.ClinicalExtractionPipelineV1.__new__(pipeline_module.ClinicalExtractionPipelineV1)
        pipeline.field_metadata = META
        for field, text in [
            ('preoperative_diagnosis', 'The diagnosis identified before surgery is a renal stone.'),
            ('postoperative_diagnosis', 'The diagnosis recorded after surgery was a lipoma.'),
            ('procedure_site', 'The surgical site is the elbow.'),
            ('incision_time', 'Skin incision took place at 2:45 in the afternoon.'),
            ('transfer_destination', 'The patient was transferred to the ward.'),
        ]:
            with self.subTest(field=field):
                results = pipeline.extract_explicit_labeled_results(text)
                accepted = [r for r in results if r['status'] == 'accepted' and r['field'] == field]
                self.assertEqual(len(accepted), 1)
                self.assertEqual(accepted[0]['selection_strategy'], 'explicit_label_parser_confirmed')
        self.assertEqual(pipeline.extract_explicit_labeled_results('The printer needs paper.'), [])

if __name__ == '__main__':
    unittest.main(verbosity=2)

