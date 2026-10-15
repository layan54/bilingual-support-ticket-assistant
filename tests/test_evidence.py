"""Focused regression checks for policy preservation and evidence-selection failures."""
import unittest
from src.pipeline import parse_selection, policy_excerpt, capability_limit, sections

class EvidenceTests(unittest.TestCase):
    def test_invalid_selector_reports_reason(self):
        for raw,error in [('not json','invalid_json'),('{"source":"1"}','invalid_source_type'),('{"source":true}','invalid_source_type'),('{"source":9}','invalid_source_id')]:
            with self.subTest(raw=raw):self.assertEqual(parse_selection(raw,3),(None,error))
        self.assertEqual(parse_selection('```json\n{"source":2}\n```',3),(2,None))
        self.assertEqual(parse_selection('{"source":0}',3),(0,None))

    def test_conditions_preserved_in_both_languages(self):
        kb=sections()
        for section in ['cancellations-0','shipping-3','arabic-1','arabic-2']:
            policy=next(c for c in kb if c['id']==section)
            out=policy_excerpt([policy],policy['lang'])
            self.assertTrue(out['reply'].endswith(policy['text']))
            self.assertEqual(out['evidence_quote'],policy['text'])
            self.assertEqual(out['citations'],[1])
            self.assertFalse(out['needs_human'])

    def test_capability_boundaries_and_general_questions(self):
        for text in ['What is the exact current GPS location of order A-12345?','اعطني الموقع المباشر لمندوب طلبي الحين','Please execute a refund of 500 SAR into my bank account right now.','نفذ استرداد 500 ريال لحسابي الآن']:
            self.assertIsNotNone(capability_limit(text,'ar'))
        for text in ['How do I cancel before shipping?','كيف أقدم طلب استرجاع مبلغ جديد؟','كم يحتاج التتبع عشان يظهر؟']:
            self.assertIsNone(capability_limit(text,'ar'))

if __name__=='__main__':unittest.main()
