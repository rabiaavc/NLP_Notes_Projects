"""Offline regression tests; require no Ollama server or model downloads."""
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.dataset import load_data, gold_for
from src.evaluation import score_entities, align_generated_entities, normalize_label
from src.models import build_schema, build_prompt
from scripts.run_experiments import assemble

class TestDataset(unittest.TestCase):
    def test_total_and_spans(self):
        a=load_data('english_test.jsonl'); b=load_data('turkish_supplementary.jsonl');d=load_data('development.jsonl')
        self.assertEqual((len(a),len(b),len(d)),(50,10,7))
        self.assertEqual(len({r['id'] for r in a+b+d}),67)
        self.assertTrue(all(len(r['text'].split())>=50 for r in a))
        self.assertTrue(all(len(r['text'].split())>=40 for r in b))

    def test_gold_extra_labels(self):
        r=next(r for r in load_data('english_test.jsonl') if r['group']=='extended')
        self.assertEqual(len(gold_for(r,['JOB_TITLE'])),1)
        self.assertEqual(len(gold_for(r,['DATE'])),1)
        self.assertFalse(any(e['label']=='JOB_TITLE' for e in gold_for(r)))

class TestEvaluation(unittest.TestCase):
    def test_exact_match_and_hallucination(self):
        src='Sarah met Microsoft in London.'
        gen=[{'text':'Sarah','label':'PERSON'},{'text':'Paris','label':'LOC'},{'text':'Microsoft','label':'ORG'}]
        spans,hallucinations=align_generated_entities(src,gen,['PERSON','ORG','LOC'])
        self.assertEqual(len(spans),2)
        self.assertEqual(len(hallucinations),1)
        self.assertEqual(hallucinations[0]['reason'],'not_found_or_duplicate')
        self.assertEqual(normalize_label('GPE'),'LOC')
        gold=[{'start':0,'end':5,'label':'PERSON'},{'start':10,'end':19,'label':'ORG'}]
        self.assertAlmostEqual(score_entities(gold,spans)['f1'],1.0)
        row={'id':'X','group':'demo','language':'en','entities':gold,'extended_entities':[]}
        result={'entities':spans,'latency_sec':0.01,'raw_json_valid':True,'clean_json_valid':True,'schema_valid':True,
                'ungrounded':hallucinations,'prompt_tokens':1,'generation_tokens':2,'raw_text':'{}'}
        assembled=assemble(row,'llm',result,['PERSON','ORG','LOC'],'demo')
        self.assertEqual((assembled['tp'],assembled['fp'],assembled['fn']),(2,1,0))

    def test_schema(self):
        schema=build_schema(['PERSON','ORG'])
        self.assertEqual(schema['properties']['entities']['items']['properties']['label']['enum'],['PERSON','ORG'])
        text=build_prompt('Sarah works at Google.',['PERSON','ORG'],'en')
        self.assertIn('exact surface substrings',text)

    def test_repeated_mentions(self):
        src='Ava met Ava.'
        spans,errors=align_generated_entities(src,[{'text':'Ava','label':'PERSON'},{'text':'Ava','label':'PERSON'}],['PERSON'])
        self.assertEqual(len(spans),2)
        self.assertEqual(len(errors),0)
        self.assertEqual([(s['start'],s['end']) for s in spans],[(0,3),(8,11)])

if __name__=='__main__': unittest.main()
