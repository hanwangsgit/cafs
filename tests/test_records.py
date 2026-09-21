"""Check recovered experiment wiring against saved manuscript records."""
import ast
import hashlib
import json
import unittest
from pathlib import Path
import torch
from cafs.experiment import ROOT, initial_operator, seeds, mri_static, samples
from cafs.acquisition import MeasurementOracle


class RecordChecks(unittest.TestCase):
    def test_extracted_definitions(self):
        manifest=json.loads((ROOT/'reproduction/source_map.json').read_text())
        for entry in manifest['symbols']:
            data=(ROOT/'cafs'/entry['file']).read_bytes()
            if 'file_sha256' in entry:
                self.assertEqual(hashlib.sha256(data).hexdigest(),entry['file_sha256'])
            else:
                node=next(n for n in ast.parse(data).body if getattr(n,'name',None)==entry['symbol'])
                digest=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
                self.assertEqual(digest,entry['ast_sha256'],entry['symbol'])

    def test_saved_seeds_masks_and_static_mri(self):
        for domain in ['face','mri']:
            self.assertEqual(len(samples(domain)),30)
        for case in json.loads((ROOT/'reproduction/reference_cases.json').read_text()):
            domain,budget=case['domain'],case['budget']
            initial,costs,_=initial_operator(domain,budget,'cpu')
            for row in case['policies']:
                streams=seeds(domain,case['unit_id'],budget,row['policy_id'])
                for key,value in row['seeds'].items():
                    self.assertEqual(streams[key],value,(domain,key))
                self.assertEqual(initial.selected_group_ids.tolist(),row['seed_action_ids'])
                self.assertEqual(list(costs),row['event_costs'])
                if domain=='mri' and row['policy_id'] in ['uniform','variable_density']:
                    context=MeasurementOracle(target=torch.zeros(1,*initial.signal_shape),initial_operator=initial,
                         event_costs=costs).policy_context(policy_id=row['policy_id'],
                         selector_generator=torch.Generator(),intermediate_reconstructor=lambda *_:None)
                    result=mri_static(context,budget)
                    self.assertEqual([list(x) for x in result.action_history],row['action_history'])


if __name__=='__main__':
    unittest.main()
