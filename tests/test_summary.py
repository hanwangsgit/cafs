import json
import tempfile
import unittest
from pathlib import Path
from summarize import summarize


class SummaryChecks(unittest.TestCase):
    def test_scope_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for i in range(2):
                block=root/str(i);block.mkdir()
                record={'complete':True,'domain':'face','budget':.1,'unit':{'unit_id':str(i)},
                        'gpu':'NVIDIA A100-SXM4-80GB' if i==0 else 'cpu',
                        'policies':[{'policy':'uniform','metrics':{'psnr':20+2*i,'ssim':.8},
                                     'acquisition':{'nfe':0,'wall_seconds':10+90*i,'peak_memory_bytes':2**30}}]}
                (block/'record.json').write_text(json.dumps(record))
            quality,costs=summarize(root)
            self.assertEqual(quality[0][3:6],[2,21,.8])
            self.assertEqual(costs[0][-2:],[1,10])
            (root/'1/record.json').write_text((root/'0/record.json').read_text())
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                summarize(root)
