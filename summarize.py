"""Aggregate complete runs using the paper's quality and timing scopes."""
import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def summarize(root):
    groups=defaultdict(list)
    seen=set()
    for path in sorted(root.glob('*/record.json')):
        record=json.loads(path.read_text())
        if not record['complete']:
            raise ValueError(f'Incomplete block: {path}')
        for row in record['policies']:
            key=(record['domain'],row['policy'],record['budget'])
            cell=(*key,record['unit']['unit_id'])
            if cell in seen:
                raise ValueError(f'Duplicate unit: {cell}')
            seen.add(cell)
            groups[key].append((record,row))
    if not groups:
        raise ValueError('No result records found')
    quality=[]
    for (domain,policy,budget),values in sorted(groups.items()):
        quality.append([domain,policy,budget,len(values),
                        statistics.mean(v['metrics']['psnr'] for _,v in values),
                        statistics.mean(v['metrics']['ssim'] for _,v in values),
                        statistics.mean(v['metrics']['nmse'] for _,v in values) if domain=='mri' else ''])
    costs=[]
    for domain,policy in sorted({key[:2] for key in groups}):
        values=[pair for key,pairs in groups.items() if key[:2]==(domain,policy) for pair in pairs]
        times=[v['acquisition']['wall_seconds'] for r,v in values
               if r['budget']==.1 and r['gpu']=='NVIDIA A100-SXM4-80GB']
        costs.append([domain,policy,
                      sorted({v['acquisition']['nfe'] for _,v in values}),
                      max(v['acquisition']['peak_memory_bytes'] for _,v in values)/2**30,
                      len(times),statistics.median(times) if times else ''])
    return quality,costs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    parser.add_argument('--efficiency',action='store_true')
    args=parser.parse_args()
    quality,costs=summarize(args.results)
    writer=csv.writer(sys.stdout)
    writer.writerow(['domain','policy','nfe_values','peak_gib','timing_n','median_seconds'] if args.efficiency
                    else ['domain','policy','budget','n','psnr','ssim','nmse'])
    writer.writerows(costs if args.efficiency else quality)


if __name__=='__main__':
    main()
