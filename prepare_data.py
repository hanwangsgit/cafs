"""Export the 30 pinned CelebA-HQ images; fastMRI is obtained separately."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from cafs.experiment import read_config,samples,file_hash,check_hash
    from datasets import load_dataset
    config=read_config('face')['dataset']
    units=samples('face')
    if any((args.output/f"{u['dataset_index']}.png").exists() for u in units):
        parser.error('Destination images already exist; use a new output directory')
    dataset=load_dataset(config['dataset_id'],revision=config['revision'],split='train')
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for unit in units:
        path=args.output/f"{unit['dataset_index']}.png"
        dataset[unit['dataset_index']]['image'].convert('RGB').save(path)
        if 'source_sha256' in unit:
            check_hash(path,unit['source_sha256'])
        rows.append({**unit,'file':path.name,'sha256':file_hash(path)})
    (args.output/'manifest.json').write_text(json.dumps({'dataset':config,'units':rows},indent=2)+'\n')
    print(f'Exported {len(rows)} images from pinned revision {config["revision"]}')


if __name__=='__main__':
    main()
