"""Run one or all frozen paper image/budget blocks."""
import argparse
import json
import importlib.metadata
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', choices=['face','mri'], required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--student', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--architecture', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--index', type=int, help='0-based sample index; omit for all 30')
    parser.add_argument('--budget', type=float, choices=[.05,.1,.25], help='omit for all three')
    parser.add_argument('--policies', nargs='+', help='omit for all eight paper/one-shot arms')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from cafs.experiment import (read_config, samples, load_models, load_target, acquire, reconstruct,
                                 face_quick_metrics, mri_metrics, file_hash)
    import torch
    config = read_config(args.domain)
    units = samples(args.domain)
    if args.index is not None:
        if not 0 <= args.index < len(units):
            parser.error('--index must be in [0,29]')
        units = [units[args.index]]
    policies = args.policies or config['policies']
    if len(policies) != len(set(policies)) or not set(policies) <= set(config['policies']):
        parser.error('--policies must be a unique subset of the configured policies')
    budgets = [args.budget] if args.budget is not None else config['budgets']
    blocks = [(u,b,args.output/f"{args.domain}-{u['index']:02d}-{b:g}") for u in units for b in budgets]
    if any(p.exists() for _,_,p in blocks):
        parser.error('An output block already exists; use a new output directory')
    device = torch.device(args.device)
    student, teacher, alphas = load_models(config,args.student,args.teacher,args.architecture,device)
    metric = face_quick_metrics if args.domain == 'face' else mri_metrics
    for unit,budget,output in blocks:
        target = load_target(args.domain,unit,args.data_root,device)
        output.mkdir(parents=True)
        rows=[]
        for policy in policies:
            result,streams,compute = acquire(config,target,unit['unit_id'],budget,policy,student,teacher,alphas)
            reconstruction = reconstruct(config,result.final_operator,target,student,teacher,alphas,streams['final_reconstruction'])
            torch.save({'mask':result.final_operator.freq_mask.cpu(), 'reconstruction':reconstruction.cpu()},output/f'{policy}.pt')
            rows.append({'policy':policy,'seeds':streams,'seed_action_ids':result.seed_action_ids,
                         'action_history':result.action_history,'event_costs':result.event_costs,
                         'achieved_m':result.final_operator.m,'acquisition':compute,
                         'metrics':metric(reconstruction,target)})
            record={'domain':args.domain,'unit':unit,'budget':budget,'config':config,'policies':rows,
                    'complete':len(rows)==len(policies),'torch':torch.__version__,
                    'packages':{name:importlib.metadata.version(name) for name in ['torch','torchvision','numpy','diffusers','h5py','Pillow']},
                    'data_sha256':file_hash(args.data_root/(unit['unit_id'] if args.domain=='mri' else str(unit['dataset_index'])+'.png')),
                    'paper_data_hash_verified':'source_sha256' in unit,
                    'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else str(device)}
            temporary=output/'record.json.tmp'
            temporary.write_text(json.dumps(record,indent=2)+'\n')
            temporary.replace(output/'record.json')
            print(unit['unit_id'],budget,policy,rows[-1]['metrics'],flush=True)


if __name__ == '__main__':
    main()
