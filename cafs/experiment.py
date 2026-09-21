"""Paper settings and a single paired image/budget experiment."""
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import time
import torch

from .actions import face_action_dictionary, initial_actions, target_budget
from .acquisition import MeasurementOracle, run_static_policy
from .baselines import ADSConfig, ADSDPSTrajectory, DDRMPosteriorSampler, adasense_select, ads_select
from .operators import FourierOp, PhaseEncodingLineOp
from .policies import VariantSpec, run_policy, _density_logits, _select
from .reconstruction import CMReconstructor, ddrm_posterior_sample
from .rng import semantic_seed, preserve_global_rng
from .metrics import face_quick_metrics, mri_metrics

ROOT = Path(__file__).resolve().parents[1]


def read_config(domain):
    return json.loads((ROOT / 'configs' / f'{domain}.json').read_text())


def samples(domain):
    return json.loads((ROOT / 'reproduction' / f'{domain}_samples.json').read_text())


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_hash(path, expected):
    if file_hash(path) != expected:
        raise ValueError(f'Checkpoint or data SHA-256 differs: {path}')


def initial_operator(domain, budget, device):
    target = target_budget(domain, budget)
    seed = semantic_seed(0, 'initial_mask', domain, budget)
    selected = initial_actions(domain, target, repeat_seed=seed)
    if domain == 'face':
        mask = torch.zeros(256, 256, dtype=torch.bool, device=device)
        groups = face_action_dictionary()
        for action_id in selected:
            mask.view(-1)[list(groups[action_id].indices)] = True
        operator = FourierOp(mask, n_channels=3, symmetrize=False, requested_m=target.achieved_m)
        pairs, singleton = divmod(target.achieved_m - operator.m, 6)
        costs = [(pairs // 5 + (i < pairs % 5)) * 6 for i in range(5)]
        costs[0] += singleton
    else:
        mask = torch.zeros(368, dtype=torch.bool, device=device)
        mask[list(selected)] = True
        operator = PhaseEncodingLineOp(mask, calibration_lines=(366,367,0,1), requested_m=target.achieved_m)
        lines = (target.achieved_m - operator.m) // 1280
        costs = [(lines // 5 + (i < lines % 5)) * 1280 for i in range(5)]
    return operator, tuple(costs), target


def seeds(domain, unit_id, budget, policy):
    keys = (unit_id, budget) if domain == 'face' else ('mri', unit_id, budget)
    return {
        'selector': [semantic_seed(0, 'selector', *keys, policy, i) for i in range(5)],
        'final_reconstruction': semantic_seed(0, 'final_reconstruction', *keys),
        'initial_mask': semantic_seed(0, 'initial_mask', domain, budget),
        'static_selection': (semantic_seed(0, 'initial_mask', domain, budget, 0, 'post_seed_static_fill')
                             if domain == 'mri' and policy in {'uniform', 'variable_density'} else
                             semantic_seed(0, 'selector', *keys, policy, 0)
                             if domain == 'face' and policy in {'uniform', 'variable_density'} else None),
    }


def mri_static(context, budget):
    """Recovered from publication_v2 static policy and saved MRI action histories."""
    base = (_density_logits(context.operator, 'mri') if context.policy_id == 'variable_density'
            else torch.zeros(context.operator.W, dtype=torch.float64))
    seed = semantic_seed(0, 'initial_mask', 'mri', budget, 0, 'post_seed_static_fill')
    uniforms = torch.rand(base.shape, generator=torch.Generator().manual_seed(seed),
                          dtype=torch.float64).clamp_(1e-12, 1-1e-12)
    scores = base - torch.log(-torch.log(uniforms))
    events, planning = [], context
    for cost in context.event_costs:
        ids = _select(planning, scores, cost)
        events.append(ids)
        planning = replace(planning, operator=planning.operator.add_groups(ids),
                           event_index=planning.event_index + 1)
    return run_static_policy(context, planned_events=events)


def acquire(config, target, unit_id, budget, policy, student, teacher, alphas):
    domain, device = config['domain'], target.device
    operator, costs, budget_target = initial_operator(domain, budget, device)
    streams = seeds(domain, unit_id, budget, policy)
    variant = None
    intermediate = None
    if policy.startswith('cm_spectral_'):
        cadence, k = policy.removeprefix('cm_spectral_').split('_k')
        variant = VariantSpec(policy, cadence, int(k))
        intermediate = CMReconstructor(unet=student, alphas_cumprod=alphas, steps=int(k),
                                       synchronize=lambda: sync(device), **config['acquisition'])
    context = MeasurementOracle(target=target, initial_operator=operator, event_costs=costs).policy_context(
        policy_id=policy, selector_generators=[torch.Generator(device=device).manual_seed(s)
                                              for s in streams['selector']],
        intermediate_reconstructor=intermediate or (lambda *_: None))
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    start = time.perf_counter()
    if policy == 'adasense_matched':
        sampler = DDRMPosteriorSampler(teacher, alphas, device)
        result = adasense_select(context, posterior_sampler=sampler, s=8, ddrm_steps=25, event_costs=costs)
    elif policy == 'ads_matched':
        raw = config['matched_baselines']['ads']
        values = raw.get('settings', raw)
        ads = ADSConfig(domain=domain, **{k:raw[k] for k in ['source_domain','source_config_path','source_config_sha256','source_commit']},
                        **{k:values[k] for k in ['n_particles','sigma','guidance','n_steps','hard_consistency','entropy_sign','aggregation','controlled_adaptation']},
                        window=tuple(values['window']))
        trajectory = ADSDPSTrajectory(unet=teacher, alphas_cumprod=alphas, config=ads,
                                     signal_shape=operator.signal_shape, generator=context.selector_generator,
                                     device=device, clamp=config['acquisition']['clamp'], time_schedule='truncated')
        result = ads_select(context, particles=16, steps=10000, window=ads.window,
                            event_costs=costs, config=ads, advance_to_event=trajectory.advance)
    elif domain == 'mri' and variant is None:
        result = mri_static(context, budget)
    else:
        result = run_policy(context, variant or SimpleNamespace(variant_id=policy))
    sync(device)
    diagnostic = sum(t.diagnostic_seconds for t in intermediate.traces) if intermediate else 0
    elapsed = max(0., time.perf_counter() - start - diagnostic)
    if result.final_operator.m != budget_target.achieved_m:
        raise RuntimeError('Acquisition did not reach the exact budget')
    compute = asdict(result.accounting)
    compute.update(wall_seconds=elapsed, diagnostic_seconds=diagnostic,
                   nfe=result.accounting.forward_sample_evaluations + result.accounting.backward_sample_evaluations,
                   peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0)
    return result, streams, compute


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def reconstruct(config, operator, target, student, teacher, alphas, seed):
    device = target.device
    measurements = operator.measure(target)
    settings = config['final_settings']
    if config['final_reconstructor'] == 'cm':
        adapter = CMReconstructor(unet=student, alphas_cumprod=alphas,
            steps=settings['steps'], t_start=settings['t_start'], t_end=settings['t_end'],
            zeta=settings['zeta'], clamp=settings['clamp'], synchronize=lambda: sync(device))
        value, _ = adapter(operator, measurements, torch.Generator(device=device).manual_seed(seed))
        return value
    with preserve_global_rng((device,)):
        torch.manual_seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(seed)
        values, _ = ddrm_posterior_sample(teacher, alphas, operator, measurements,
            n_samples=1, steps=settings['steps'], t_start=settings['t_start'],
            zeta=settings['zeta'], max_batch=1, device=str(device), clamp=settings['clamp'])
        return values.mean(dim=0, keepdim=True)


def load_models(config, student_path, teacher_path, architecture, device):
    check_hash(student_path, config['checkpoints']['student'])
    check_hash(teacher_path, config['checkpoints']['teacher'])
    if config['domain'] == 'face':
        from diffusers import DDPMPipeline
        if architecture is None:
            raise ValueError('--architecture must point to the local face DDPM pipeline')
        check_hash(Path(architecture) / 'model_index.json', config['architecture']['manifest_sha256'])
        models = []
        for path in (student_path, teacher_path):
            pipe = DDPMPipeline.from_pretrained(architecture, local_files_only=True, torch_dtype=torch.float32).to(device)
            pipe.unet.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            models.append(pipe.unet)
            alphas = pipe.scheduler.alphas_cumprod.to(device)
        student, teacher = models
    else:
        from .mri_prior import build_mri_unet, load_mri_teacher
        student = build_mri_unet()
        student.load_state_dict(torch.load(student_path, map_location='cpu', weights_only=True))
        student.to(device)
        teacher, alphas = load_mri_teacher(str(teacher_path), str(device))
        teacher.enable_gradient_checkpointing()
    for model in (student, teacher):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    shape = (3,256,256) if config['domain'] == 'face' else (2,640,368)
    with torch.inference_mode():
        for model in (student, teacher):
            model(torch.zeros(1,*shape,device=device), torch.tensor([800],device=device))
    sync(device)
    return student, teacher, alphas


def load_target(domain, unit, data_root, device):
    if domain == 'mri':
        from .data import load_mri_slice_deterministic, tensor_sha256
        path = data_root / unit['unit_id']
        check_hash(path, unit['source_sha256'])
        target = load_mri_slice_deterministic(path, unit['slice_index'])
        if target.shape != (2,640,368):
            raise ValueError('Expected a 2 x 640 x 368 single-coil MRI slice')
        if tensor_sha256(target.unsqueeze(0)) != unit['target_sha256']:
            raise ValueError('MRI target tensor SHA-256 differs from the paper binding')
    else:
        from PIL import Image
        import torchvision.transforms as T
        path = data_root / f"{unit['dataset_index']}.png"
        transform = T.Compose([T.Resize((256,256), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
                               T.ToTensor(), T.Lambda(lambda x:x*2-1)])
        with Image.open(path) as image:
            target = transform(image.convert('RGB'))
        manifest = json.loads((data_root / 'manifest.json').read_text())
        if manifest['dataset'] != read_config('face')['dataset']:
            raise ValueError('Face export dataset revision/preprocessing differs')
        row = next(r for r in manifest['units'] if r['unit_id'] == unit['unit_id'])
        check_hash(path, row['sha256'])
        check_hash(path, unit['source_sha256'])
    return target.unsqueeze(0).to(device)
