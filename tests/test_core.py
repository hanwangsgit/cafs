"""Small CPU checks; no checkpoints or datasets required."""
import unittest
from types import SimpleNamespace
import torch
from cafs.operators import FourierOp, PhaseEncodingLineOp
from cafs.reconstruction import cm_reconstruct
from cafs.actions import target_budget, initial_actions, select_exact_cost
from cafs.groups import MeasurementGroup
from cafs.rng import semantic_seed


class CoreChecks(unittest.TestCase):
    def test_projection_schedule_and_budget(self):
        for operator in (FourierOp(torch.eye(16, dtype=torch.bool)),
                         PhaseEncodingLineOp(torch.arange(16) < 4,
                                             calibration_lines=(0, 1), readout_size=16)):
            truth = torch.randn(1, *operator.signal_shape)
            y = operator.measure(truth)
            seen = []
            def model(x, t):
                seen.append(int(t[0]))
                return SimpleNamespace(sample=x * 0.01)
            x = cm_reconstruct(model, torch.linspace(.999, .01, 1000), operator, y,
                               k=5, clamp=None, generator=torch.Generator().manual_seed(3))
            torch.testing.assert_close(operator.measure(x), y, atol=2e-5, rtol=2e-5)
            self.assertEqual(seen, [800, 650, 500, 350, 200])
        self.assertEqual([target_budget('mri', b).target_actions for b in [.05,.1,.25]], [18,37,92])
        self.assertEqual(semantic_seed(0, 'initial_mask', 'face', .1), 3606486858758819322)
        groups = [MeasurementGroup(0, (0,), 3), MeasurementGroup(1, (1,2), 6)]
        self.assertEqual(select_exact_cost(torch.tensor([2., 3.]), groups, remaining_cost=6), (1,))

class PolicyChecks(unittest.TestCase):
    def test_all_selection_adapters_with_toy_prior(self):
        from cafs.acquisition import MeasurementOracle
        from cafs.policies import VariantSpec, run_policy
        from cafs.reconstruction import CMReconstructor
        from cafs.baselines import ADSConfig, ADSDPSTrajectory, DDRMPosteriorSampler, adasense_select, ads_select
        from cafs.experiment import read_config
        class Prior:
            def __call__(self,x,t):
                return SimpleNamespace(sample=x*.01)
        prior=Prior()
        alpha=torch.linspace(.999,.01,1000)
        operator=PhaseEncodingLineOp(torch.arange(16)<4,calibration_lines=(0,1),readout_size=16)
        truth=torch.randn(1,*operator.signal_shape)
        for policy,k in [('cm_spectral_repeat_k1',1),('cm_spectral_repeat_k5',5),
                         ('cm_spectral_repeat_k10',10),('cm_spectral_once_k1',1),
                         ('adasense_matched',0),('ads_matched',0)]:
            adapter=CMReconstructor(unet=prior,alphas_cumprod=alpha,steps=max(1,k),t_start=800,t_end=50,zeta=1.,clamp=None)
            context=MeasurementOracle(target=truth,initial_operator=operator,event_costs=(32,)*5).policy_context(
                policy_id=policy,selector_generators=[torch.Generator().manual_seed(i) for i in range(5)],
                intermediate_reconstructor=adapter)
            if k:
                result=run_policy(context,VariantSpec(policy,'once' if 'once' in policy else 'repeat',k))
                nfe=k if 'once' in policy else 5*k
            elif policy=='adasense_matched':
                result=adasense_select(context,posterior_sampler=DDRMPosteriorSampler(prior,alpha,'cpu'),event_costs=context.event_costs)
                nfe=1000
            else:
                raw=read_config('mri')['matched_baselines']['ads']
                fields=ADSConfig.__dataclass_fields__
                config=ADSConfig(domain='mri',**{k:(tuple(v) if k=='window' else v) for k,v in raw.items() if k in fields})
                trajectory=ADSDPSTrajectory(unet=prior,alphas_cumprod=alpha,config=config,
                    signal_shape=operator.signal_shape,generator=context.selector_generator,device='cpu')
                result=ads_select(context,particles=16,steps=10000,window=config.window,
                    event_costs=context.event_costs,config=config,advance_to_event=trajectory.advance)
                nfe=80032
            self.assertEqual(result.final_operator.m,288)
            self.assertEqual(len(result.action_history),5)
            self.assertEqual(result.accounting.intermediate_reconstruction_nfe+result.accounting.probe_nfe,nfe)


if __name__ == '__main__':
    unittest.main()
