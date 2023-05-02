import vi
import irl_maxent

#python3 -m canonical_task_generation_sim_exp.run_dispersion_vs_acc_exp --num-experiments=250 --max-complex-action-space-size=16 --max-feature-space-size=5 --weight-samples=125 --num-workers 24 --metric num-task-trajectories --weight-space spherical --max-canonical-action-space-size=8 --num-test-users=100 --num-test-tasks=50
 python3 -m canonical_task_generation_sim_exp.run_dispersion_vs_acc_exp --num-experiments=256 --max-complex-action-space-size=16 --max-feature-space-size=5 --weight-samples=128 --num-workers 24 --metric num-task-trajectories --weight-space spherical --max-canonical-action-space-size=8 --num-test-users=64 --num-test-tasks=32

