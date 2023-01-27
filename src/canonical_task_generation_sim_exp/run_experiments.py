import subprocess
import argparse
from rich.progress import track
from lib.arguments import parser
import simulate_user_demos
import irl
import evaluate_results


def main(args):
    generate_canonical_task_archive(args)
    generate_complex_task_archive(args)

    for f in range(3, args.max_feature_space_size):
        for canonical_as in range(2, args.max_canonical_action_space_size):
            for complex_as in range(2, args.max_complex_action_space_size):


                canonical_demo, complex_demo = simulate_user_demos.sim(canonical_task=canonical_task, complex_task=complex_task)
                irl.learn(canonical_task=canonical_task,
                    canonical_demo=canonical_demo,
                    complex_task=complex_task,
                    complex_demo=complex_demo)

                evaluate_results.evaluate_acc()




    return


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)