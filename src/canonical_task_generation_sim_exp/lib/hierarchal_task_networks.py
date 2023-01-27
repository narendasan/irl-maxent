import numpy as np

def recursive_dependencies(actions, dependencies, result=None):
    """
    Recursively populate all dependencies of actions.
    """

    if result is None:
        result = set()
    for a in actions:
        result.update(dependencies[a])
        recursive_dependencies(dependencies[a], dependencies, result)

    return result

def checkHTN(precondition, all_preconditions):
    """
    Function to check if new action affects the preconditions of other actions.
    Args:
        precondition: preconditions of new action
        all_preconditions: preconditions of all actions in the task

    Returns:
        True if new action can be added to the task without affecting other actions.
    """

    # dependency for each action
    dependencies = [set(np.where(oa_dependency)[0]) for oa_dependency in all_preconditions]

    # dependency for new action
    a_dependency = set(np.where(precondition)[0])
    num_dependencies = len(a_dependency)

    if num_dependencies > 0:

        verification = True

        # check if new action affects the dependency of previous actions
        for oa, oa_dependency in enumerate(dependencies):
            if oa not in a_dependency:
                common_dependencies = a_dependency.intersection(oa_dependency)
                unique_dependencies = a_dependency.symmetric_difference(oa_dependency)
                if common_dependencies and unique_dependencies:
                    # check if the common dependencies are dependent on the unique
                    all_oa_dependencies = recursive_dependencies(common_dependencies, dependencies)
                    if all_oa_dependencies != unique_dependencies:
                        verification = False
                        break
    else:
        # no check required for actions without any dependencies
        verification = True

    return verification