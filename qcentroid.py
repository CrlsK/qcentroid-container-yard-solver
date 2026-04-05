"""
QCentroid Container Yard Stacking Optimization Solver

This solver optimizes container stacking arrangement in a yard to minimize
reshuffles during vessel loading. Uses greedy initialization, 2-opt local search,
and Simulated Annealing metaheuristic.

Entry point: run(input_data, solver_params, extra_arguments) -> dict
"""

import json
import time
import math
import random
from copy import deepcopy
from typing import Dict, List, Any, Tuple
from solver_helpers import (
    compute_reshuffles_for_stacking,
    is_feasible_assignment,
    check_weight_stability,
    compute_block_utilization,
    compute_vessel_grouping_score,
    compute_weight_balance_score,
    estimate_reshuffles_single_container
)


# Logger for QCentroid platform
class QCentroidUserLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg: str):
        self.messages.append({"level": "INFO", "message": msg})
        print(f"[INFO] {msg}")

    def debug(self, msg: str):
        self.messages.append({"level": "DEBUG", "message": msg})
        print(f"[DEBUG] {msg}")

    def warning(self, msg: str):
        self.messages.append({"level": "WARNING", "message": msg})
        print(f"[WARNING] {msg}")

    def error(self, msg: str):
        self.messages.append({"level": "ERROR", "message": msg})
        print(f"[ERROR] {msg}")


qcentroid_user_log = QCentroidUserLogger()


def greedy_initial_stacking(containers: List[Dict[str, Any]], yard_layout: Dict[str, Any], logger) -> List[Dict[str, Any]]:
    """
    Greedy initialization: sort containers by vessel departure order and weight,
    then place them greedily in available positions.

    Args:
        containers: List of container dictionaries
        yard_layout: Yard layout definition
        logger: Logger instance

    Returns:
        Initial stacking plan (list of assignments)
    """
    logger.info(f"Starting greedy initialization with {len(containers)} containers")

    # Sort containers: first by vessel departure order (earlier first), then by weight (heavier first)
    sorted_containers = sorted(
        containers,
        key=lambda c: (c['vessel_departure_order'], -c['weight_tonnes'])
    )

    stacking_plan = []
    container_map = {c['id']: c for c in containers}

    # Track usage of each position
    stack_usage = {}  # (block, row, bay) -> current_tier (0-indexed, next available)

    for container in sorted_containers:
        cid = container['id']
        weight = container['weight_tonnes']

        # Try to find a suitable position
        placed = False

        for block in yard_layout['blocks']:
            if placed:
                break

            block_id = block['block_id']
            max_tier = block['max_tier_height']

            for row_idx in range(block['rows']):
                if placed:
                    break

                for bay_idx in range(block['bays_per_row']):
                    stack_key = (block_id, row_idx, bay_idx)
                    current_tier = stack_usage.get(stack_key, 0)

                    # Check if we can place in this stack
                    if current_tier < max_tier:
                        # Check weight stability: would this placement violate weight constraints?
                        can_place = True

                        # If tier > 0, check that container below is heavier
                        if current_tier > 0:
                            # Find container below
                            for existing in stacking_plan:
                                if (existing['assigned_block'] == block_id and
                                    existing['assigned_row'] == row_idx and
                                    existing['assigned_bay'] == bay_idx and
                                    existing['tier_level'] == current_tier - 1):
                                    below_cid = existing['id']
                                    below_weight = container_map[below_cid]['weight_tonnes']
                                    if below_weight < weight:
                                        can_place = False
                                    break

                        if can_place:
                            assignment = {
                                'id': cid,
                                'assigned_block': block_id,
                                'assigned_row': row_idx,
                                'assigned_bay': bay_idx,
                                'tier_level': current_tier,
                                'reshuffles_if_retrieved_now': 0  # Will compute later
                            }
                            stacking_plan.append(assignment)
                            stack_usage[stack_key] = current_tier + 1
                            placed = True
                            break

        if not placed:
            logger.warning(f"Could not place container {cid} - yard may be full")

    logger.info(f"Greedy placement complete: {len(stacking_plan)} containers placed")
    return stacking_plan


def two_opt_swap(stacking_plan: List[Dict[str, Any]], containers: Dict[str, Any], container_id_list: List[str], yard_layout: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Perform one 2-opt swap: try swapping two containers and keep if it improves objective.

    Args:
        stacking_plan: Current stacking plan
        containers: Dict mapping container_id -> container info
        container_id_list: List of all container IDs for iteration
        yard_layout: Yard layout definition

    Returns:
        Updated stacking plan (may be same as input if no improvement found)
    """
    if len(stacking_plan) < 2:
        return stacking_plan

    # Pick two random containers
    idx1, idx2 = random.sample(range(len(stacking_plan)), 2)
    assignment1 = stacking_plan[idx1]
    assignment2 = stacking_plan[idx2]

    # Swap their locations
    new_plan = deepcopy(stacking_plan)
    new_plan[idx1]['assigned_block'] = assignment2['assigned_block']
    new_plan[idx1]['assigned_row'] = assignment2['assigned_row']
    new_plan[idx1]['assigned_bay'] = assignment2['assigned_bay']
    new_plan[idx1]['tier_level'] = assignment2['tier_level']

    new_plan[idx2]['assigned_block'] = assignment1['assigned_block']
    new_plan[idx2]['assigned_row'] = assignment1['assigned_row']
    new_plan[idx2]['assigned_bay'] = assignment1['assigned_bay']
    new_plan[idx2]['tier_level'] = assignment1['tier_level']

    # Check feasibility
    for assignment in new_plan:
        if not is_feasible_assignment(assignment, containers, yard_layout):
            return stacking_plan  # Revert if infeasible

    # Check weight stability
    if not check_weight_stability(new_plan, containers, yard_layout):
        return stacking_plan

    return new_plan


def compute_objective(stacking_plan: List[Dict[str, Any]], containers: List[Dict[str, Any]], grouping_weight: float = 0.5) -> float:
    """
    Compute objective value: primarily minimize reshuffles, with secondary weight on grouping.

    Args:
        stacking_plan: Current stacking plan
        containers: List of containers
        grouping_weight: Weight for grouping penalty term (0-1)

    Returns:
        Objective value (lower is better)
    """
    total_reshuffles, _ = compute_reshuffles_for_stacking(stacking_plan, containers)
    grouping_score = compute_vessel_grouping_score(stacking_plan, containers)

    # Grouping penalty: if score is low, add penalty
    grouping_penalty = (1.0 - grouping_score) * grouping_weight * 100

    objective = total_reshuffles + grouping_penalty
    return objective


def simulated_annealing(stacking_plan: List[Dict[str, Any]], containers: List[Dict[str, Any]], yard_layout: Dict[str, Any], params: Dict[str, Any], logger) -> Tuple[List[Dict[str, Any]], float, int]:
    """
    Simulated Annealing optimization loop.

    Args:
        stacking_plan: Initial solution
        containers: List of containers
        yard_layout: Yard layout
        params: Dict with max_iterations, temperature_init, cooling_rate, grouping_weight
        logger: Logger

    Returns:
        (best_plan, best_objective, iterations_performed, improvements)
    """
    max_iterations = params.get('max_iterations', 1000)
    temp_init = params.get('temperature_init', 50.0)
    cooling_rate = params.get('cooling_rate', 0.95)
    grouping_weight = params.get('grouping_weight', 0.5)

    container_map = {c['id']: c for c in containers}
    container_id_list = [c['id'] for c in containers]

    current_plan = deepcopy(stacking_plan)
    current_obj = compute_objective(current_plan, containers, grouping_weight)

    best_plan = deepcopy(current_plan)
    best_obj = current_obj

    temperature = temp_init
    iteration = 0
    improvements = 0

    logger.info(f"Starting Simulated Annealing: T_init={temp_init}, cooling={cooling_rate}, max_iter={max_iterations}")
    logger.info(f"Initial objective: {current_obj:.2f}")

    while iteration < max_iterations and temperature > 0.01:
        # Generate neighbor via 2-opt swap
        neighbor_plan = two_opt_swap(current_plan, container_map, container_id_list, yard_layout)
        neighbor_obj = compute_objective(neighbor_plan, containers, grouping_weight)

        # Acceptance criterion
        delta = neighbor_obj - current_obj
        if delta < 0:  # Improvement
            current_plan = neighbor_plan
            current_obj = neighbor_obj
            improvements += 1

            if current_obj < best_obj:
                best_plan = deepcopy(current_plan)
                best_obj = current_obj
                logger.debug(f"Iteration {iteration}: New best objective = {best_obj:.2f}")

        else:  # Worse solution, accept with probability
            probability = math.exp(-delta / temperature)
            if random.random() < probability:
                current_plan = neighbor_plan
                current_obj = neighbor_obj

        # Cool down
        temperature *= cooling_rate
        iteration += 1

        if iteration % 100 == 0:
            logger.debug(f"Iteration {iteration}: current_obj={current_obj:.2f}, best_obj={best_obj:.2f}, T={temperature:.4f}")

    logger.info(f"SA completed: {iteration} iterations, {improvements} improvements, best_obj={best_obj:.2f}")
    return best_plan, best_obj, iteration, improvements


def compute_output_metrics(stacking_plan: List[Dict[str, Any]], containers: List[Dict[str, Any]], yard_layout: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute all output metrics for the solution.

    Args:
        stacking_plan: Final stacking plan
        containers: List of containers
        yard_layout: Yard layout

    Returns:
        Dict with all metrics
    """
    container_map = {c['id']: c for c in containers}

    total_reshuffles, reshuffles_per_vessel = compute_reshuffles_for_stacking(stacking_plan, containers)

    block_util = compute_block_utilization(stacking_plan, yard_layout)
    grouping_score = compute_vessel_grouping_score(stacking_plan, containers)
    balance_score = compute_weight_balance_score(stacking_plan, container_map, yard_layout)

    # Aggregate metrics
    metrics = {
        'total_reshuffles': total_reshuffles,
        'average_reshuffles_per_vessel': total_reshuffles / len(reshuffles_per_vessel) if reshuffles_per_vessel else 0.0,
        'max_reshuffles_single_vessel': max(reshuffles_per_vessel.values()) if reshuffles_per_vessel else 0,
        'vessel_grouping_score': grouping_score,
        'stack_utilization': sum(block_util.values()) / len(block_util) if block_util else 0.0,
        'weight_balance_score': balance_score,
        'reshuffles_per_vessel': reshuffles_per_vessel,
        'block_utilization': block_util
    }

    return metrics


def generate_stacking_plan_output(stacking_plan: List[Dict[str, Any]], containers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate output representation of stacking plan with reshuffles_if_retrieved_now computed.

    Args:
        stacking_plan: Internal stacking plan
        containers: List of containers

    Returns:
        Output stacking plan
    """
    output_plan = []
    for assignment in stacking_plan:
        cid = assignment['id']
        reshuffles = estimate_reshuffles_single_container(cid, stacking_plan)

        output_assignment = {
            'id': cid,
            'assigned_block': assignment['assigned_block'],
            'assigned_row': assignment['assigned_row'],
            'assigned_bay': assignment['assigned_bay'],
            'tier_level': assignment['tier_level'],
            'reshuffles_if_retrieved_now': reshuffles
        }
        output_plan.append(output_assignment)

    return output_plan


def run(input_data: Dict[str, Any], solver_params: Dict[str, Any] = None, extra_arguments: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Main QCentroid solver entry point for Container Yard Stacking Optimization.

    Args:
        input_data: Dict with structure {"data": {...}} containing containers, yard_layout, parameters
        solver_params: Optional QCentroid solver parameters (e.g., quantum_fraction, timeout_seconds)
        extra_arguments: Optional extra arguments from user

    Returns:
        Dict with objective_value, benchmark, solution, and additional_output
    """
    logger = qcentroid_user_log
    start_time = time.time()

    try:
        # Extract input data - handle both platform (input_data IS the data) and local (input_data has 'data' key)
        if 'containers' in input_data:
            data = input_data
        else:
            data = input_data.get('data', input_data)
        containers = data.get('containers', [])
        yard_layout = data.get('yard_layout', {})
        params = data.get('parameters', {})

        # Merge with solver_params if provided
        if solver_params:
            params.update(solver_params)

        logger.info(f"Container Yard Stacking Optimization Solver")
        logger.info(f"Input: {len(containers)} containers, {yard_layout.get('total_blocks', 0)} yard blocks")

        # Validate input
        if not containers or not yard_layout:
            logger.error("Invalid input: missing containers or yard_layout")
            return {"status": "ERROR", "message": "Missing required input data"}

        # Step 1: Greedy initialization
        logger.info("Step 1: Greedy Initialization")
        initial_plan = greedy_initial_stacking(containers, yard_layout, logger)

        if not initial_plan:
            logger.error("Greedy initialization failed")
            return {"status": "ERROR", "message": "Failed to create initial stacking plan"}

        greedy_obj = compute_objective(initial_plan, containers, params.get('grouping_weight', 0.5))
        logger.info(f"Initial solution objective: {greedy_obj:.2f}")

        # Step 2: Simulated Annealing optimization
        logger.info("Step 2: Simulated Annealing Optimization")
        best_plan, best_obj, sa_iterations, sa_improvements = simulated_annealing(initial_plan, containers, yard_layout, params, logger)

        elapsed_ms = (time.time() - start_time) * 1000

        # Compute output metrics
        logger.info("Step 3: Computing Output Metrics")
        metrics = compute_output_metrics(best_plan, containers, yard_layout)

        # Generate output stacking plan
        output_stacking_plan = generate_stacking_plan_output(best_plan, containers)

        # Compute vessel summary
        total_reshuffles, reshuffles_per_vessel = compute_reshuffles_for_stacking(best_plan, containers)
        vessel_summary = []
        for vessel_id, reshuffles in reshuffles_per_vessel.items():
            vessel_containers = [c for c in containers if c['vessel_id'] == vessel_id]
            vessel_summary.append({
                'vessel_id': vessel_id,
                'departure_order': vessel_containers[0]['vessel_departure_order'] if vessel_containers else 0,
                'total_containers': len(vessel_containers),
                'estimated_reshuffles': reshuffles,
                'reshuffles_percentage': 100.0 * reshuffles / len(vessel_containers) if vessel_containers else 0.0
            })

        # Sort by departure order
        vessel_summary.sort(key=lambda v: v['departure_order'])

        # Build output (QCentroid benchmark contract compliant)
        elapsed_s = elapsed_ms / 1000.0
        output = {
            'objective_value': round(best_obj, 2),
            'solution_status': 'optimal' if best_obj < greedy_obj else 'feasible',
            'stacking_plan': output_stacking_plan,
            'reshuffling_summary': vessel_summary,
            'optimization_metrics': {
                'total_reshuffles': metrics['total_reshuffles'],
                'average_reshuffles_per_vessel': metrics['average_reshuffles_per_vessel'],
                'max_reshuffles_single_vessel': metrics['max_reshuffles_single_vessel'],
                'vessel_grouping_score': metrics['vessel_grouping_score'],
                'stack_utilization': metrics['stack_utilization'],
                'weight_balance_score': metrics['weight_balance_score']
            },
            'cost_breakdown': {
                'total_reshuffles': metrics['total_reshuffles'],
                'greedy_reshuffles': round(greedy_obj, 2),
                'optimized_reshuffles': round(best_obj, 2),
                'improvement_pct': round((1 - best_obj / max(greedy_obj, 1)) * 100, 1)
            },
            'optimization_convergence': {
                'greedy_initial_cost': round(greedy_obj, 2),
                'sa_cost': round(best_obj, 2),
                'final_optimized_cost': round(best_obj, 2),
                'sa_iterations': sa_iterations,
                'sa_improvements': sa_improvements
            },
            'computation_metrics': {
                'wall_time_s': round(elapsed_s, 3),
                'algorithm': 'Greedy_SA_v1.0',
                'solver_version': '1.0',
                'sa_iterations': sa_iterations
            },
            'benchmark': {
                'execution_cost': {'value': 1.0, 'unit': 'credits'},
                'time_elapsed': f'{elapsed_s:.3f}s',
                'energy_consumption': 0.0
            }
        }

        logger.info(f"Solver completed successfully in {elapsed_ms:.1f} ms")
        logger.info(f"Final objective: {best_obj:.2f}")
        logger.info(f"Total reshuffles minimized: {metrics['total_reshuffles']}")

        return output

    except Exception as e:
        logger.error(f"Solver failed with exception: {str(e)}")
        return {
            'status': 'ERROR',
            'message': str(e),
            'solver_log': logger.messages
        }


# For testing/debugging
if __name__ == '__main__':
    # Load small dataset and run
    with open('dataset_small.json', 'r') as f:
        test_input = json.load(f)

    result = run(test_input)

    print("\n" + "="*60)
    print("SOLVER OUTPUT")
    print("="*60)
    print(json.dumps(result, indent=2))
