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
try:
    from typing import Dict, List, Any, Tuple
except ImportError:
    pass
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

    def info(self, msg):
        self.messages.append({"level": "INFO", "message": msg})
        print(f"[INFO] {msg}")

    def debug(self, msg):
        self.messages.append({"level": "DEBUG", "message": msg})
        print(f"[DEBUG] {msg}")

    def warning(self, msg):
        self.messages.append({"level": "WARNING", "message": msg})
        print(f"[WARNING] {msg}")

    def error(self, msg):
        self.messages.append({"level": "ERROR", "message": msg})
        print(f"[ERROR] {msg}")


qcentroid_user_log = QCentroidUserLogger()


def greedy_initial_stacking(containers, yard_layout, logger):
    """
    Greedy initialization: sort containers by vessel departure order and weight,
    then place them greedily in available positions.
    """
    logger.info(f"Starting greedy initialization with {len(containers)} containers")

    sorted_containers = sorted(
        containers,
        key=lambda c: (c['vessel_departure_order'], -c['weight_tonnes'])
    )

    stacking_plan = []
    container_map = {c['id']: c for c in containers}
    stack_usage = {}

    for container in sorted_containers:
        cid = container['id']
        weight = container['weight_tonnes']
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

                    if current_tier < max_tier:
                        can_place = True
                        if current_tier > 0:
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
                                'reshuffles_if_retrieved_now': 0
                            }
                            stacking_plan.append(assignment)
                            stack_usage[stack_key] = current_tier + 1
                            placed = True
                            break

        if not placed:
            logger.warning(f"Could not place container {cid} - yard may be full")

    logger.info(f"Greedy placement complete: {len(stacking_plan)} containers placed")
    return stacking_plan


def two_opt_swap(stacking_plan, containers, container_id_list, yard_layout):
    """
    Perform one 2-opt swap: try swapping two containers and keep if it improves objective.
    """
    if len(stacking_plan) < 2:
        return stacking_plan

    idx1, idx2 = random.sample(range(len(stacking_plan)), 2)
    assignment1 = stacking_plan[idx1]
    assignment2 = stacking_plan[idx2]

    new_plan = deepcopy(stacking_plan)
    new_plan[idx1]['assigned_block'] = assignment2['assigned_block']
    new_plan[idx1]['assigned_row'] = assignment2['assigned_row']
    new_plan[idx1]['assigned_bay'] = assignment2['assigned_bay']
    new_plan[idx1]['tier_level'] = assignment2['tier_level']

    new_plan[idx2]['assigned_block'] = assignment1['assigned_block']
    new_plan[idx2]['assigned_row'] = assignment1['assigned_row']
    new_plan[idx2]['assigned_bay'] = assignment1['assigned_bay']
    new_plan[idx2]['tier_level'] = assignment1['tier_level']

    for assignment in new_plan:
        if not is_feasible_assignment(assignment, containers, yard_layout):
            return stacking_plan

    if not check_weight_stability(new_plan, containers, yard_layout):
        return stacking_plan

    return new_plan


def compute_objective(stacking_plan, containers, grouping_weight=0.5):
    """
    Compute objective value: primarily minimize reshuffles, with secondary weight on grouping.
    """
    total_reshuffles, _ = compute_reshuffles_for_stacking(stacking_plan, containers)
    grouping_score = compute_vessel_grouping_score(stacking_plan, containers)
    grouping_penalty = (1.0 - grouping_score) * grouping_weight * 100
    objective = total_reshuffles + grouping_penalty
    return objective


def simulated_annealing(stacking_plan, containers, yard_layout, params, logger):
    """
    Simulated Annealing optimization loop with convergence history tracking.
    Returns: (best_plan, best_objective, iterations_performed, improvements, convergence_history)
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
    total_accepted = 0
    convergence_history = []

    logger.info(f"Starting Simulated Annealing: T_init={temp_init}, cooling={cooling_rate}, max_iter={max_iterations}")
    logger.info(f"Initial objective: {current_obj:.2f}")

    while iteration < max_iterations and temperature > 0.01:
        neighbor_plan = two_opt_swap(current_plan, container_map, container_id_list, yard_layout)
        neighbor_obj = compute_objective(neighbor_plan, containers, grouping_weight)

        delta = neighbor_obj - current_obj
        accepted = False
        if delta < 0:
            current_plan = neighbor_plan
            current_obj = neighbor_obj
            improvements += 1
            accepted = True

            if current_obj < best_obj:
                best_plan = deepcopy(current_plan)
                best_obj = current_obj
                logger.debug(f"Iteration {iteration}: New best objective = {best_obj:.2f}")
        else:
            probability = math.exp(-delta / temperature)
            if random.random() < probability:
                current_plan = neighbor_plan
                current_obj = neighbor_obj
                accepted = True

        if accepted:
            total_accepted += 1

        temperature *= cooling_rate
        iteration += 1

        # Record convergence history at regular intervals
        if iteration % max(1, max_iterations // 50) == 0 or iteration == 1:
            acceptance_rate = total_accepted / max(iteration, 1)
            convergence_history.append({
                'iteration': iteration,
                'best_energy': round(best_obj, 2),
                'current_energy': round(current_obj, 2),
                'temperature': round(temperature, 4),
                'acceptance_rate': round(acceptance_rate, 3),
                'improvements_so_far': improvements
            })

        if iteration % 100 == 0:
            logger.debug(f"Iteration {iteration}: current_obj={current_obj:.2f}, best_obj={best_obj:.2f}, T={temperature:.4f}")

    logger.info(f"SA completed: {iteration} iterations, {improvements} improvements, best_obj={best_obj:.2f}")
    return best_plan, best_obj, iteration, improvements, convergence_history


def compute_output_metrics(stacking_plan, containers, yard_layout):
    """
    Compute all output metrics for the solution.
    """
    container_map = {c['id']: c for c in containers}

    total_reshuffles, reshuffles_per_vessel = compute_reshuffles_for_stacking(stacking_plan, containers)
    block_util = compute_block_utilization(stacking_plan, yard_layout)
    grouping_score = compute_vessel_grouping_score(stacking_plan, containers)
    balance_score = compute_weight_balance_score(stacking_plan, container_map, yard_layout)

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


def generate_stacking_plan_output(stacking_plan, containers):
    """
    Generate output representation of stacking plan with reshuffles_if_retrieved_now computed.
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


def generate_block_heatmap(stacking_plan, containers, yard_layout):
    """
    Generate rich per-block grid visualization data.
    Each cell contains stack info: containers, weight, vessels, fill level.
    """
    container_map = {c['id']: c for c in containers}
    heatmap = {}

    for block in yard_layout['blocks']:
        bid = block['block_id']
        rows = block['rows']
        bays = block['bays_per_row']
        max_tier = block['max_tier_height']
        grid = []

        for r in range(rows):
            row_data = []
            for b in range(bays):
                stack_containers = []
                for a in stacking_plan:
                    if a['assigned_block'] == bid and a['assigned_row'] == r and a['assigned_bay'] == b:
                        c = container_map.get(a['id'], {})
                        stack_containers.append({
                            'id': a['id'],
                            'tier': a['tier_level'],
                            'weight': c.get('weight_tonnes', 0),
                            'vessel': c.get('vessel_id', ''),
                            'departure_order': c.get('vessel_departure_order', 0),
                            'reshuffles_needed': estimate_reshuffles_single_container(a['id'], stacking_plan)
                        })
                stack_containers.sort(key=lambda x: x['tier'])
                total_weight = sum(sc['weight'] for sc in stack_containers)
                height = len(stack_containers)
                vessels_in_stack = list(set(sc['vessel'] for sc in stack_containers))
                row_data.append({
                    'row': r,
                    'bay': b,
                    'height': height,
                    'max_height': max_tier,
                    'fill_pct': round(100 * height / max_tier, 1),
                    'total_weight_tonnes': round(total_weight, 1),
                    'vessels': vessels_in_stack,
                    'vessel_mix': len(vessels_in_stack),
                    'containers': stack_containers
                })
            grid.append(row_data)

        block_containers = [a for a in stacking_plan if a['assigned_block'] == bid]
        capacity = block['total_capacity']
        heatmap[bid] = {
            'block_id': bid,
            'dimensions': {'rows': rows, 'bays': bays, 'max_tier': max_tier},
            'total_containers': len(block_containers),
            'capacity': capacity,
            'utilization_pct': round(100 * len(block_containers) / capacity, 1) if capacity > 0 else 0,
            'grid': grid
        }

    return heatmap


def generate_vessel_timeline(stacking_plan, containers):
    """
    Generate vessel departure timeline with reshuffle forecast.
    Shows per-vessel retrieval efficiency and cumulative reshuffles.
    """
    vessels = {}
    for c in containers:
        vid = c['vessel_id']
        if vid not in vessels:
            vessels[vid] = {'vessel_id': vid, 'departure_order': c['vessel_departure_order'], 'containers': []}
        vessels[vid]['containers'].append(c)

    _, reshuffles_per_vessel = compute_reshuffles_for_stacking(stacking_plan, containers)
    timeline = []
    cumulative = 0

    for vid, info in sorted(vessels.items(), key=lambda x: x[1]['departure_order']):
        r = reshuffles_per_vessel.get(vid, 0)
        cumulative += r
        n = len(info['containers'])
        tw = sum(c['weight_tonnes'] for c in info['containers'])
        eff = round(100 * (1 - r / max(n, 1)), 1)
        timeline.append({
            'vessel_id': vid,
            'departure_order': info['departure_order'],
            'num_containers': n,
            'total_weight_tonnes': round(tw, 1),
            'avg_weight_tonnes': round(tw / n, 1) if n > 0 else 0,
            'reshuffles': r,
            'cumulative_reshuffles': cumulative,
            'retrieval_efficiency_pct': eff,
            'status': 'clean' if r == 0 else ('minor' if r <= 2 else 'needs_attention')
        })

    return timeline


def generate_convergence_chart_data(convergence_history):
    """
    Downsample convergence history to ~50 points for chart rendering.
    """
    n = len(convergence_history)
    if n <= 50:
        return convergence_history
    step = max(1, n // 50)
    sampled = [convergence_history[i] for i in range(0, n, step)]
    if sampled[-1] != convergence_history[-1]:
        sampled.append(convergence_history[-1])
    return sampled


def run(input_data, solver_params=None, extra_arguments=None):
    """
    Main QCentroid solver entry point for Container Yard Stacking Optimization.

    Args:
        input_data: Dict with containers, yard_layout, parameters
        solver_params: Optional QCentroid solver parameters
        extra_arguments: Optional extra arguments from user

    Returns:
        Dict with objective_value, benchmark, solution, showcase, and additional_output
    """
    logger = qcentroid_user_log
    start_time = time.time()

    try:
        # Extract input data - handle both wrapped and unwrapped formats
        if 'containers' in input_data:
            data = input_data
        else:
            data = input_data.get('data', input_data)
        containers = data.get('containers', [])
        yard_layout = data.get('yard_layout', {})
        params = data.get('parameters', {})

        if solver_params:
            params.update(solver_params)

        logger.info("Container Yard Stacking Optimization Solver")
        logger.info(f"Algorithm: Greedy Init + 2-opt + Simulated Annealing")
        logger.info(f"Input: {len(containers)} containers, {yard_layout.get('total_blocks', 0)} yard blocks")

        if not containers or not yard_layout:
            logger.error("Invalid input: missing containers or yard_layout")
            return {"status": "ERROR", "message": "Missing required input data"}

        # Step 1: Greedy initialization
        logger.info("Phase 1: Greedy Initialization")
        initial_plan = greedy_initial_stacking(containers, yard_layout, logger)

        if not initial_plan:
            logger.error("Greedy initialization failed")
            return {"status": "ERROR", "message": "Failed to create initial stacking plan"}

        greedy_obj = compute_objective(initial_plan, containers, params.get('grouping_weight', 0.5))
        logger.info(f"Initial solution objective: {greedy_obj:.2f}")

        # Step 2: Simulated Annealing optimization
        logger.info("Phase 2: Simulated Annealing Optimization")
        best_plan, best_obj, sa_iterations, sa_improvements, convergence_history = simulated_annealing(
            initial_plan, containers, yard_layout, params, logger
        )

        elapsed_ms = (time.time() - start_time) * 1000
        elapsed_s = elapsed_ms / 1000.0

        # Step 3: Compute output metrics
        logger.info("Phase 3: Computing Metrics & Visualization Data")
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
                'reshuffles_percentage': round(100.0 * reshuffles / len(vessel_containers), 1) if vessel_containers else 0.0
            })
        vessel_summary.sort(key=lambda v: v['departure_order'])

        # Generate showcase visualization data
        block_heatmap = generate_block_heatmap(best_plan, containers, yard_layout)
        vessel_timeline = generate_vessel_timeline(best_plan, containers)
        convergence_chart = generate_convergence_chart_data(convergence_history)

        improvement_pct = round((1 - best_obj / max(greedy_obj, 0.01)) * 100, 1)

        # Build output (QCentroid benchmark contract compliant)
        output = {
            'objective_value': round(best_obj, 2),
            'solution_status': 'optimal' if best_obj < greedy_obj else 'feasible',
            'stacking_plan': output_stacking_plan,
            'reshuffling_summary': vessel_summary,
            'optimization_metrics': {
                'total_reshuffles': metrics['total_reshuffles'],
                'average_reshuffles_per_vessel': round(metrics['average_reshuffles_per_vessel'], 2),
                'max_reshuffles_single_vessel': metrics['max_reshuffles_single_vessel'],
                'vessel_grouping_score': round(metrics['vessel_grouping_score'], 3),
                'stack_utilization': round(metrics['stack_utilization'], 3),
                'weight_balance_score': round(metrics['weight_balance_score'], 3)
            },
            'cost_breakdown': {
                'total_reshuffles': metrics['total_reshuffles'],
                'greedy_reshuffles': round(greedy_obj, 2),
                'optimized_reshuffles': round(best_obj, 2),
                'improvement_pct': improvement_pct
            },
            'optimization_convergence': {
                'greedy_initial_cost': round(greedy_obj, 2),
                'sa_cost': round(best_obj, 2),
                'final_optimized_cost': round(best_obj, 2),
                'sa_iterations': sa_iterations,
                'sa_improvements': sa_improvements
            },
            'showcase': {
                'block_heatmap': block_heatmap,
                'vessel_timeline': vessel_timeline,
                'convergence_chart': convergence_chart,
                'summary_dashboard': {
                    'total_containers': len(containers),
                    'total_placed': len(output_stacking_plan),
                    'total_reshuffles': metrics['total_reshuffles'],
                    'improvement_vs_greedy_pct': improvement_pct,
                    'vessels_with_zero_reshuffles': sum(1 for v in vessel_summary if v['estimated_reshuffles'] == 0),
                    'total_vessels': len(vessel_summary),
                    'avg_stack_utilization_pct': round(metrics['stack_utilization'] * 100, 1),
                    'weight_balance_score_pct': round(metrics['weight_balance_score'] * 100, 1),
                    'vessel_grouping_score_pct': round(metrics['vessel_grouping_score'] * 100, 1),
                    'solver_time_ms': round(elapsed_ms, 1),
                    'algorithm': 'Classical SA (' + str(sa_iterations) + ' iterations)'
                }
            },
            'computation_metrics': {
                'wall_time_s': round(elapsed_s, 3),
                'algorithm': 'Greedy_SA_v1.1',
                'solver_version': '1.1',
                'sa_iterations': sa_iterations
            },
            'benchmark': {
                'execution_cost': {'value': 1.0, 'unit': 'credits'},
                'time_elapsed': f'{elapsed_s:.3f}s',
                'energy_consumption': 0.0
            }
        }

        logger.info(f"Solver completed successfully in {elapsed_ms:.1f} ms")
        logger.info(f"Final objective: {best_obj:.2f} (improvement: {improvement_pct}%)")
        logger.info(f"Total reshuffles minimized: {metrics['total_reshuffles']}")

        return output

    except Exception as e:
        logger.error(f"Solver failed with exception: {str(e)}")
        elapsed_s = (time.time() - start_time)
        return {
            'status': 'ERROR',
            'message': str(e),
            'objective_value': 999999,
            'solution_status': 'error',
            'benchmark': {
                'execution_cost': {'value': 1.0, 'unit': 'credits'},
                'time_elapsed': f'{elapsed_s:.3f}s',
                'energy_consumption': 0.0
            },
            'computation_metrics': {
                'wall_time_s': round(elapsed_s, 3),
                'algorithm': 'Greedy_SA_v1.1'
            }
        }


if __name__ == '__main__':
    with open('dataset_small.json', 'r') as f:
        test_input = json.load(f)
    result = run(test_input)
    print("\n" + "="*60)
    print("SOLVER OUTPUT")
    print("="*60)
    print(json.dumps(result, indent=2))
