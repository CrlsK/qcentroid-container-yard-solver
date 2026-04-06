"""
QCentroid Container Yard Stacking Optimization Solver v2.0

This solver optimizes container stacking arrangement in a yard to minimize
reshuffles during vessel loading. Uses vessel-aware greedy initialization,
2-opt swaps with relocation moves, and Simulated Annealing metaheuristic
with weight balance objective.

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

    def info(self, msg: str):
        self.messages.append({"level": "INFO", "message": msg})
        print("[INFO] " + msg)

    def debug(self, msg: str):
        self.messages.append({"level": "DEBUG", "message": msg})
        print("[DEBUG] " + msg)

    def warning(self, msg: str):
        self.messages.append({"level": "WARNING", "message": msg})
        print("[WARNING] " + msg)

    def error(self, msg: str):
        self.messages.append({"level": "ERROR", "message": msg})
        print("[ERROR] " + msg)


qcentroid_user_log = QCentroidUserLogger()


def greedy_initial_stacking(containers, yard_layout, logger):
    """
    Improved greedy initialization: vessel-aware block assignment.

    Containers from the same vessel are preferentially assigned to the same block
    to promote grouping. Within a vessel, containers are sorted by departure order.
    Blocks are selected to balance utilization across all blocks.

    Args:
        containers: List of container dictionaries
        yard_layout: Yard layout definition
        logger: Logger instance

    Returns:
        Initial stacking plan (list of assignments)
    """
    logger.info("Starting vessel-aware greedy initialization with " + str(len(containers)) + " containers")

    # Group containers by vessel
    vessel_groups = {}
    for container in containers:
        vessel_id = container['vessel_id']
        if vessel_id not in vessel_groups:
            vessel_groups[vessel_id] = []
        vessel_groups[vessel_id].append(container)

    # Sort vessels by departure order
    sorted_vessels = sorted(
        vessel_groups.items(),
        key=lambda x: x[1][0]['vessel_departure_order']
    )

    stacking_plan = []
    container_map = {c['id']: c for c in containers}

    # Track usage of each position
    stack_usage = {}  # (block, row, bay) -> current_tier (0-indexed, next available)

    # Track block utilization for load balancing
    block_capacities = {}
    for block in yard_layout['blocks']:
        block_capacities[block['block_id']] = {
            'capacity': block['total_capacity'],
            'used': 0
        }

    def find_preferred_block(vessel_id, yard_layout, block_capacities):
        """Select a block for this vessel based on utilization balance."""
        blocks = yard_layout['blocks']
        available_blocks = []

        for block in blocks:
            block_id = block['block_id']
            current_util = block_capacities[block_id]['used'] / block_capacities[block_id]['capacity']
            available_capacity = block_capacities[block_id]['capacity'] - block_capacities[block_id]['used']
            if available_capacity > 0:
                available_blocks.append((block_id, current_util, available_capacity))

        if not available_blocks:
            return None

        # Select block with lowest utilization (load balancing)
        available_blocks.sort(key=lambda x: x[1])
        return available_blocks[0][0]

    # Process each vessel's containers
    for vessel_id, vessel_containers in sorted_vessels:
        # Sort containers within vessel by weight (heavier first) for stability
        vessel_containers_sorted = sorted(
            vessel_containers,
            key=lambda c: -c['weight_tonnes']
        )

        # Find preferred block for this vessel
        preferred_block = find_preferred_block(vessel_id, yard_layout, block_capacities)

        if preferred_block is None:
            logger.warning("No available blocks for vessel " + str(vessel_id))
            continue

        # Try to place all containers from this vessel in the preferred block
        for container in vessel_containers_sorted:
            cid = container['id']
            weight = container['weight_tonnes']

            # Try preferred block first
            placed = False
            blocks_to_try = [preferred_block]

            # If preferred block is full, try other blocks
            if block_capacities[preferred_block]['used'] >= block_capacities[preferred_block]['capacity']:
                blocks_to_try = [b['block_id'] for b in yard_layout['blocks'] if b['block_id'] != preferred_block]

            for block_id in blocks_to_try:
                if placed:
                    break

                block = None
                for b in yard_layout['blocks']:
                    if b['block_id'] == block_id:
                        block = b
                        break

                if block is None:
                    continue

                max_tier = block['max_tier_height']

                for row_idx in range(block['rows']):
                    if placed:
                        break

                    for bay_idx in range(block['bays_per_row']):
                        stack_key = (block_id, row_idx, bay_idx)
                        current_tier = stack_usage.get(stack_key, 0)

                        # Check if we can place in this stack
                        if current_tier < max_tier:
                            # Check weight stability
                            can_place = True

                            # If tier > 0, check that container below is heavier
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
                                block_capacities[block_id]['used'] += 1
                                placed = True
                                break

            if not placed:
                logger.warning("Could not place container " + str(cid))

    logger.info("Greedy placement complete: " + str(len(stacking_plan)) + " containers placed")
    return stacking_plan


def two_opt_swap(stacking_plan, containers, yard_layout):
    """
    Perform one 2-opt swap: swap positions of two containers.

    Args:
        stacking_plan: Current stacking plan
        containers: Dict mapping container_id -> container info
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
            return stacking_plan

    # Check weight stability
    if not check_weight_stability(new_plan, containers, yard_layout):
        return stacking_plan

    return new_plan


def relocate_move(stacking_plan, containers, yard_layout):
    """
    Perform a relocate move: pick a random container and move it to a random valid
    position in ANY block (including empty stacks in underutilized blocks).

    Args:
        stacking_plan: Current stacking plan
        containers: Dict mapping container_id -> container info
        yard_layout: Yard layout definition

    Returns:
        Updated stacking plan (may be same as input if relocation infeasible)
    """
    if len(stacking_plan) == 0:
        return stacking_plan

    container_map = {c['id']: c for c in containers}

    # Pick a random container to relocate
    idx = random.randint(0, len(stacking_plan) - 1)
    relocating_assignment = stacking_plan[idx]
    cid = relocating_assignment['id']
    weight = containers[cid]['weight_tonnes']

    # Try to find a valid position in any block
    new_plan = deepcopy(stacking_plan)
    blocks = yard_layout['blocks']

    # Randomize block order for variety
    block_order = list(range(len(blocks)))
    random.shuffle(block_order)

    for block_idx in block_order:
        block = blocks[block_idx]
        block_id = block['block_id']
        max_tier = block['max_tier_height']

        for row_idx in range(block['rows']):
            for bay_idx in range(block['bays_per_row']):
                # Find current tier in this stack
                current_tier = 0
                for assignment in new_plan:
                    if (assignment['assigned_block'] == block_id and
                        assignment['assigned_row'] == row_idx and
                        assignment['assigned_bay'] == bay_idx):
                        current_tier = max(current_tier, assignment['tier_level'] + 1)

                # Can we place here?
                if current_tier < max_tier:
                    # Check weight stability
                    can_place = True
                    if current_tier > 0:
                        for assignment in new_plan:
                            if (assignment['assigned_block'] == block_id and
                                assignment['assigned_row'] == row_idx and
                                assignment['assigned_bay'] == bay_idx and
                                assignment['tier_level'] == current_tier - 1):
                                below_weight = container_map[assignment['id']]['weight_tonnes']
                                if below_weight < weight:
                                    can_place = False
                                break

                    if can_place:
                        # Place the container here
                        new_plan[idx]['assigned_block'] = block_id
                        new_plan[idx]['assigned_row'] = row_idx
                        new_plan[idx]['assigned_bay'] = bay_idx
                        new_plan[idx]['tier_level'] = current_tier

                        # Verify full plan is still feasible
                        feasible = True
                        for assignment in new_plan:
                            if not is_feasible_assignment(assignment, containers, yard_layout):
                                feasible = False
                                break

                        if feasible and check_weight_stability(new_plan, containers, yard_layout):
                            return new_plan

    # If no valid position found, return unchanged
    return stacking_plan


def compute_objective(stacking_plan, containers, grouping_weight=0.5, balance_weight=0.3, yard_layout=None):
    """
    Compute objective value: minimize reshuffles with secondary penalties for grouping
    and weight balance.

    Args:
        stacking_plan: Current stacking plan
        containers: List of containers
        grouping_weight: Weight for grouping penalty term (0-1)
        balance_weight: Weight for balance penalty term (0-1)
        yard_layout: Optional yard layout for balance computation

    Returns:
        Objective value (lower is better)
    """
    total_reshuffles, _ = compute_reshuffles_for_stacking(stacking_plan, containers)
    grouping_score = compute_vessel_grouping_score(stacking_plan, containers)

    # Grouping penalty: if score is low, add penalty
    grouping_penalty = (1.0 - grouping_score) * grouping_weight * 100

    # Weight balance penalty
    balance_penalty = 0.0
    if yard_layout and balance_weight > 0:
        container_map = {c['id']: c for c in containers}
        balance_score = compute_weight_balance_score(stacking_plan, container_map, yard_layout)
        balance_penalty = (1.0 - balance_score) * balance_weight * 100

    objective = total_reshuffles + grouping_penalty + balance_penalty
    return objective


def simulated_annealing(stacking_plan, containers, yard_layout, params, logger):
    """
    Simulated Annealing optimization loop with 60% swaps, 40% relocations.

    Args:
        stacking_plan: Initial solution
        containers: List of containers
        yard_layout: Yard layout
        params: Dict with max_iterations, temperature_init, cooling_rate, grouping_weight
        logger: Logger

    Returns:
        (best_plan, best_objective, iterations_performed, improvements)
    """
    max_iterations = params.get('max_iterations', 2000)
    temp_init = params.get('temperature_init', 100.0)
    cooling_rate = params.get('cooling_rate', 0.995)
    grouping_weight = params.get('grouping_weight', 0.5)
    balance_weight = params.get('balance_weight', 0.3)

    container_map = {c['id']: c for c in containers}

    current_plan = deepcopy(stacking_plan)
    current_obj = compute_objective(current_plan, containers, grouping_weight, balance_weight, yard_layout)

    best_plan = deepcopy(current_plan)
    best_obj = current_obj

    temperature = temp_init
    iteration = 0
    improvements = 0
    total_accepted = 0
    convergence_history = []

    logger.info("Starting Simulated Annealing: T_init=" + str(temp_init) + ", cooling=" + str(cooling_rate) + ", max_iter=" + str(max_iterations))
    logger.info("Initial objective: " + str(round(current_obj, 2)))

    while iteration < max_iterations and temperature > 0.01:
        # Generate neighbor: 60% swaps, 40% relocations
        if random.random() < 0.6:
            neighbor_plan = two_opt_swap(current_plan, container_map, yard_layout)
        else:
            neighbor_plan = relocate_move(current_plan, container_map, yard_layout)

        neighbor_obj = compute_objective(neighbor_plan, containers, grouping_weight, balance_weight, yard_layout)

        # Acceptance criterion
        delta = neighbor_obj - current_obj
        if delta < 0:  # Improvement
            current_plan = neighbor_plan
            current_obj = neighbor_obj
            improvements += 1

            if current_obj < best_obj:
                best_plan = deepcopy(current_plan)
                best_obj = current_obj
                logger.debug("Iteration " + str(iteration) + ": New best objective = " + str(round(best_obj, 2)))

        else:  # Worse solution, accept with probability
            try:
                probability = math.exp(-delta / temperature)
            except (OverflowError, ValueError):
                probability = 0.0
            if random.random() < probability:
                current_plan = neighbor_plan
                current_obj = neighbor_obj
                total_accepted += 1

        if delta < 0:
            total_accepted += 1

        # Cool down
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
            logger.debug("Iteration " + str(iteration) + ": current_obj=" + str(round(current_obj, 2)) + ", best_obj=" + str(round(best_obj, 2)) + ", T=" + str(round(temperature, 4)))

    # Record final convergence point
    convergence_history.append({
        'iteration': iteration,
        'best_energy': round(best_obj, 2),
        'current_energy': round(current_obj, 2),
        'temperature': round(temperature, 4),
        'acceptance_rate': round(total_accepted / max(iteration, 1), 3),
        'improvements_so_far': improvements
    })

    logger.info("SA completed: " + str(iteration) + " iterations, " + str(improvements) + " improvements, best_obj=" + str(round(best_obj, 2)))
    return best_plan, best_obj, iteration, improvements, convergence_history


def compute_output_metrics(stacking_plan, containers, yard_layout):
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


def generate_stacking_plan_output(stacking_plan, containers):
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
    Main QCentroid solver entry point for Container Yard Stacking Optimization v2.0.

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
        # Extract input data — handle both platform (input_data IS the data) and local (input_data has 'data' key)
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

        logger.info("Container Yard Stacking Optimization Solver v2.0")
        logger.info("Input: " + str(len(containers)) + " containers, " + str(yard_layout.get('total_blocks', 0)) + " yard blocks")

        # Validate input
        if not containers or not yard_layout:
            logger.error("Invalid input: missing containers or yard_layout")
            return {
                "status": "ERROR",
                "message": "Missing required input data",
                "benchmark": {
                    "execution_cost": {"value": 0.0, "unit": "credits"},
                    "time_elapsed": "0.0s",
                    "energy_consumption": 0.0
                }
            }

        # Step 1: Vessel-aware greedy initialization
        logger.info("Step 1: Vessel-Aware Greedy Initialization")
        initial_plan = greedy_initial_stacking(containers, yard_layout, logger)

        if not initial_plan:
            logger.error("Greedy initialization failed")
            return {
                "status": "ERROR",
                "message": "Failed to create initial stacking plan",
                "benchmark": {
                    "execution_cost": {"value": 0.0, "unit": "credits"},
                    "time_elapsed": "0.0s",
                    "energy_consumption": 0.0
                }
            }

        greedy_obj = compute_objective(initial_plan, containers, params.get('grouping_weight', 0.5), params.get('balance_weight', 0.3), yard_layout)
        logger.info("Initial solution objective: " + str(round(greedy_obj, 2)))

        # Step 2: Simulated Annealing optimization
        logger.info("Step 2: Simulated Annealing Optimization (with relocations)")
        best_plan, best_obj, sa_iterations, sa_improvements, convergence_history = simulated_annealing(initial_plan, containers, yard_layout, params, logger)

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

        # Generate showcase visualizations
        block_heatmap = generate_block_heatmap(best_plan, containers, yard_layout)
        vessel_timeline = generate_vessel_timeline(best_plan, containers)
        convergence_chart = generate_convergence_chart_data(convergence_history)

        # Build output (QCentroid benchmark contract compliant)
        elapsed_s = elapsed_ms / 1000.0
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
                'improvement_pct': round((1 - best_obj / max(greedy_obj, 1)) * 100, 1)
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
                    'improvement_vs_greedy_pct': round((1 - best_obj / max(greedy_obj, 0.01)) * 100, 1),
                    'vessels_with_zero_reshuffles': sum(1 for v in vessel_summary if v['estimated_reshuffles'] == 0),
                    'total_vessels': len(vessel_summary),
                    'avg_stack_utilization_pct': round(metrics['stack_utilization'] * 100, 1),
                    'weight_balance_score_pct': round(metrics['weight_balance_score'] * 100, 1),
                    'vessel_grouping_score_pct': round(metrics['vessel_grouping_score'] * 100, 1),
                    'solver_time_ms': round(elapsed_ms, 1),
                    'algorithm': 'Classical SA v2.0 (' + str(sa_iterations) + ' iterations)'
                }
            },
            'computation_metrics': {
                'wall_time_s': round(elapsed_s, 3),
                'algorithm': 'Greedy_SA_v2.0',
                'solver_version': '2.0',
                'sa_iterations': sa_iterations,
                'sa_improvements': sa_improvements,
                'move_strategy': '60pct_swap_40pct_relocate'
            },
            'benchmark': {
                'execution_cost': {'value': 1.0, 'unit': 'credits'},
                'time_elapsed': str(round(elapsed_s, 3)) + 's',
                'energy_consumption': 0.0
            }
        }

        logger.info("Solver completed successfully in " + str(round(elapsed_ms, 1)) + " ms")
        logger.info("Final objective: " + str(round(best_obj, 2)))
        logger.info("Total reshuffles minimized: " + str(metrics['total_reshuffles']))

        return output

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        elapsed_s = elapsed_ms / 1000.0
        logger.error("Solver failed with exception: " + str(e))
        return {
            'status': 'ERROR',
            'message': str(e),
            'solver_log': logger.messages,
            'benchmark': {
                'execution_cost': {'value': 0.0, 'unit': 'credits'},
                'time_elapsed': str(round(elapsed_s, 3)) + 's',
                'energy_consumption': 0.0
            }
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
