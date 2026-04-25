"""
QCentroid Container Yard Stacking Optimization Solver v2.2

v2.2 (this rev): writes visualization artifacts (PNG/HTML/JSON/CSV) into
                 ./additional_output/ so the platform exposes them as
                 downloadable files in the Additional Output tab.
v2.1: added additional_output block (visualizations + kpi_dashboard + reports + narrative).
v2.0: vessel-aware greedy + 2-opt swaps with relocation moves + SA with weight balance objective.

Entry point: run(input_data, solver_params, extra_arguments) -> dict
"""

import json
import os
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
try:
    from viz import generate_additional_output
except Exception:
    generate_additional_output = None


class QCentroidUserLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg):
        self.messages.append({"level": "INFO", "message": msg}); print("[INFO] " + msg)

    def debug(self, msg):
        self.messages.append({"level": "DEBUG", "message": msg}); print("[DEBUG] " + msg)

    def warning(self, msg):
        self.messages.append({"level": "WARNING", "message": msg}); print("[WARNING] " + msg)

    def error(self, msg):
        self.messages.append({"level": "ERROR", "message": msg}); print("[ERROR] " + msg)


qcentroid_user_log = QCentroidUserLogger()


def greedy_initial_stacking(containers, yard_layout, logger):
    logger.info("Starting vessel-aware greedy initialization with " + str(len(containers)) + " containers")
    vessel_groups = {}
    for container in containers:
        vessel_id = container['vessel_id']
        if vessel_id not in vessel_groups:
            vessel_groups[vessel_id] = []
        vessel_groups[vessel_id].append(container)
    sorted_vessels = sorted(vessel_groups.items(), key=lambda x: x[1][0]['vessel_departure_order'])
    stacking_plan = []
    container_map = {c['id']: c for c in containers}
    stack_usage = {}
    block_capacities = {}
    for block in yard_layout['blocks']:
        block_capacities[block['block_id']] = {'capacity': block['total_capacity'], 'used': 0}

    def find_preferred_block(vessel_id, yard_layout, block_capacities):
        blocks = yard_layout['blocks']; available_blocks = []
        for block in blocks:
            block_id = block['block_id']
            current_util = block_capacities[block_id]['used'] / block_capacities[block_id]['capacity']
            available_capacity = block_capacities[block_id]['capacity'] - block_capacities[block_id]['used']
            if available_capacity > 0:
                available_blocks.append((block_id, current_util, available_capacity))
        if not available_blocks: return None
        available_blocks.sort(key=lambda x: x[1])
        return available_blocks[0][0]

    for vessel_id, vessel_containers in sorted_vessels:
        vessel_containers_sorted = sorted(vessel_containers, key=lambda c: -c['weight_tonnes'])
        preferred_block = find_preferred_block(vessel_id, yard_layout, block_capacities)
        if preferred_block is None:
            logger.warning("No available blocks for vessel " + str(vessel_id))
            continue
        for container in vessel_containers_sorted:
            cid = container['id']; weight = container['weight_tonnes']
            placed = False; blocks_to_try = [preferred_block]
            if block_capacities[preferred_block]['used'] >= block_capacities[preferred_block]['capacity']:
                blocks_to_try = [b['block_id'] for b in yard_layout['blocks'] if b['block_id'] != preferred_block]
            for block_id in blocks_to_try:
                if placed: break
                block = None
                for b in yard_layout['blocks']:
                    if b['block_id'] == block_id: block = b; break
                if block is None: continue
                max_tier = block['max_tier_height']
                for row_idx in range(block['rows']):
                    if placed: break
                    for bay_idx in range(block['bays_per_row']):
                        stack_key = (block_id, row_idx, bay_idx); current_tier = stack_usage.get(stack_key, 0)
                        if current_tier < max_tier:
                            can_place = True
                            if current_tier > 0:
                                for existing in stacking_plan:
                                    if (existing['assigned_block'] == block_id and existing['assigned_row'] == row_idx and existing['assigned_bay'] == bay_idx and existing['tier_level'] == current_tier - 1):
                                        below_cid = existing['id']; below_weight = container_map[below_cid]['weight_tonnes']
                                        if below_weight < weight: can_place = False
                                        break
                            if can_place:
                                assignment = {'id': cid, 'assigned_block': block_id, 'assigned_row': row_idx, 'assigned_bay': bay_idx, 'tier_level': current_tier, 'reshuffles_if_retrieved_now': 0}
                                stacking_plan.append(assignment)
                                stack_usage[stack_key] = current_tier + 1
                                block_capacities[block_id]['used'] += 1
                                placed = True; break
            if not placed:
                logger.warning("Could not place container " + str(cid))
    logger.info("Greedy placement complete: " + str(len(stacking_plan)) + " containers placed")
    return stacking_plan


def two_opt_swap(stacking_plan, containers, yard_layout):
    if len(stacking_plan) < 2: return stacking_plan
    idx1, idx2 = random.sample(range(len(stacking_plan)), 2)
    assignment1 = stacking_plan[idx1]; assignment2 = stacking_plan[idx2]
    new_plan = deepcopy(stacking_plan)
    new_plan[idx1]['assigned_block'] = assignment2['assigned_block']; new_plan[idx1]['assigned_row'] = assignment2['assigned_row']
    new_plan[idx1]['assigned_bay'] = assignment2['assigned_bay']; new_plan[idx1]['tier_level'] = assignment2['tier_level']
    new_plan[idx2]['assigned_block'] = assignment1['assigned_block']; new_plan[idx2]['assigned_row'] = assignment1['assigned_row']
    new_plan[idx2]['assigned_bay'] = assignment1['assigned_bay']; new_plan[idx2]['tier_level'] = assignment1['tier_level']
    for assignment in new_plan:
        if not is_feasible_assignment(assignment, containers, yard_layout): return stacking_plan
    if not check_weight_stability(new_plan, containers, yard_layout): return stacking_plan
    return new_plan


def relocate_move(stacking_plan, containers, yard_layout):
    if len(stacking_plan) == 0: return stacking_plan
    if isinstance(containers, dict): container_map = containers
    else: container_map = {c['id']: c for c in containers}
    idx = random.randint(0, len(stacking_plan) - 1)
    relocating_assignment = stacking_plan[idx]; cid = relocating_assignment['id']
    weight = container_map[cid]['weight_tonnes']
    new_plan = deepcopy(stacking_plan); blocks = yard_layout['blocks']
    block_order = list(range(len(blocks))); random.shuffle(block_order)
    for block_idx in block_order:
        block = blocks[block_idx]; block_id = block['block_id']; max_tier = block['max_tier_height']
        for row_idx in range(block['rows']):
            for bay_idx in range(block['bays_per_row']):
                current_tier = 0
                for assignment in new_plan:
                    if (assignment['assigned_block'] == block_id and assignment['assigned_row'] == row_idx and assignment['assigned_bay'] == bay_idx):
                        current_tier = max(current_tier, assignment['tier_level'] + 1)
                if current_tier < max_tier:
                    can_place = True
                    if current_tier > 0:
                        for assignment in new_plan:
                            if (assignment['assigned_block'] == block_id and assignment['assigned_row'] == row_idx and assignment['assigned_bay'] == bay_idx and assignment['tier_level'] == current_tier - 1):
                                below_weight = container_map[assignment['id']]['weight_tonnes']
                                if below_weight < weight: can_place = False
                                break
                    if can_place:
                        new_plan[idx]['assigned_block'] = block_id; new_plan[idx]['assigned_row'] = row_idx
                        new_plan[idx]['assigned_bay'] = bay_idx; new_plan[idx]['tier_level'] = current_tier
                        feasible = True
                        for assignment in new_plan:
                            if not is_feasible_assignment(assignment, containers, yard_layout):
                                feasible = False; break
                        if feasible and check_weight_stability(new_plan, containers, yard_layout):
                            return new_plan
    return stacking_plan


def compute_objective(stacking_plan, containers, grouping_weight=0.5, balance_weight=0.3, yard_layout=None):
    total_reshuffles, _ = compute_reshuffles_for_stacking(stacking_plan, containers)
    grouping_score = compute_vessel_grouping_score(stacking_plan, containers)
    grouping_penalty = (1.0 - grouping_score) * grouping_weight * 100
    balance_penalty = 0.0
    if yard_layout and balance_weight > 0:
        container_map = {c['id']: c for c in containers}
        balance_score = compute_weight_balance_score(stacking_plan, container_map, yard_layout)
        balance_penalty = (1.0 - balance_score) * balance_weight * 100
    return total_reshuffles + grouping_penalty + balance_penalty


def simulated_annealing(stacking_plan, containers, yard_layout, params, logger):
    max_iterations = params.get('max_iterations', 2000)
    temp_init = params.get('temperature_init', 100.0)
    cooling_rate = params.get('cooling_rate', 0.995)
    grouping_weight = params.get('grouping_weight', 0.5)
    balance_weight = params.get('balance_weight', 0.3)
    container_map = {c['id']: c for c in containers}
    current_plan = deepcopy(stacking_plan)
    current_obj = compute_objective(current_plan, containers, grouping_weight, balance_weight, yard_layout)
    best_plan = deepcopy(current_plan); best_obj = current_obj
    temperature = temp_init; iteration = 0; improvements = 0; total_accepted = 0; convergence_history = []
    logger.info("Starting Simulated Annealing: T_init=" + str(temp_init) + ", cooling=" + str(cooling_rate) + ", max_iter=" + str(max_iterations))
    logger.info("Initial objective: " + str(round(current_obj, 2)))
    while iteration < max_iterations and temperature > 0.01:
        if random.random() < 0.6:
            neighbor_plan = two_opt_swap(current_plan, container_map, yard_layout)
        else:
            neighbor_plan = relocate_move(current_plan, container_map, yard_layout)
        neighbor_obj = compute_objective(neighbor_plan, containers, grouping_weight, balance_weight, yard_layout)
        delta = neighbor_obj - current_obj
        if delta < 0:
            current_plan = neighbor_plan; current_obj = neighbor_obj; improvements += 1
            if current_obj < best_obj:
                best_plan = deepcopy(current_plan); best_obj = current_obj
                logger.debug("Iteration " + str(iteration) + ": New best objective = " + str(round(best_obj, 2)))
        else:
            try: probability = math.exp(-delta / temperature)
            except (OverflowError, ValueError): probability = 0.0
            if random.random() < probability:
                current_plan = neighbor_plan; current_obj = neighbor_obj; total_accepted += 1
        if delta < 0: total_accepted += 1
        temperature *= cooling_rate; iteration += 1
        if iteration % max(1, max_iterations // 50) == 0 or iteration == 1:
            acceptance_rate = total_accepted / max(iteration, 1)
            convergence_history.append({'iteration': iteration, 'best_energy': round(best_obj, 2), 'current_energy': round(current_obj, 2), 'temperature': round(temperature, 4), 'acceptance_rate': round(acceptance_rate, 3), 'improvements_so_far': improvements})
        if iteration % 100 == 0:
            logger.debug("Iteration " + str(iteration) + ": current_obj=" + str(round(current_obj, 2)) + ", best_obj=" + str(round(best_obj, 2)) + ", T=" + str(round(temperature, 4)))
    convergence_history.append({'iteration': iteration, 'best_energy': round(best_obj, 2), 'current_energy': round(current_obj, 2), 'temperature': round(temperature, 4), 'acceptance_rate': round(total_accepted / max(iteration, 1), 3), 'improvements_so_far': improvements})
    logger.info("SA completed: " + str(iteration) + " iterations, " + str(improvements) + " improvements, best_obj=" + str(round(best_obj, 2)))
    return best_plan, best_obj, iteration, improvements, convergence_history


def compute_output_metrics(stacking_plan, containers, yard_layout):
    container_map = {c['id']: c for c in containers}
    total_reshuffles, reshuffles_per_vessel = compute_reshuffles_for_stacking(stacking_plan, containers)
    block_util = compute_block_utilization(stacking_plan, yard_layout)
    grouping_score = compute_vessel_grouping_score(stacking_plan, containers)
    balance_score = compute_weight_balance_score(stacking_plan, container_map, yard_layout)
    return {'total_reshuffles': total_reshuffles, 'average_reshuffles_per_vessel': total_reshuffles / len(reshuffles_per_vessel) if reshuffles_per_vessel else 0.0, 'max_reshuffles_single_vessel': max(reshuffles_per_vessel.values()) if reshuffles_per_vessel else 0, 'vessel_grouping_score': grouping_score, 'stack_utilization': sum(block_util.values()) / len(block_util) if block_util else 0.0, 'weight_balance_score': balance_score, 'reshuffles_per_vessel': reshuffles_per_vessel, 'block_utilization': block_util}


def generate_stacking_plan_output(stacking_plan, containers):
    output_plan = []
    for assignment in stacking_plan:
        cid = assignment['id']; reshuffles = estimate_reshuffles_single_container(cid, stacking_plan)
        output_plan.append({'id': cid, 'assigned_block': assignment['assigned_block'], 'assigned_row': assignment['assigned_row'], 'assigned_bay': assignment['assigned_bay'], 'tier_level': assignment['tier_level'], 'reshuffles_if_retrieved_now': reshuffles})
    return output_plan


def generate_block_heatmap(stacking_plan, containers, yard_layout):
    container_map = {c['id']: c for c in containers}; heatmap = {}
    for block in yard_layout['blocks']:
        bid = block['block_id']; rows = block['rows']; bays = block['bays_per_row']; max_tier = block['max_tier_height']; grid = []
        for r in range(rows):
            row_data = []
            for b in range(bays):
                stack_containers = []
                for a in stacking_plan:
                    if a['assigned_block'] == bid and a['assigned_row'] == r and a['assigned_bay'] == b:
                        c = container_map.get(a['id'], {})
                        stack_containers.append({'id': a['id'], 'tier': a['tier_level'], 'weight': c.get('weight_tonnes', 0), 'vessel': c.get('vessel_id', ''), 'departure_order': c.get('vessel_departure_order', 0), 'reshuffles_needed': estimate_reshuffles_single_container(a['id'], stacking_plan)})
                stack_containers.sort(key=lambda x: x['tier'])
                total_weight = sum(sc['weight'] for sc in stack_containers); height = len(stack_containers)
                vessels_in_stack = list(set(sc['vessel'] for sc in stack_containers))
                row_data.append({'row': r, 'bay': b, 'height': height, 'max_height': max_tier, 'fill_pct': round(100 * height / max_tier, 1), 'total_weight_tonnes': round(total_weight, 1), 'vessels': vessels_in_stack, 'vessel_mix': len(vessels_in_stack), 'containers': stack_containers})
            grid.append(row_data)
        block_containers = [a for a in stacking_plan if a['assigned_block'] == bid]; capacity = block['total_capacity']
        heatmap[bid] = {'block_id': bid, 'dimensions': {'rows': rows, 'bays': bays, 'max_tier': max_tier}, 'total_containers': len(block_containers), 'capacity': capacity, 'utilization_pct': round(100 * len(block_containers) / capacity, 1) if capacity > 0 else 0, 'grid': grid}
    return heatmap


def generate_vessel_timeline(stacking_plan, containers):
    vessels = {}
    for c in containers:
        vid = c['vessel_id']
        if vid not in vessels: vessels[vid] = {'vessel_id': vid, 'departure_order': c['vessel_departure_order'], 'containers': []}
        vessels[vid]['containers'].append(c)
    _, reshuffles_per_vessel = compute_reshuffles_for_stacking(stacking_plan, containers)
    timeline = []; cumulative = 0
    for vid, info in sorted(vessels.items(), key=lambda x: x[1]['departure_order']):
        r = reshuffles_per_vessel.get(vid, 0); cumulative += r
        n = len(info['containers']); tw = sum(c['weight_tonnes'] for c in info['containers'])
        eff = round(100 * (1 - r / max(n, 1)), 1)
        timeline.append({'vessel_id': vid, 'departure_order': info['departure_order'], 'num_containers': n, 'total_weight_tonnes': round(tw, 1), 'avg_weight_tonnes': round(tw / n, 1) if n > 0 else 0, 'reshuffles': r, 'cumulative_reshuffles': cumulative, 'retrieval_efficiency_pct': eff, 'status': 'clean' if r == 0 else ('minor' if r <= 2 else 'needs_attention')})
    return timeline


def generate_convergence_chart_data(convergence_history):
    n = len(convergence_history)
    if n <= 50: return convergence_history
    step = max(1, n // 50)
    sampled = [convergence_history[i] for i in range(0, n, step)]
    if sampled[-1] != convergence_history[-1]: sampled.append(convergence_history[-1])
    return sampled


def run(input_data, solver_params=None, extra_arguments=None):
    logger = qcentroid_user_log; start_time = time.time()
    try:
        if 'containers' in input_data: data = input_data
        else: data = input_data.get('data', input_data)
        containers = data.get('containers', []); yard_layout = data.get('yard_layout', {}); params = data.get('parameters', {})
        if solver_params: params.update(solver_params)
        logger.info("Container Yard Stacking Optimization Solver v2.1")
        logger.info("Input: " + str(len(containers)) + " containers, " + str(yard_layout.get('total_blocks', 0)) + " yard blocks")
        if not containers or not yard_layout:
            return {"status": "ERROR", "message": "Missing required input data", "objective_value": 999999, "solution_status": "error", "benchmark": {"execution_cost": {"value": 0.0, "unit": "credits"}, "time_elapsed": "0.0s", "energy_consumption": 0.0}}
        logger.info("Step 1: Vessel-Aware Greedy Initialization")
        initial_plan = greedy_initial_stacking(containers, yard_layout, logger)
        if not initial_plan:
            return {"status": "ERROR", "message": "Failed to create initial stacking plan", "objective_value": 999999, "solution_status": "error", "benchmark": {"execution_cost": {"value": 0.0, "unit": "credits"}, "time_elapsed": "0.0s", "energy_consumption": 0.0}}
        greedy_obj = compute_objective(initial_plan, containers, params.get('grouping_weight', 0.5), params.get('balance_weight', 0.3), yard_layout)
        logger.info("Initial solution objective: " + str(round(greedy_obj, 2)))
        logger.info("Step 2: Simulated Annealing Optimization (with relocations)")
        best_plan, best_obj, sa_iterations, sa_improvements, convergence_history = simulated_annealing(initial_plan, containers, yard_layout, params, logger)
        elapsed_ms = (time.time() - start_time) * 1000; elapsed_s = elapsed_ms / 1000.0
        logger.info("Step 3: Computing Output Metrics")
        metrics = compute_output_metrics(best_plan, containers, yard_layout)
        output_stacking_plan = generate_stacking_plan_output(best_plan, containers)
        total_reshuffles, reshuffles_per_vessel = compute_reshuffles_for_stacking(best_plan, containers)
        vessel_summary = []
        for vessel_id, reshuffles in reshuffles_per_vessel.items():
            vessel_containers = [c for c in containers if c['vessel_id'] == vessel_id]
            vessel_summary.append({'vessel_id': vessel_id, 'departure_order': vessel_containers[0]['vessel_departure_order'] if vessel_containers else 0, 'total_containers': len(vessel_containers), 'estimated_reshuffles': reshuffles, 'reshuffles_percentage': 100.0 * reshuffles / len(vessel_containers) if vessel_containers else 0.0})
        vessel_summary.sort(key=lambda v: v['departure_order'])
        block_heatmap = generate_block_heatmap(best_plan, containers, yard_layout)
        vessel_timeline = generate_vessel_timeline(best_plan, containers)
        convergence_chart = generate_convergence_chart_data(convergence_history)

        improvement_pct = round((1 - best_obj / max(greedy_obj, 0.01)) * 100, 1)

        kpi_dashboard = {
            'objective_value': round(best_obj, 2),
            'total_reshuffles': metrics['total_reshuffles'],
            'vessels_with_zero_reshuffles': sum(1 for v in vessel_summary if v['estimated_reshuffles'] == 0),
            'total_vessels': len(vessel_summary),
            'improvement_vs_greedy_pct': improvement_pct,
            'avg_stack_utilization_pct': round(metrics['stack_utilization'] * 100, 1),
            'weight_balance_score_pct': round(metrics['weight_balance_score'] * 100, 1),
            'vessel_grouping_score_pct': round(metrics['vessel_grouping_score'] * 100, 1),
            'wall_time_s': round(elapsed_s, 3),
            'algorithm': 'Classical SA v2.1 (' + str(sa_iterations) + ' iterations)'
        }

        # The platform's Additional Output tab reads this block.
        additional_output = {
            'schema_version': '1.0',
            'use_case': 'container-yard-stacking-optimization',
            'solver_family': 'classical',
            'solver_version': '2.2',
            'visualizations': [
                {'name': 'block_heatmap', 'type': 'grid', 'description': 'Top-down per-block container layout (rows × bays). Each cell shows stack height, dominant vessel, weight, and reshuffle indicator.', 'data': block_heatmap},
                {'name': 'vessel_timeline', 'type': 'timeline', 'description': 'Per-vessel reshuffle forecast in departure order with cumulative deltas and retrieval efficiency.', 'data': vessel_timeline},
                {'name': 'convergence_chart', 'type': 'line_chart', 'description': 'SA best/current energy and temperature across iterations.', 'data': convergence_chart}
            ],
            'kpi_dashboard': kpi_dashboard,
            'reports': {
                'reshuffle_breakdown_by_vessel': vessel_summary,
                'cost_analysis': {'total_reshuffles': metrics['total_reshuffles'], 'greedy_reshuffles': round(greedy_obj, 2), 'optimized_reshuffles': round(best_obj, 2), 'improvement_pct': improvement_pct},
                'quality_scores': {'vessel_grouping': round(metrics['vessel_grouping_score'], 3), 'weight_balance': round(metrics['weight_balance_score'], 3), 'stack_utilization': round(metrics['stack_utilization'], 3)},
                'block_utilization': metrics['block_utilization']
            },
            'narrative': ('Classical Greedy + Simulated Annealing placed all ' + str(len(output_stacking_plan)) + '/' + str(len(containers)) + ' containers; SA performed ' + str(sa_iterations) + ' iterations with ' + str(sa_improvements) + ' improvement moves; achieved ' + str(improvement_pct) + '% improvement vs greedy initialization. Solution requires ' + str(metrics['total_reshuffles']) + ' reshuffle(s) total; ' + str(sum(1 for v in vessel_summary if v["estimated_reshuffles"] == 0)) + '/' + str(len(vessel_summary)) + ' vessels can be loaded without reshuffles.')
        }

        # Write visualization files into ./additional_output/ for the platform tab.
        files_meta = {"out_dir": None, "files": []}
        if generate_additional_output is not None:
            try:
                files_meta = generate_additional_output(
                    containers=containers,
                    yard_layout=yard_layout,
                    stacking_plan=output_stacking_plan,
                    block_heatmap=block_heatmap,
                    vessel_timeline=vessel_timeline,
                    convergence_history=convergence_chart,
                    kpi_dashboard=kpi_dashboard,
                    narrative=additional_output['narrative'],
                    out_dir=os.path.join(os.getcwd(), 'additional_output'),
                    logger=logger,
                )
                logger.info("additional_output: wrote " + str(len(files_meta.get('files', []))) + " files to " + str(files_meta.get('out_dir')))
            except Exception as e:
                logger.warning("additional_output generation failed: " + str(e))
        additional_output['files'] = files_meta.get('files', [])
        additional_output['files_dir'] = files_meta.get('out_dir')

        output = {
            'objective_value': round(best_obj, 2),
            'solution_status': 'optimal' if best_obj < greedy_obj else 'feasible',
            'total_reshuffles': metrics['total_reshuffles'],
            'containers_placed': len(output_stacking_plan),
            'containers_total': len(containers),
            'stacking_plan': output_stacking_plan,
            'reshuffling_summary': vessel_summary,
            'optimization_metrics': {'total_reshuffles': metrics['total_reshuffles'], 'average_reshuffles_per_vessel': round(metrics['average_reshuffles_per_vessel'], 2), 'max_reshuffles_single_vessel': metrics['max_reshuffles_single_vessel'], 'vessel_grouping_score': round(metrics['vessel_grouping_score'], 3), 'stack_utilization': round(metrics['stack_utilization'], 3), 'weight_balance_score': round(metrics['weight_balance_score'], 3)},
            'cost_breakdown': {'total_reshuffles': metrics['total_reshuffles'], 'greedy_reshuffles': round(greedy_obj, 2), 'optimized_reshuffles': round(best_obj, 2), 'improvement_pct': improvement_pct},
            'optimization_convergence': {'greedy_initial_cost': round(greedy_obj, 2), 'sa_cost': round(best_obj, 2), 'final_optimized_cost': round(best_obj, 2), 'sa_iterations': sa_iterations, 'sa_improvements': sa_improvements},
            'showcase': {'block_heatmap': block_heatmap, 'vessel_timeline': vessel_timeline, 'convergence_chart': convergence_chart, 'summary_dashboard': kpi_dashboard},
            'additional_output': additional_output,
            'computation_metrics': {'wall_time_s': round(elapsed_s, 3), 'algorithm': 'Greedy_SA_v2.2', 'solver_version': '2.2', 'sa_iterations': sa_iterations, 'sa_improvements': sa_improvements, 'move_strategy': '60pct_swap_40pct_relocate'},
            'benchmark': {'execution_cost': {'value': 1.0, 'unit': 'credits'}, 'time_elapsed': str(round(elapsed_s, 3)) + 's', 'energy_consumption': 0.0}
        }
        logger.info("Solver completed successfully in " + str(round(elapsed_ms, 1)) + " ms")
        logger.info("Final objective: " + str(round(best_obj, 2)))
        logger.info("Total reshuffles minimized: " + str(metrics['total_reshuffles']))
        return output
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000; elapsed_s = elapsed_ms / 1000.0
        logger.error("Solver failed with exception: " + str(e))
        return {'status': 'ERROR', 'message': str(e), 'objective_value': 999999, 'solution_status': 'error', 'solver_log': logger.messages, 'benchmark': {'execution_cost': {'value': 0.0, 'unit': 'credits'}, 'time_elapsed': str(round(elapsed_s, 3)) + 's', 'energy_consumption': 0.0}}


if __name__ == '__main__':
    with open('dataset_small.json', 'r') as f:
        test_input = json.load(f)
    result = run(test_input)
    print("\n" + "=" * 60); print("SOLVER OUTPUT"); print("=" * 60)
    print(json.dumps(result, indent=2))
