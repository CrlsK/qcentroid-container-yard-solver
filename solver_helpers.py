"""
Helper functions for container yard stacking optimization.
Includes reshuffling cost computation, feasibility checks, and metrics.
"""

import math
try:
    from typing import Dict, List, Tuple, Any
except ImportError:
    pass


def compute_reshuffles_for_stacking(stacking_plan, containers):
    """
    Compute total reshuffles needed given a stacking plan.
    For each vessel, simulate the retrieval process.
    """
    vessels = {}
    for container in containers:
        vessel_id = container['vessel_id']
        if vessel_id not in vessels:
            vessels[vessel_id] = {
                'departure_order': container['vessel_departure_order'],
                'containers': []
            }
        vessels[vessel_id]['containers'].append(container)

    sorted_vessels = sorted(vessels.items(), key=lambda x: x[1]['departure_order'])

    stack_location = {}
    for assignment in stacking_plan:
        cid = assignment['id']
        stack_location[cid] = {
            'block': assignment['assigned_block'],
            'row': assignment['assigned_row'],
            'bay': assignment['assigned_bay'],
            'tier': assignment['tier_level']
        }

    total_reshuffles = 0
    reshuffles_per_vessel = {}
    globally_removed = set()

    for vessel_id, vessel_info in sorted_vessels:
        vessel_containers = vessel_info['containers']
        vessel_containers_sorted = sorted(vessel_containers, key=lambda c: c['priority'])
        vessel_reshuffles = 0

        for target_container in vessel_containers_sorted:
            target_id = target_container['id']
            loc = stack_location[target_id]

            containers_above = 0
            for other_assignment in stacking_plan:
                if (other_assignment['assigned_block'] == loc['block'] and
                    other_assignment['assigned_row'] == loc['row'] and
                    other_assignment['assigned_bay'] == loc['bay'] and
                    other_assignment['tier_level'] > loc['tier']):
                    other_id = other_assignment['id']
                    if other_id not in globally_removed:
                        containers_above += 1

            vessel_reshuffles += containers_above
            globally_removed.add(target_id)

        total_reshuffles += vessel_reshuffles
        reshuffles_per_vessel[vessel_id] = vessel_reshuffles

    return total_reshuffles, reshuffles_per_vessel


def is_feasible_assignment(assignment, containers, yard_layout):
    """
    Check if an assignment violates any hard constraints.
    """
    block_id = assignment['assigned_block']
    row_idx = assignment['assigned_row']
    bay_idx = assignment['assigned_bay']
    tier = assignment['tier_level']

    block = None
    for b in yard_layout['blocks']:
        if b['block_id'] == block_id:
            block = b
            break

    if block is None:
        return False

    if row_idx < 0 or row_idx >= block['rows']:
        return False
    if bay_idx < 0 or bay_idx >= block['bays_per_row']:
        return False
    if tier < 0 or tier >= block['max_tier_height']:
        return False

    return True


def check_weight_stability(stacking_plan, containers, yard_layout):
    """
    Check if weight stability constraints are satisfied.
    Heavier containers must be below lighter ones.
    """
    stacks = {}
    for assignment in stacking_plan:
        stack_key = (assignment['assigned_block'], assignment['assigned_row'], assignment['assigned_bay'])
        tier = assignment['tier_level']
        container_id = assignment['id']
        weight = containers[container_id]['weight_tonnes']

        if stack_key not in stacks:
            stacks[stack_key] = []
        stacks[stack_key].append((tier, weight))

    for stack_key, tier_list in stacks.items():
        tier_list.sort()
        for i in range(len(tier_list) - 1):
            tier_below, weight_below = tier_list[i]
            tier_above, weight_above = tier_list[i + 1]
            if weight_below < weight_above:
                return False

    return True


def compute_block_utilization(stacking_plan, yard_layout):
    """
    Compute utilization percentage for each block.
    """
    block_counts = {}
    for assignment in stacking_plan:
        block_id = assignment['assigned_block']
        if block_id not in block_counts:
            block_counts[block_id] = 0
        block_counts[block_id] += 1

    utilization = {}
    for block in yard_layout['blocks']:
        block_id = block['block_id']
        capacity = block['total_capacity']
        count = block_counts.get(block_id, 0)
        utilization[block_id] = count / capacity if capacity > 0 else 0.0

    return utilization


def compute_vessel_grouping_score(stacking_plan, containers):
    """
    Compute a score (0-1) measuring how well containers of the same vessel are grouped.
    """
    vessel_assignments = {}
    for assignment in stacking_plan:
        cid = assignment['id']
        vessel_id = None
        for c in containers:
            if c['id'] == cid:
                vessel_id = c['vessel_id']
                break

        if vessel_id not in vessel_assignments:
            vessel_assignments[vessel_id] = []

        loc = (assignment['assigned_block'], assignment['assigned_row'], assignment['assigned_bay'])
        vessel_assignments[vessel_id].append(loc)

    total_distance = 0
    total_pairs = 0

    for vessel_id, locations in vessel_assignments.items():
        if len(locations) <= 1:
            continue
        vessel_distance = 0
        for i in range(len(locations)):
            for j in range(i + 1, len(locations)):
                loc1 = locations[i]
                loc2 = locations[j]
                dist = abs(ord(loc1[0][-1]) - ord(loc2[0][-1]))
                dist += abs(loc1[1] - loc2[1])
                dist += abs(loc1[2] - loc2[2])
                vessel_distance += dist
        total_distance += vessel_distance
        total_pairs += len(locations) * (len(locations) - 1) / 2

    if total_pairs == 0:
        return 1.0

    avg_distance = total_distance / total_pairs
    max_distance = 20.0
    grouping_score = max(0, 1.0 - (avg_distance / max_distance))
    return grouping_score


def compute_weight_balance_score(stacking_plan, containers, yard_layout):
    """
    Compute a score (0-1) measuring how evenly weight is distributed across blocks.
    """
    block_weights = {}
    for block in yard_layout['blocks']:
        block_weights[block['block_id']] = 0.0

    for assignment in stacking_plan:
        block_id = assignment['assigned_block']
        container_id = assignment['id']
        weight = containers[container_id]['weight_tonnes']
        block_weights[block_id] += weight

    weights = list(block_weights.values())
    if len(weights) == 0 or sum(weights) == 0:
        return 1.0

    mean_weight = sum(weights) / len(weights)
    if mean_weight == 0:
        return 1.0

    variance = sum((w - mean_weight) ** 2 for w in weights) / len(weights)
    std_dev = math.sqrt(variance)
    cv = std_dev / mean_weight

    balance_score = max(0, 1.0 - cv)
    return balance_score


def estimate_reshuffles_single_container(container_id, stacking_plan):
    """
    Estimate reshuffles needed to retrieve this single container.
    Returns the number of containers stacked above it.
    """
    my_location = None
    for assignment in stacking_plan:
        if assignment['id'] == container_id:
            my_location = assignment
            break

    if my_location is None:
        return 0

    count = 0
    stack_key = (my_location['assigned_block'], my_location['assigned_row'], my_location['assigned_bay'])

    for assignment in stacking_plan:
        if (assignment['assigned_block'] == stack_key[0] and
            assignment['assigned_row'] == stack_key[1] and
            assignment['assigned_bay'] == stack_key[2] and
            assignment['tier_level'] > my_location['tier_level']):
            count += 1

    return count
