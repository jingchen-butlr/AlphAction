"""
Custom Task Action Filter
Filter AVA predictions to show only task-specific actions:
- fall-down, lay-down, sleep, sit, stand, walk, run
"""

# Mapping from custom task actions to AVA categories
CUSTOM_TASK_MAPPING = {
    'fall-down': {
        'ava_index': 4,
        'ava_name': 'fall down',
        'detection_difficulty': 'Hard',
        'expected_accuracy': '60-75%'
    },
    'lay-down': {
        'ava_index': 7,
        'ava_name': 'lie/sleep',
        'detection_difficulty': 'Medium',
        'expected_accuracy': '75-85%'
    },
    'sleep': {
        'ava_index': 7,
        'ava_name': 'lie/sleep',
        'detection_difficulty': 'Medium',
        'expected_accuracy': '75-85%'
    },
    'run': {
        'ava_index': 9,
        'ava_name': 'run/jog',
        'detection_difficulty': 'Easy',
        'expected_accuracy': '85-90%'
    },
    'sit': {
        'ava_index': 10,
        'ava_name': 'sit',
        'detection_difficulty': 'Very Easy',
        'expected_accuracy': '90-95%'
    },
    'stand': {
        'ava_index': 11,
        'ava_name': 'stand',
        'detection_difficulty': 'Very Easy',
        'expected_accuracy': '90-95%'
    },
    'walk': {
        'ava_index': 13,
        'ava_name': 'walk',
        'detection_difficulty': 'Easy',
        'expected_accuracy': '85-90%'
    },
}

# Get unique AVA indices for filtering
TARGET_AVA_INDICES = sorted(list(set([v['ava_index'] for v in CUSTOM_TASK_MAPPING.values()])))
print(f"Target AVA indices: {TARGET_AVA_INDICES}")
# Output: [4, 7, 9, 10, 11, 13]

# AVA category names (for reference)
TARGET_AVA_NAMES = [
    'fall down',   # index 4
    'lie/sleep',   # index 7
    'run/jog',     # index 9
    'sit',         # index 10
    'stand',       # index 11
    'walk',        # index 13
]


def filter_predictions(scores, threshold=0.5):
    """
    Filter AVA predictions to show only custom task actions
    
    Args:
        scores: Tensor of shape (N, 80) where N is number of persons
        threshold: Confidence threshold (default: 0.5)
    
    Returns:
        filtered_actions: List of detected actions per person
    """
    import torch
    
    if isinstance(scores, list):
        scores = torch.stack(scores)
    
    filtered_results = []
    
    for person_idx, person_scores in enumerate(scores):
        person_actions = []
        
        # Check each target index
        for custom_action, info in CUSTOM_TASK_MAPPING.items():
            ava_idx = info['ava_index']
            ava_name = info['ava_name']
            score = person_scores[ava_idx].item()
            
            if score >= threshold:
                person_actions.append({
                    'custom_name': custom_action,
                    'ava_name': ava_name,
                    'score': score,
                    'ava_index': ava_idx
                })
        
        filtered_results.append(person_actions)
    
    return filtered_results


def print_filtered_results(filtered_results, person_ids=None):
    """
    Print filtered action detection results
    
    Args:
        filtered_results: Output from filter_predictions()
        person_ids: Optional list of person IDs
    """
    print("\n" + "="*60)
    print("Custom Task Action Detection Results")
    print("="*60)
    
    for person_idx, actions in enumerate(filtered_results):
        person_id = person_ids[person_idx] if person_ids else person_idx
        print(f"\nPerson {person_id}:")
        
        if not actions:
            print("  No target actions detected")
        else:
            # Sort by score
            actions = sorted(actions, key=lambda x: x['score'], reverse=True)
            for action in actions:
                print(f"  {action['custom_name']:12s} ({action['ava_name']:12s}) - {action['score']:.2%}")


def create_custom_visualizer_filter():
    """
    Returns a filter function that can be used in visualizer.py
    to only show custom task actions
    """
    def filter_func(category_idx, category_name):
        """Returns True if this category should be displayed"""
        return category_idx in TARGET_AVA_INDICES
    
    return filter_func


# Example usage
if __name__ == "__main__":
    import torch
    
    print("\n" + "="*60)
    print("Custom Task Action Filter - Configuration")
    print("="*60)
    
    print("\n📋 Target Actions:")
    for custom_name, info in CUSTOM_TASK_MAPPING.items():
        print(f"  {custom_name:12s} → {info['ava_name']:12s} (index {info['ava_index']:2d}) - {info['detection_difficulty']}")
    
    print(f"\n🎯 Unique AVA indices to monitor: {TARGET_AVA_INDICES}")
    print(f"   (6 out of 80 categories = {6/80*100:.1f}% of model output)")
    
    print("\n💡 Integration Options:")
    print("   1. Filter scores in action_predictor.py")
    print("   2. Filter display in visualizer.py")
    print("   3. Post-process results for custom analytics")
    
    # Example: Simulate some predictions
    print("\n" + "="*60)
    print("Example: Filtering Sample Predictions")
    print("="*60)
    
    # Create fake predictions (2 persons, 80 categories each)
    fake_scores = torch.rand(2, 80) * 0.3  # Low baseline
    
    # Boost scores for target actions (simulate detections)
    fake_scores[0, 11] = 0.95  # Person 0: stand
    fake_scores[0, 13] = 0.75  # Person 0: walk
    fake_scores[1, 10] = 0.88  # Person 1: sit
    fake_scores[1, 7] = 0.65   # Person 1: lie/sleep
    
    # Filter to custom task actions
    filtered = filter_predictions(fake_scores, threshold=0.5)
    print_filtered_results(filtered, person_ids=[101, 102])
    
    print("\n" + "="*60)
    print("✅ Custom task filter ready for integration!")
    print("="*60)

